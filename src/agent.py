"""A tool-calling agent that picks the right tool for each question.

Tools:
- search_articles:  Q&A over the FAISS index (the default for factual questions)
- extract_insights: key insights for one article
- compare_articles: what each article says about the same company or topic
"""

from dataclasses import dataclass, field

import config  # first, so USER_AGENT is set before langchain_community loads

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.insights import ArticleInsights, extract_insights
from src.llm import get_chat_model
from src.qa_chain import answer_question

SYSTEM_PROMPT = """You are a financial news research assistant. The user has loaded a few \
news articles. Answer only from what your tools return; never use outside knowledge.

Choosing a tool:
- search_articles: the default. Use it for any factual question (figures, dates, who/what/why).
- extract_insights: only when the user asks for insights, a summary, sentiment or the \
investor takeaway of an article or company. Call it once per article asked about.
- compare_articles: only when the user asks to compare companies or articles, or asks \
how they differ or what they have in common. It already reads every article, so don't \
follow it with search_articles unless it says something is missing.

If a tool says the information isn't in the articles, tell the user that plainly. \
Keep answers concise, keep figures exactly as the tools give them, and use short \
bullet points or a small table when comparing.

Loaded articles (when the user says "each" or "all" articles, cover every one of these):
{article_list}"""

COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Compare what each news article below says about the topic. Use only these "
     "excerpts and copy figures exactly. Format:\n"
     "1. One short bullet list per article (use the article title) with its key points.\n"
     "2. 'Similarities' and 'Differences' bullets.\n"
     "If an article says nothing about the topic, say so for that article."),
    ("human", "Topic: {topic}\n\n{excerpts}"),
])


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # [{"url": ..., "title": ...}]


def format_insights(ins: ArticleInsights, title: str) -> str:
    lines = [f"Insights for: {title}", f"Summary: {ins.summary}"]
    lines.append("Companies: " + ", ".join(f"{c.name} ({c.ticker or '-'})" for c in ins.companies))
    lines.append("Key numbers:")
    lines += [f"- {n.label}: {n.value}" + (f" ({n.change})" if n.change else "") for n in ins.key_numbers]
    lines.append(f"Sentiment: {ins.sentiment} - {ins.sentiment_reason}")
    lines.append(f"Investor takeaway: {ins.investor_takeaway}")
    return "\n".join(lines)


def find_article(articles: list[Document], query: str) -> Document | None:
    """Find an article by URL, or by the words in its title / opening text."""
    query = query.strip().lower()
    for a in articles:
        if a.metadata["source"].lower() == query:
            return a
    words = [w for w in query.replace("'s", "").split() if len(w) > 2]
    best, best_score = None, 0
    for a in articles:
        haystack = (a.metadata.get("title", "") + " " + a.metadata["source"] + " " + a.page_content[:300]).lower()
        score = sum(w in haystack for w in words)
        if score > best_score:
            best, best_score = a, score
    return best


class ResearchAgent:
    def __init__(self, store: FAISS, articles: list[Document]):
        self.store = store
        self.articles = articles
        self._sources: list[dict] = []
        article_list = "\n".join(
            f"- {a.metadata.get('title') or a.metadata['source']}" for a in articles
        )
        self.graph = create_agent(
            model=get_chat_model(),
            tools=self._make_tools(),
            system_prompt=SYSTEM_PROMPT.format(article_list=article_list),
        )

    def _add_source(self, url: str, title: str = "") -> None:
        if url not in {s["url"] for s in self._sources}:
            self._sources.append({"url": url, "title": title})

    def _make_tools(self):
        @tool
        def search_articles(query: str) -> str:
            """Search the loaded news articles and answer a factual question from them.
            `query` must be a complete, standalone question (name the company)."""
            r = answer_question(query, self.store)
            for s in r.sources:
                self._add_source(s["url"], s["title"])
            sources = "\n".join(f"- {s['title'] or s['url']}" for s in r.sources)
            return r.answer + (f"\nSources:\n{sources}" if sources else "")

        @tool
        def extract_insights_tool(article: str) -> str:
            """Get key insights (summary, companies, key numbers, sentiment, investor
            takeaway) for ONE loaded article. `article` is the company name, a few words
            of the headline, or the URL."""
            match = find_article(self.articles, article)
            if match is None:
                titles = "\n".join(f"- {a.metadata.get('title') or a.metadata['source']}" for a in self.articles)
                return f"No loaded article matches '{article}'. Loaded articles:\n{titles}"
            title = match.metadata.get("title") or match.metadata["source"]
            self._add_source(match.metadata["source"], match.metadata.get("title", ""))
            return format_insights(extract_insights(match), title)

        @tool
        def compare_articles(topic: str) -> str:
            """Compare what the different loaded articles say about the same company or
            topic, e.g. 'net profit growth' or 'deal wins and AI revenue'."""
            sections = []
            for a in self.articles:
                url, title = a.metadata["source"], a.metadata.get("title") or a.metadata["source"]
                chunks = self.store.similarity_search(topic, k=3, filter={"source": url}, fetch_k=50)
                if chunks:
                    self._add_source(url, a.metadata.get("title", ""))
                    text = "\n...\n".join(c.page_content for c in chunks)
                    sections.append(f"### Article: {title}\n{text}")
            if not sections:
                return "None of the loaded articles cover that topic."
            chain = COMPARE_PROMPT | get_chat_model()
            return chain.invoke({"topic": topic, "excerpts": "\n\n".join(sections)}).text

        extract_insights_tool.name = "extract_insights"
        return [search_articles, extract_insights_tool, compare_articles]

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """`history` is a list of {"role": "user"/"assistant", "content": ...}."""
        self._sources = []
        messages = (history or [])[-config.HISTORY_TURNS * 2:] + [{"role": "user", "content": question}]
        result = self.graph.invoke({"messages": messages})

        new_messages = result["messages"][len(messages):]
        tools_used = [
            call["name"]
            for m in new_messages if isinstance(m, AIMessage)
            for call in m.tool_calls
        ]
        return AgentResult(answer=result["messages"][-1].text, tools_used=tools_used, sources=self._sources)


if __name__ == "__main__":
    # Try it: python -m src.agent   (needs the index from: python -m src.vectorstore)
    from src.vectorstore import load_articles, load_vectorstore

    agent = ResearchAgent(load_vectorstore(), load_articles())
    tests = [
        ("What was HCLTech's EBIT margin in Q2?", "search_articles"),
        ("Give me the key insights for the Wipro article", "extract_insights"),
        ("Compare the net profit growth of TCS, HCLTech and Wipro", "compare_articles"),
        ("What did Infosys report for Q2?", "search_articles"),  # not in the articles
    ]
    history: list[dict] = []
    for question, expected in tests:
        r = agent.ask(question, history)
        status = "OK " if expected in r.tools_used else "CHECK"
        print("=" * 90)
        print(f"Q: {question}")
        print(f"[{status}] tools used: {r.tools_used}  (expected: {expected})")
        print(f"\nA: {r.answer}")
        for s in r.sources:
            print(f"   source: {s['title'] or s['url']}")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": r.answer}]

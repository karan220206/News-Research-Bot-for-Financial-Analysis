"""Answer questions about the loaded articles, with sources and chat history.

Steps for each question:
1. If there's chat history, rewrite a follow-up ("What about its profit?") as a
   standalone question ("What was Wipro's Q2 profit?").
2. Retrieve the top-k matching chunks from FAISS.
3. Ask the LLM to answer using only those chunks and say which ones it used.
"""

from dataclasses import dataclass, field

import config  # first, so USER_AGENT is set before langchain_community loads

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import get_chat_model

NOT_FOUND_MESSAGE = "I couldn't find that in the loaded articles."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question so it can be understood without the chat "
     "history: replace words like 'it', 'its', 'they' or 'that company' with the names "
     "they refer to. If the question is already standalone, return it unchanged. "
     "Return only the question."),
    ("human", "Chat history:\n{history}\n\nLatest question: {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial research assistant. Answer the question using ONLY the "
     "numbered news excerpts below. Do not use outside knowledge.\n"
     "- Quote exact figures (₹ crore, %, $) as written in the excerpts.\n"
     "- Keep the answer short and clear.\n"
     "- List the numbers of the excerpts you used.\n"
     "- If the excerpts don't contain the answer, set found to false and don't guess.\n\n"
     "Excerpts:\n{context}"),
    ("human", "{question}"),
])


class LLMAnswer(BaseModel):
    answer: str = Field(description="The answer, based only on the excerpts")
    found: bool = Field(description="False if the excerpts don't contain the answer")
    used_excerpts: list[int] = Field(description="Numbers of the excerpts the answer is based on")


@dataclass
class QAResult:
    question: str
    standalone_question: str
    answer: str
    found: bool
    sources: list[dict] = field(default_factory=list)  # [{"url": ..., "title": ...}]


def _format_history(history: list[dict]) -> str:
    recent = history[-config.HISTORY_TURNS * 2:]
    return "\n".join(f"{m['role'].title()}: {m['content']}" for m in recent)


def _format_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.metadata.get('title') or c.metadata['source']})\n{c.page_content}"
        for i, c in enumerate(chunks, start=1)
    )


def make_standalone(question: str, history: list[dict], llm=None) -> str:
    if not history:
        return question
    llm = llm or get_chat_model()
    chain = CONDENSE_PROMPT | llm
    return chain.invoke({"history": _format_history(history), "question": question}).content.strip()


def answer_question(
    question: str, store: FAISS, history: list[dict] | None = None, k: int = config.TOP_K
) -> QAResult:
    """Answer one question. `history` is a list of {"role": "user"/"assistant", "content": ...}."""
    llm = get_chat_model()
    standalone = make_standalone(question, history or [], llm)
    chunks = store.similarity_search(standalone, k=k)

    chain = ANSWER_PROMPT | llm.with_structured_output(LLMAnswer)
    result: LLMAnswer = chain.invoke({"context": _format_context(chunks), "question": standalone})

    if not result.found:
        return QAResult(question, standalone, NOT_FOUND_MESSAGE, found=False)

    sources, seen = [], set()
    for n in result.used_excerpts:
        if 1 <= n <= len(chunks):
            meta = chunks[n - 1].metadata
            if meta["source"] not in seen:
                seen.add(meta["source"])
                sources.append({"url": meta["source"], "title": meta.get("title", "")})
    return QAResult(question, standalone, result.answer, found=True, sources=sources)


if __name__ == "__main__":
    # Try it: python -m src.qa_chain   (needs the index from: python -m src.vectorstore)
    from src.vectorstore import load_vectorstore

    store = load_vectorstore()
    history: list[dict] = []
    questions = [
        "What was TCS's net profit in Q2 FY27?",
        "And how much was its dividend per share?",  # follow-up: "its" = TCS
        "How did Wipro's profit change compared to last year?",
        "What about its revenue?",  # follow-up: "its" = Wipro
        "What was Reliance Industries' Q2 profit?",  # not in the articles
    ]
    for q in questions:
        r = answer_question(q, store, history)
        print(f"Q: {q}")
        if r.standalone_question != q:
            print(f"   (understood as: {r.standalone_question})")
        print(f"A: {r.answer}")
        for s in r.sources:
            print(f"   source: {s['title'] or s['url']}")
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]

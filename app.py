"""Streamlit UI for the News Research Bot.

Run: streamlit run app.py
"""

import config  # first, so USER_AGENT is set before langchain_community loads

import streamlit as st

from src.agent import ResearchAgent
from src.insights import extract_insights
from src.llm import friendly_error
from src.loader import load_urls, split_documents
from src.vectorstore import (
    build_vectorstore,
    delete_vectorstore,
    load_articles,
    load_vectorstore,
    save_articles,
    vectorstore_exists,
)

MAX_URLS = 5
SENTIMENT_COLORS = {"Positive": "green", "Negative": "red", "Neutral": "gray"}

st.set_page_config(page_title="News Research Bot", page_icon="📈", layout="wide")


# --- State -------------------------------------------------------------------

def reset_session() -> None:
    st.session_state.messages = []  # [{"role", "content", "sources", "tools"}]
    st.session_state.insights = {}  # url -> ArticleInsights
    st.session_state.store = None
    st.session_state.articles = []
    st.session_state.agent = None


def init_state() -> None:
    if "messages" in st.session_state:
        return
    reset_session()
    if config.has_api_key() and vectorstore_exists():
        try:
            st.session_state.store = load_vectorstore()
            st.session_state.articles = load_articles()
        except Exception as e:
            st.session_state.load_error = friendly_error(e) if not isinstance(e, ValueError) else str(e)


def get_agent() -> ResearchAgent:
    if st.session_state.agent is None:
        st.session_state.agent = ResearchAgent(st.session_state.store, st.session_state.articles)
    return st.session_state.agent


def article_title(article) -> str:
    return article.metadata.get("title") or article.metadata["source"]


def md(text: str) -> str:
    """Escape $ so Streamlit doesn't render '$9.6 billion ... $3 billion' as maths."""
    return text.replace("$", r"\$")


# --- Processing ----------------------------------------------------------------

def process_urls(urls: list[str]) -> None:
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        st.sidebar.warning("Paste at least one URL first.")
        return

    progress = st.sidebar.progress(0, text="Loading articles...")
    with st.sidebar.status("Processing URLs...", expanded=True) as status:
        try:
            st.write("Loading articles...")
            result = load_urls(urls)
            for w in result.warnings:
                st.warning(w)
            if not result.documents:
                status.update(label="No articles could be loaded", state="error")
                progress.empty()
                return

            progress.progress(35, text="Splitting into chunks...")
            st.write("Splitting into chunks...")
            chunks = split_documents(result.documents)

            progress.progress(60, text="Building index...")
            st.write(f"Building index from {len(chunks)} chunks...")
            delete_vectorstore()
            store = build_vectorstore(chunks)
            save_articles(result.documents)

            reset_session()
            st.session_state.store = store
            st.session_state.articles = result.documents
            progress.progress(100, text="Done")
            status.update(
                label=f"Done: {len(result.documents)} of {len(urls)} articles ready",
                state="complete",
                expanded=bool(result.warnings),
            )
        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.error(friendly_error(e))
            progress.empty()


# --- Sidebar -------------------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        st.header("📰 News articles")
        st.caption(f"Paste up to {MAX_URLS} news article URLs.")
        urls = [
            st.text_input(f"URL {i + 1}", key=f"url_{i}", placeholder="https://...")
            for i in range(MAX_URLS)
        ]
        clicked = st.button("Process URLs", type="primary", width="stretch")

    if clicked:
        process_urls(urls)

    with st.sidebar:
        if st.session_state.articles:
            st.divider()
            st.subheader("Loaded articles")
            for a in st.session_state.articles:
                st.markdown(f"- [{md(article_title(a))}]({a.metadata['source']})")

        st.divider()
        if st.button("🗑️ Clear / start over", width="stretch"):
            delete_vectorstore()
            reset_session()
            for i in range(MAX_URLS):
                st.session_state.pop(f"url_{i}", None)
            st.rerun()


# --- Chat tab ------------------------------------------------------------------

def show_sources(sources: list[dict], tools: list[str]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- [{md(s['title'] or s['url'])}]({s['url']})")
    if tools:
        st.caption("Tools used: " + ", ".join(f"`{t}`" for t in dict.fromkeys(tools)))


def chat_tab() -> None:
    if st.session_state.store is None:
        st.info("Paste news URLs in the sidebar and click **Process URLs** to start.")
        return

    messages_box = st.container()  # messages go above the input box
    question = st.chat_input("Ask about the articles, e.g. 'What was TCS's net profit?'")

    with messages_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(md(m["content"]))
                if m["role"] == "assistant":
                    show_sources(m.get("sources", []), m.get("tools", []))
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with messages_box.chat_message("user"):
        st.markdown(md(question))

    with messages_box.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                r = get_agent().ask(question, history)
                reply = {"role": "assistant", "content": r.answer, "sources": r.sources, "tools": r.tools_used}
            except Exception as e:
                reply = {"role": "assistant", "content": f"⚠️ {friendly_error(e)}"}
        st.markdown(md(reply["content"]))
        show_sources(reply.get("sources", []), reply.get("tools", []))
    st.session_state.messages.append(reply)


# --- Insights tab --------------------------------------------------------------

def insight_card(article, ins) -> None:
    with st.container(border=True):
        top = st.columns([5, 1])
        top[0].markdown(f"#### [{md(article_title(article))}]({article.metadata['source']})")
        with top[1]:
            st.badge(ins.sentiment, color=SENTIMENT_COLORS.get(ins.sentiment, "gray"))

        companies, seen = [], set()
        for c in ins.companies:  # the model sometimes lists one company under two names
            key = (c.ticker or c.name).upper()
            if key not in seen:
                seen.add(key)
                companies.append(c)
        if companies:
            st.markdown("**Companies:** " + "  ".join(
                f"`{c.name}{' · ' + c.ticker if c.ticker else ''}`" for c in companies
            ))
        st.markdown(f"**Summary**  \n{md(ins.summary)}")
        if ins.key_numbers:
            st.markdown("**Key numbers**")
            st.dataframe(
                [{"Metric": n.label, "Value": n.value, "Change": n.change or ""} for n in ins.key_numbers],
                hide_index=True,
                width="stretch",
            )
        st.markdown(f"**Sentiment ({ins.sentiment}):** {md(ins.sentiment_reason)}")
        st.info(f"**Investor takeaway:** {md(ins.investor_takeaway)}")


def insights_tab() -> None:
    articles = st.session_state.articles
    if not articles:
        st.info("Process some URLs first, then come back for key insights.")
        return

    if st.button("✨ Get Key Insights", type="primary"):
        progress = st.progress(0)
        for i, a in enumerate(articles, start=1):
            progress.progress(i / len(articles), text=f"Analysing {i} of {len(articles)}: {article_title(a)}")
            try:
                st.session_state.insights[a.metadata["source"]] = extract_insights(a)
            except Exception as e:
                st.error(f"{article_title(a)}: {friendly_error(e)}")
        progress.empty()

    if not st.session_state.insights:
        st.caption("Click the button to get a summary, key numbers, sentiment and takeaway for each article.")
    for a in articles:
        ins = st.session_state.insights.get(a.metadata["source"])
        if ins:
            insight_card(a, ins)


# --- Page ----------------------------------------------------------------------

def main() -> None:
    st.title("📈 News Research Bot")
    st.caption("Ask questions about financial news articles and get answers with sources.")

    if not config.has_api_key():
        st.error("OpenAI API key is missing. Add `OPENAI_API_KEY=...` to the `.env` file and restart the app.")
        st.stop()

    init_state()
    if err := st.session_state.pop("load_error", None):
        st.warning(f"Couldn't load the saved index: {err}")

    sidebar()
    chat, insights = st.tabs(["💬 Chat", "🔍 Key Insights"])
    with chat:
        chat_tab()
    with insights:
        insights_tab()


main()

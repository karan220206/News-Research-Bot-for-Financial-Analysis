"""End-to-end evaluation with real articles, real embeddings and the real LLM.

Runs the checks from the spec (§8) on the 3 sample articles in config.SAMPLE_URLS
and writes the results to reports/test_results.md.

Run: python -m tests.run_evaluation
It builds its own index in a temporary folder, so the app's saved index is untouched.
Cost: a few cents of OpenAI credits.
"""

import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import config
from src.agent import ResearchAgent
from src.insights import extract_insights
from src.loader import load_urls, split_documents
from src.qa_chain import answer_question
from src.vectorstore import build_vectorstore

TCS, HCL, WIPRO = config.SAMPLE_URLS
SHORT = {TCS: "TCS (Upstox)", HCL: "HCLTech (Free Press Journal)", WIPRO: "Wipro (HDFC Sky)"}

# (question, any of these strings must be in the answer, expected source)
FACTUAL = [
    ("What was TCS's consolidated net profit in Q2 FY27?", ["13,884"], TCS),
    ("What was HCLTech's revenue for the September quarter?", ["35,620"], HCL),
    ("By how much did Wipro's net profit fall year on year?", ["1.4"], WIPRO),
    ("What was TCS's total contract value (TCV) for the quarter?", ["9.6"], TCS),
    ("What were HCLTech's new deal bookings in Q2?", ["3,258", "3.26"], HCL),
]
NOT_IN_ARTICLES = [
    "What was Infosys's net profit in Q2 FY27?",
    "What target price have analysts set for Reliance Industries shares?",
]
FOLLOW_UP = (
    "What was Wipro's revenue from operations in Q2 FY27?",
    "And how much did its net profit change?",
    "Wipro",  # the rewritten question must name the company
    ["3,200.6", "1.4"],  # and the answer must contain one of these
)
EXPECTED_INSIGHTS = {
    TCS: {"sentiment": {"Positive"}, "ticker": "TCS"},
    HCL: {"sentiment": {"Positive"}, "ticker": "HCLTECH"},
    WIPRO: {"sentiment": {"Neutral", "Negative"}, "ticker": "WIPRO"},
}
AGENT_CASES = [
    ("What was HCLTech's EBIT margin in Q2?", "search_articles", 1),
    ("Give me the key insights for the Wipro article", "extract_insights", 1),
    ("Compare the net profit growth of TCS, HCLTech and Wipro", "compare_articles", 1),
    ("What is the sentiment of each article?", "extract_insights", 3),
]


def mark(ok: bool) -> str:
    return "✅ Pass" if ok else "❌ Fail"


def cell(text: str, limit: int = 220) -> str:
    """Make text safe and short enough for a markdown table cell."""
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def numbers_in(value: str) -> list[str]:
    """'₹13,884 crore (+15%)' -> ['13,884', '15']"""
    return re.findall(r"\d[\d,]*(?:\.\d+)?", value)


def main() -> None:
    started = time.time()
    lines: list[str] = []
    summary: list[tuple[str, int, int]] = []

    # --- Build ------------------------------------------------------------------
    result = load_urls(config.SAMPLE_URLS)
    articles = {a.metadata["source"]: a for a in result.documents}
    chunks = split_documents(result.documents)
    store = build_vectorstore(chunks, path=Path(tempfile.mkdtemp()) / "index")
    print(f"Indexed {len(articles)} articles, {len(chunks)} chunks")

    # --- 1. Factual questions -----------------------------------------------------
    rows, passed = [], 0
    for q, expected, source in FACTUAL:
        r = answer_question(q, store)
        urls = [s["url"] for s in r.sources]
        ok = r.found and any(e in r.answer for e in expected) and source in urls
        passed += ok
        rows.append(f"| {cell(q)} | {cell(r.answer)} | {', '.join(SHORT.get(u, u) for u in urls) or '-'} | {mark(ok)} |")
    summary.append(("Factual questions (correct answer and source)", passed, len(FACTUAL)))
    lines += [
        "## 1. Factual questions",
        "",
        "Pass = the answer contains the expected figure **and** the right article is cited.",
        "",
        "| Question | Answer | Sources | Result |",
        "|---|---|---|---|",
        *rows,
        "",
    ]
    print(f"Factual: {passed}/{len(FACTUAL)}")

    # --- 2. Not in the articles ----------------------------------------------------
    rows, passed = [], 0
    for q in NOT_IN_ARTICLES:
        r = answer_question(q, store)
        ok = not r.found
        passed += ok
        rows.append(f"| {cell(q)} | {cell(r.answer)} | {mark(ok)} |")
    summary.append(("Questions not in the articles (says it doesn't know)", passed, len(NOT_IN_ARTICLES)))
    lines += [
        "## 2. Questions whose answers are not in the articles",
        "",
        "Pass = the bot says it couldn't find the answer instead of guessing.",
        "",
        "| Question | Answer | Result |",
        "|---|---|---|",
        *rows,
        "",
    ]
    print(f"Not in articles: {passed}/{len(NOT_IN_ARTICLES)}")

    # --- 3. Follow-up question ------------------------------------------------------
    first_q, follow_q, must_name, expected = FOLLOW_UP
    first = answer_question(first_q, store)
    history = [{"role": "user", "content": first_q}, {"role": "assistant", "content": first.answer}]
    follow = answer_question(follow_q, store, history)
    ok = must_name.lower() in follow.standalone_question.lower() and any(e in follow.answer for e in expected)
    summary.append(("Follow-up question uses chat history", int(ok), 1))
    lines += [
        "## 3. Follow-up question (chat history)",
        "",
        f"Pass = the follow-up is rewritten to name **{must_name}** and the answer is correct.",
        "",
        "| Step | Text |",
        "|---|---|",
        f"| Question 1 | {cell(first_q)} |",
        f"| Answer 1 | {cell(first.answer)} |",
        f"| Follow-up | {cell(follow_q)} |",
        f"| Understood as | {cell(follow.standalone_question)} |",
        f"| Answer 2 | {cell(follow.answer)} |",
        f"| **Result** | {mark(ok)} |",
        "",
    ]
    print(f"Follow-up: {'pass' if ok else 'FAIL'}")

    # --- 4. Key insights ------------------------------------------------------------
    rows, passed, details = [], 0, []
    for url, expect in EXPECTED_INSIGHTS.items():
        article = articles[url]
        ins = extract_insights(article)
        text = article.page_content
        tickers = {(c.ticker or "").upper() for c in ins.companies}
        figures = [n for kn in ins.key_numbers for n in numbers_in(kn.value)]
        missing = [f for f in figures if f not in text]
        sentiment_ok = ins.sentiment in expect["sentiment"]
        ticker_ok = expect["ticker"] in tickers
        grounded = not missing
        ok = sentiment_ok and ticker_ok and grounded and 3 <= len(re.findall(r"[.!?](?:\s|$)", ins.summary)) <= 6
        passed += ok
        rows.append(
            f"| {SHORT[url]} | {ins.sentiment} {'✅' if sentiment_ok else '❌'} | "
            f"{', '.join(sorted(t for t in tickers if t)) or '-'} {'✅' if ticker_ok else '❌'} | "
            f"{len(figures) - len(missing)}/{len(figures)} found in article {'✅' if grounded else '❌'} | {mark(ok)} |"
        )
        details += [
            f"**{SHORT[url]}**: {cell(ins.summary, 600)}",
            "",
            *[f"- {kn.label}: {kn.value}" + (f" ({kn.change})" if kn.change else "") for kn in ins.key_numbers],
            f"- *Sentiment reason:* {cell(ins.sentiment_reason, 300)}",
            f"- *Takeaway:* {cell(ins.investor_takeaway, 400)}",
            "",
        ]
        if missing:
            details += [f"> ⚠️ Figures not found word-for-word in the article: {', '.join(missing)}", ""]
    summary.append(("Key insights accurate (sentiment, ticker, figures, 3-5 sentence summary)", passed, len(EXPECTED_INSIGHTS)))
    lines += [
        "## 4. Key insights",
        "",
        "Pass = sentiment as expected, the right ticker, every key figure appears in the "
        "article text (no invented numbers), and a 3–5 sentence summary.",
        "",
        "| Article | Sentiment | Tickers | Key figures grounded | Result |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "<details><summary>Full insights</summary>",
        "",
        *details,
        "</details>",
        "",
    ]
    print(f"Insights: {passed}/{len(EXPECTED_INSIGHTS)}")

    # --- 5. Agent tool choice -------------------------------------------------------
    agent = ResearchAgent(store, list(articles.values()))
    rows, passed = [], 0
    for q, tool, times in AGENT_CASES:
        r = agent.ask(q)
        ok = r.tools_used.count(tool) == times and set(r.tools_used) == {tool}
        passed += ok
        expected = f"{tool} ×{times}" if times > 1 else tool
        rows.append(f"| {cell(q)} | {expected} | {', '.join(r.tools_used) or '-'} | {cell(r.answer, 160)} | {mark(ok)} |")
    summary.append(("Agent picks the right tool", passed, len(AGENT_CASES)))
    lines += [
        "## 5. Agent tool choice",
        "",
        "Pass = the agent calls exactly the expected tool (and no others).",
        "",
        "| Question | Expected tool | Tools used | Answer (start) | Result |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    print(f"Agent: {passed}/{len(AGENT_CASES)}")

    # --- Write the report -----------------------------------------------------------
    total_pass = sum(p for _, p, _ in summary)
    total = sum(t for _, _, t in summary)
    header = [
        "# Test Results: News Research Bot",
        "",
        f"- **Run on:** {datetime.now():%Y-%m-%d %H:%M}",
        f"- **Chat model:** `{config.CHAT_MODEL}` (temperature {config.TEMPERATURE}) · "
        f"**Embeddings:** `{config.EMBEDDING_MODEL}`",
        f"- **Index:** {len(articles)} articles, {len(chunks)} chunks "
        f"(chunk size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}, top-k {config.TOP_K})",
        f"- **Time taken:** {time.time() - started:.0f} s",
        "",
        "## Articles used",
        "",
        *[f"- [{a.metadata.get('title') or url}]({url})" for url, a in articles.items()],
        "",
        "## Summary",
        "",
        "| Check | Passed |",
        "|---|---|",
        *[f"| {name} | {p}/{t} {'✅' if p == t else '❌'} |" for name, p, t in summary],
        f"| **Total** | **{total_pass}/{total}** |",
        "",
        "Offline unit tests (splitting, loader fallbacks, index build/save/load): "
        "run `python -m pytest -v`.",
        "",
    ]
    report = Path(__file__).resolve().parent.parent / "reports" / "test_results.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(header + lines), encoding="utf-8")
    print(f"\nTotal: {total_pass}/{total}. Report written to {report}")


if __name__ == "__main__":
    main()

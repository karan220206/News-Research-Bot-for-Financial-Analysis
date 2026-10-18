"""Extract key investor insights from each article as structured data."""

import hashlib
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import get_chat_model

INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst. Extract key insights for an investor from the news "
     "article below. Use only facts stated in the article. Focus on the main story and "
     "ignore page clutter such as other stock prices, ads, related links and legal "
     "disclaimers. Copy figures exactly as written (₹ crore, %, $, basis points)."),
    ("human", "Article title: {title}\nURL: {url}\n\nArticle text:\n{text}"),
])


class Company(BaseModel):
    name: str = Field(description="Company name as used in the article")
    ticker: str | None = Field(
        default=None,
        description="Stock ticker if the article gives it, or the company's well-known "
        "NSE/NYSE/NASDAQ symbol; null if unsure",
    )


class KeyNumber(BaseModel):
    label: str = Field(description="What the number is, e.g. 'Net profit', 'Revenue', 'EBIT margin'")
    value: str = Field(description="The figure as written, e.g. '₹13,884 crore'")
    change: str | None = Field(
        default=None,
        description="Only a stated change, e.g. '+15% YoY' or '-162 bps'; null if none",
    )


class ArticleInsights(BaseModel):
    summary: str = Field(description="3 to 5 sentence summary of the article")
    companies: list[Company] = Field(
        description="Companies the article is mainly about or discusses, each listed once"
    )
    key_numbers: list[KeyNumber] = Field(
        description="The most important figures: revenue, profit, % changes, margins, "
        "deal wins, dividends, price targets. At most 8."
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Overall tone of the news for the main company's investors"
    )
    sentiment_reason: str = Field(description="One sentence explaining the sentiment")
    investor_takeaway: str = Field(description="The main takeaway for an investor, in one or two sentences")


_cache: dict[str, ArticleInsights] = {}  # avoids paying twice for the same article


def extract_insights(article: Document) -> ArticleInsights:
    """Run the insight prompt on one full article."""
    key = hashlib.md5(article.page_content.encode()).hexdigest()
    if key not in _cache:
        chain = INSIGHTS_PROMPT | get_chat_model().with_structured_output(ArticleInsights)
        _cache[key] = chain.invoke({
            "title": article.metadata.get("title", ""),
            "url": article.metadata.get("source", ""),
            "text": article.page_content,
        })
    return _cache[key]


if __name__ == "__main__":
    # Try it: python -m src.insights   (needs articles saved by: python -m src.vectorstore)
    from src.vectorstore import load_articles

    for article in load_articles():
        ins = extract_insights(article)
        print("=" * 90)
        print(article.metadata.get("title") or article.metadata["source"])
        print(f"\nSummary: {ins.summary}")
        print("\nCompanies:", ", ".join(f"{c.name} ({c.ticker or '-'})" for c in ins.companies))
        print("\nKey numbers:")
        for n in ins.key_numbers:
            print(f"  - {n.label}: {n.value}" + (f" ({n.change})" if n.change else ""))
        print(f"\nSentiment: {ins.sentiment} - {ins.sentiment_reason}")
        print(f"Takeaway: {ins.investor_takeaway}\n")

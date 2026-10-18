"""All settings for the News Research Bot live here.

Change a model or a number here and the whole app picks it up.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- API key -----------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Some news sites block requests that don't send a browser-like user agent.
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (NewsResearchBot)")

# --- Models ------------------------------------------------------------------
CHAT_MODEL = "gpt-5.4-mini"  # backup if credits run low: "gpt-5.4-nano"
# Don't change after building the index. If you do, delete vectorstore/ and rebuild.
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0  # 0 keeps answers factual

# --- Text splitting ----------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_ARTICLE_CHARS = 500  # pages with less text than this are skipped (paywall / blocked)

# --- Retrieval ---------------------------------------------------------------
TOP_K = 4
HISTORY_TURNS = 4  # past question/answer pairs used to understand follow-ups

# --- Storage -----------------------------------------------------------------
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# --- Sample articles for testing (IT sector Q2 FY27 results, Oct 2026) --------
SAMPLE_URLS = [
    "https://upstox.com/news/market-news/earnings/tcs-q2-results-net-profit-rises-15-to-13-884-crore-revenue-jumps-11-yo-y-tcv-rises-to-9-6-billion/article-201534/",
    "https://www.freepressjournal.in/business/hcltech-net-profit-rises-21-to-5132-crore-in-q2-fy27-revenue-up-115",
    "https://hdfcsky.com/news/wipro-q2-fy27-results-profit-revenue",
]


def has_api_key() -> bool:
    """True if a real-looking OpenAI key is set in .env."""
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY != "your-key-here"

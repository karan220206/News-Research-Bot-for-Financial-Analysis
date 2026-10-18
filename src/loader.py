"""Load news articles from URLs and split them into chunks."""

import hashlib
import os
import re
from dataclasses import dataclass, field

import config  # first, so USER_AGENT is set before langchain_community loads

from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class LoadResult:
    """Articles that loaded, plus warnings for the ones that didn't."""

    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    """Collapse runs of blank lines and spaces left over from HTML extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _is_article_text(element: Document) -> bool:
    """Keep paragraphs, tables and real bullet points; drop menus, share-price
    tickers, share buttons and other short page furniture."""
    category = element.metadata.get("category")
    text = element.page_content.strip()
    if len(text) < 30:
        return False
    if category in ("NarrativeText", "Table"):
        return True
    if category == "ListItem":
        return len(text.split()) >= 8  # ticker entries are 2-4 words
    return False


def _headline(elements: list[Document]) -> str:
    """The first reasonably long Title element is usually the headline."""
    for el in elements:
        if el.metadata.get("category") == "Title" and len(el.page_content.split()) >= 5:
            return el.page_content.strip()
    return ""


def _unstructured_elements(url: str) -> list[Document]:
    loader = UnstructuredURLLoader(
        urls=[url], mode="elements", headers={"User-Agent": os.environ["USER_AGENT"]}
    )
    return loader.load()  # returns [] on failure instead of raising


def _webbase_text(url: str) -> str:
    docs = WebBaseLoader(url).load()
    return "\n\n".join(d.page_content for d in docs)


def load_url(url: str) -> tuple[Document | None, str | None]:
    """Load one article. Returns (document, None) or (None, warning).

    Tries, in order: UnstructuredURLLoader with the noise filter, the same text
    unfiltered, then WebBaseLoader. Uses the first that gives enough text.
    """
    try:
        elements = _unstructured_elements(url)
    except Exception:  # site blocked us, bad URL, network error...
        elements = []
    title = _headline(elements)

    attempts = [
        ("unstructured", lambda: "\n\n".join(e.page_content for e in elements if _is_article_text(e))),
        ("unstructured-full", lambda: "\n\n".join(e.page_content for e in elements)),
        ("webbase", lambda: _webbase_text(url)),
    ]
    text = ""
    for method, get_text in attempts:
        try:
            text = _clean(get_text())
        except Exception:
            text = ""
        if len(text) >= config.MIN_ARTICLE_CHARS:
            metadata = {"source": url, "title": title, "loader": method}
            return Document(page_content=text, metadata=metadata), None

    return None, (
        f"Skipped {url}: got only {len(text)} characters of text "
        "(the site may block scraping or be behind a paywall)."
    )


def load_urls(urls: list[str]) -> LoadResult:
    """Load several articles, skipping blank entries and duplicate URLs."""
    result = LoadResult()
    seen = set()
    for url in (u.strip() for u in urls):
        if not url or url in seen:
            continue
        seen.add(url)
        doc, warning = load_url(url)
        if doc:
            result.documents.append(doc)
        else:
            result.warnings.append(warning)
    return result


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split articles into overlapping chunks and drop duplicate chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    unique, seen = [], set()
    for chunk in chunks:
        key = hashlib.md5(" ".join(chunk.page_content.lower().split()).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


if __name__ == "__main__":
    # Try it: python -m src.loader
    result = load_urls(config.SAMPLE_URLS)
    for w in result.warnings:
        print("WARNING:", w)

    chunks = split_documents(result.documents)
    print(f"\nLoaded {len(result.documents)} of {len(config.SAMPLE_URLS)} articles")
    for doc in result.documents:
        n = sum(c.metadata["source"] == doc.metadata["source"] for c in chunks)
        print(f"- {doc.metadata['title'] or doc.metadata['source']}")
        print(f"  {len(doc.page_content):>6} chars, {n:>3} chunks, via {doc.metadata['loader']}")
    print(f"Total chunks: {len(chunks)}")

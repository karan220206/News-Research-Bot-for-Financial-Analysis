"""Basic offline tests: splitting, loading fallbacks and FAISS build/save/load.

These use fake embeddings and fake page content, so they need no internet and
no OpenAI credits. Run: python -m pytest -v
"""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import config
from src import loader
from src.agent import find_article
from src.llm import MissingAPIKeyError, friendly_error
from src.vectorstore import (
    INFO_FILE,
    build_vectorstore,
    load_articles,
    load_vectorstore,
    save_articles,
    vectorstore_exists,
)

ARTICLE_TEXT = (
    "Tata Consultancy Services reported a consolidated net profit of ₹13,884 crore for Q2 FY27, "
    "up 15% year on year. Revenue from operations rose 11.2% to ₹73,188 crore. "
) * 30  # ~5,000 characters, enough for several chunks


def make_doc(text=ARTICLE_TEXT, url="https://example.com/tcs", title="TCS Q2 results") -> Document:
    return Document(page_content=text, metadata={"source": url, "title": title})


def fake_embeddings():
    return DeterministicFakeEmbedding(size=32)


# --- Splitting -----------------------------------------------------------------

def test_split_respects_chunk_size():
    chunks = loader.split_documents([make_doc()], chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_split_keeps_source_and_title():
    chunks = loader.split_documents([make_doc()])
    assert {c.metadata["source"] for c in chunks} == {"https://example.com/tcs"}
    assert all(c.metadata["title"] == "TCS Q2 results" for c in chunks)


def test_split_removes_duplicate_chunks():
    text = "A short paragraph that appears twice on the page."
    doc = make_doc(text=f"{text}\n\n{text}")
    chunks = loader.split_documents([doc], chunk_size=60, chunk_overlap=0)
    assert len(chunks) == 1


# --- Loading -------------------------------------------------------------------

def element(text: str, category: str) -> Document:
    return Document(page_content=text, metadata={"category": category})


def test_noise_filter_drops_tickers_and_keeps_article_text():
    assert not loader._is_article_text(element("TCS₹2,465115.00 (4.89%)", "ListItem"))
    assert not loader._is_article_text(element("Share on Twitter", "NarrativeText"))
    assert loader._is_article_text(element("Wipro reported a slight decline of 1.40 per cent in net profit.", "NarrativeText"))
    assert loader._is_article_text(element("Revenue grew 9.1 per cent to Rs 24,752.7 crore in the quarter", "ListItem"))


def test_load_url_skips_pages_with_too_little_text(monkeypatch):
    monkeypatch.setattr(loader, "_unstructured_elements", lambda url: [])
    monkeypatch.setattr(loader, "_webbase_text", lambda url: "Subscribe to read this article.")
    doc, warning = loader.load_url("https://example.com/paywalled")
    assert doc is None
    assert "Skipped https://example.com/paywalled" in warning


def test_load_url_falls_back_to_webbase(monkeypatch):
    def broken(url):
        raise ConnectionError("blocked")

    monkeypatch.setattr(loader, "_unstructured_elements", broken)
    monkeypatch.setattr(loader, "_webbase_text", lambda url: ARTICLE_TEXT)
    doc, warning = loader.load_url("https://example.com/tcs")
    assert warning is None
    assert doc.metadata["loader"] == "webbase"
    assert doc.metadata["source"] == "https://example.com/tcs"


def test_load_urls_ignores_blank_and_repeated_urls(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "load_url", lambda url: (calls.append(url), (make_doc(url=url), None))[1])
    result = loader.load_urls(["https://a.com", "  ", "https://a.com", "https://b.com"])
    assert calls == ["https://a.com", "https://b.com"]
    assert len(result.documents) == 2


# --- Vector store --------------------------------------------------------------

def test_index_build_save_and_load(tmp_path):
    chunks = loader.split_documents([make_doc()])
    build_vectorstore(chunks, path=tmp_path, embeddings=fake_embeddings())
    assert vectorstore_exists(tmp_path)

    store = load_vectorstore(tmp_path, embeddings=fake_embeddings())
    assert store.index.ntotal == len(chunks)
    results = store.similarity_search("net profit", k=2)
    assert len(results) == 2
    assert results[0].metadata["source"] == "https://example.com/tcs"


def test_index_info_records_embedding_model(tmp_path):
    build_vectorstore(loader.split_documents([make_doc()]), path=tmp_path, embeddings=fake_embeddings())
    info = json.loads((tmp_path / INFO_FILE).read_text(encoding="utf-8"))
    assert info["embedding_model"] == config.EMBEDDING_MODEL
    assert info["sources"] == ["https://example.com/tcs"]


def test_load_refuses_index_from_another_embedding_model(tmp_path):
    build_vectorstore(loader.split_documents([make_doc()]), path=tmp_path, embeddings=fake_embeddings())
    info_file = tmp_path / INFO_FILE
    info = json.loads(info_file.read_text(encoding="utf-8"))
    info["embedding_model"] = "some-other-model"
    info_file.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="some-other-model"):
        load_vectorstore(tmp_path, embeddings=fake_embeddings())


def test_load_without_index_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Process some URLs"):
        load_vectorstore(tmp_path / "missing", embeddings=fake_embeddings())


def test_articles_save_and_load(tmp_path):
    save_articles([make_doc()], path=tmp_path)
    articles = load_articles(tmp_path)
    assert len(articles) == 1
    assert articles[0].page_content == ARTICLE_TEXT
    assert articles[0].metadata["title"] == "TCS Q2 results"


# --- Small helpers -------------------------------------------------------------

def test_find_article_by_company_name_or_url():
    tcs = make_doc(url="https://example.com/tcs", title="TCS Q2 results: profit rises 15%")
    wipro = make_doc(text="Wipro reported...", url="https://example.com/wipro", title="Wipro profit falls")
    assert find_article([tcs, wipro], "Wipro") is wipro
    assert find_article([tcs, wipro], "https://example.com/tcs") is tcs


def test_missing_key_message_is_friendly():
    assert "Add OPENAI_API_KEY" in friendly_error(MissingAPIKeyError("x"))

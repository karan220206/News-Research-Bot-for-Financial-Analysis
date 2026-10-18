"""Turn chunks into OpenAI embeddings and store them in a FAISS index on disk."""

import json
import shutil
from pathlib import Path

import config  # first, so USER_AGENT is set before langchain_community loads

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.llm import require_api_key

INFO_FILE = "index_info.json"  # remembers which embedding model built the index
ARTICLES_FILE = "articles.json"  # full article text, for insight extraction


def get_embeddings() -> OpenAIEmbeddings:
    require_api_key()
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)


def build_vectorstore(
    chunks: list[Document], path: Path = config.VECTORSTORE_DIR, embeddings: Embeddings | None = None
) -> FAISS:
    """Embed the chunks, build a FAISS index and save it to disk.

    `embeddings` is only passed by tests (fake embeddings, no API calls).
    """
    if not chunks:
        raise ValueError("No chunks to index. Load at least one article first.")
    store = FAISS.from_documents(chunks, embeddings or get_embeddings())
    save_vectorstore(store, path)
    return store


def save_vectorstore(store: FAISS, path: Path = config.VECTORSTORE_DIR) -> None:
    path = Path(path)
    store.save_local(str(path))
    sources = sorted({d.metadata.get("source", "") for d in store.docstore._dict.values()})
    info = {"embedding_model": config.EMBEDDING_MODEL, "chunks": store.index.ntotal, "sources": sources}
    (path / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def vectorstore_exists(path: Path = config.VECTORSTORE_DIR) -> bool:
    return (Path(path) / "index.faiss").exists()


def load_vectorstore(path: Path = config.VECTORSTORE_DIR, embeddings: Embeddings | None = None) -> FAISS:
    """Load the saved index. Refuses if it was built with a different embedding model."""
    path = Path(path)
    if not vectorstore_exists(path):
        raise FileNotFoundError("No saved index yet. Process some URLs first.")

    info_path = path / INFO_FILE
    if info_path.exists():
        built_with = json.loads(info_path.read_text(encoding="utf-8")).get("embedding_model")
        if built_with != config.EMBEDDING_MODEL:
            raise ValueError(
                f"The index was built with '{built_with}' but config uses "
                f"'{config.EMBEDDING_MODEL}'. Clear the index and process the URLs again."
            )

    # Safe here: we only ever load an index this app saved itself.
    return FAISS.load_local(str(path), embeddings or get_embeddings(), allow_dangerous_deserialization=True)


def save_articles(documents: list[Document], path: Path = config.VECTORSTORE_DIR) -> None:
    """Keep the full (unsplit) articles next to the index."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    data = [{"text": d.page_content, "metadata": d.metadata} for d in documents]
    (path / ARTICLES_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_articles(path: Path = config.VECTORSTORE_DIR) -> list[Document]:
    file = Path(path) / ARTICLES_FILE
    if not file.exists():
        return []
    data = json.loads(file.read_text(encoding="utf-8"))
    return [Document(page_content=a["text"], metadata=a["metadata"]) for a in data]


def delete_vectorstore(path: Path = config.VECTORSTORE_DIR) -> None:
    shutil.rmtree(path, ignore_errors=True)


def get_retriever(store: FAISS, k: int = config.TOP_K):
    return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Try it: python -m src.vectorstore
    from src.loader import load_urls, split_documents

    result = load_urls(config.SAMPLE_URLS)
    for w in result.warnings:
        print("WARNING:", w)
    chunks = split_documents(result.documents)

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}...")
    build_vectorstore(chunks)
    save_articles(result.documents)
    print(f"Saved to {config.VECTORSTORE_DIR}")

    store = load_vectorstore()
    print(f"Reloaded from disk: {store.index.ntotal} vectors\n")

    for question in ["What was TCS's net profit in Q2?", "How did Wipro's profit change?"]:
        print("Q:", question)
        for doc, score in store.similarity_search_with_score(question, k=config.TOP_K):
            snippet = " ".join(doc.page_content.split())[:110]
            print(f"  [{score:.3f}] {doc.metadata['source'][:45]}... | {snippet}")
        print()

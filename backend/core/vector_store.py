"""
Vector store management.

Wraps FAISS to index document chunks and perform semantic similarity
search. Handles building an index from chunks, persisting it to disk,
loading it back, and querying it.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import settings
from backend.core.embeddings import get_embeddings


def build_vector_store(chunks: list[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Each chunk is embedded and indexed. The resulting store can be
    queried for semantically similar chunks.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    return store


def save_vector_store(store: FAISS, path: Path | None = None) -> Path:
    """
    Persist a FAISS vector store to disk so it can be reloaded later
    without re-embedding all documents.
    """
    target = path or settings.vector_store_path
    target.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))
    return target


def load_vector_store(path: Path | None = None) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    allow_dangerous_deserialization is required because FAISS uses
    pickle internally. This is safe here because we only ever load
    indexes we created ourselves.
    """
    source = path or settings.vector_store_path

    if not source.exists():
        raise FileNotFoundError(
            f"No vector store found at {source}. Build and save one first."
        )

    embeddings = get_embeddings()
    store = FAISS.load_local(
        str(source),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


def search(store: FAISS, query: str, k: int = 5) -> list[Document]:
    """
    Search the vector store for the k chunks most similar to the query.

    Returns the matching Document chunks, ordered by similarity
    (most relevant first).
    """
    return store.similarity_search(query, k=k)


def search_with_scores(
    store: FAISS, query: str, k: int = 5
) -> list[tuple[Document, float]]:
    """
    Like search(), but also returns the similarity score for each result.

    Lower scores mean closer matches (FAISS uses L2 distance).
    Useful for debugging retrieval quality and setting thresholds.
    """
    return store.similarity_search_with_score(query, k=k)
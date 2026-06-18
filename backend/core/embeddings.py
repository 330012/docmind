"""
Embedding generation.

Wraps a sentence-transformers model to convert text into dense vectors
for semantic search. The same model must be used for both indexing
documents and embedding queries, so this module exposes a single
shared embedding instance.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached embedding model instance.

    The model is loaded lazily on first call and reused thereafter.
    Loading is expensive (downloads weights on first run, ~90MB), so
    lru_cache ensures it happens only once per process.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string into a vector.

    Used at search time to convert the user's question into the same
    vector space as the indexed document chunks.
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple text chunks into vectors.

    Used at indexing time to convert document chunks into vectors
    before storing them in the vector database.
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)
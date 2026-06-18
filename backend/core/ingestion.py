"""
Document ingestion pipeline.

Loads documents from various formats (PDF, DOCX, TXT) and splits them
into overlapping chunks with metadata for downstream embedding and retrieval.

Note:
    Loaders are imported from `langchain_community`, which carries a sunset
    deprecation notice. As of the current LangChain release there is no
    dedicated standalone replacement package for these specific loaders, so
    community loaders remain the documented path. Migration is tracked as
    future tech debt.
"""

from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings


SupportedFormat = Literal["pdf", "docx", "txt"]


def _get_loader(file_path: Path):
    """Return the appropriate LangChain loader for the file's extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path))
    if suffix == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")

    raise ValueError(f"Unsupported file format: {suffix}")


def load_document(file_path: Path) -> list[Document]:
    """
    Load a document from disk into LangChain Document objects.

    Each page (for PDFs) or section becomes one Document with metadata
    including the source path and page number.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    loader = _get_loader(file_path)
    documents = loader.load()

    # Enrich metadata so we can cite sources later
    for doc in documents:
        doc.metadata["source"] = file_path.name

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter, which respects semantic boundaries:
    it tries to split on paragraphs, then sentences, then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(file_path: Path) -> list[Document]:
    """
    End-to-end ingestion: load a file and return chunked Documents
    ready for embedding.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    return chunks

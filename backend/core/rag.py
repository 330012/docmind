"""
Retrieval-Augmented Generation (RAG) chain.

Combines vector store retrieval with LLM generation to answer questions
grounded in document content. This is the baseline RAG pipeline; advanced
retrieval techniques (hybrid search, reranking, contextual retrieval) are
layered on top in later modules.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.llm import get_llm
from backend.core.vector_store import search


# The prompt that instructs the LLM how to answer
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are DocMind, a precise document assistant. Answer the question \
using ONLY the context provided below. If the context does not contain \
enough information to answer, say so clearly — do not invent facts.

Cite the source of each fact using the [source] given in the context.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Each chunk is labelled with its source so the LLM can cite it.
    """
    formatted = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        formatted.append(f"[{source}] {chunk.page_content}")
    return "\n\n".join(formatted)


def answer_question(store: FAISS, question: str, k: int = 5) -> dict:
    """
    Answer a question using RAG over the given vector store.

    Steps:
        1. Retrieve the k most relevant chunks.
        2. Format them into a context string.
        3. Ask the LLM to answer using only that context.

    Returns a dict with the answer and the source chunks used,
    so the caller can display citations.
    """
    # 1. Retrieve
    chunks = search(store, question, k=k)

    # 2. Augment
    context = _format_context(chunks)

    # 3. Generate
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [
            {
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
            }
            for chunk in chunks
        ],
    }
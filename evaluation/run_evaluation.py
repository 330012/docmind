"""
LLM-as-judge evaluation harness for DocMind.

Measures RAG quality without external eval libraries. Uses the configured
LLM as a judge to score three core metrics on a 0-1 scale:

  - Faithfulness:     Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Were the retrieved chunks actually relevant?

These mirror the standard RAGAS metrics but are implemented transparently
so the scoring logic is fully auditable.
"""

import json
import re
from pathlib import Path

from backend.core.ingestion import ingest
from backend.core.vector_store import build_vector_store, search
from backend.core.rag import answer_question
from backend.core.llm import get_llm
from evaluation.test_set import GOLDEN_SET


# ─── Judge prompts ───

FAITHFULNESS_PROMPT = """You are an evaluation judge. Score how well the ANSWER \
is supported by the CONTEXT, on a scale of 0.0 to 1.0.

1.0 = every claim in the answer is directly supported by the context.
0.0 = the answer contains claims not found in the context (hallucination).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


RELEVANCY_PROMPT = """You are an evaluation judge. Score how well the ANSWER \
addresses the QUESTION, on a scale of 0.0 to 1.0.

1.0 = the answer fully and directly answers the question.
0.0 = the answer is off-topic or does not address the question.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


PRECISION_PROMPT = """You are an evaluation judge. The CONTEXT below was \
retrieved to answer the QUESTION. Score what fraction of the context is \
relevant to answering the question, on a scale of 0.0 to 1.0.

1.0 = all retrieved context is relevant.
0.0 = none of the retrieved context is relevant.

QUESTION:
{question}

CONTEXT:
{context}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


def _judge(prompt: str) -> float:
    """Send a judge prompt to the LLM and parse the numeric score."""
    llm = get_llm()
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    # Extract the JSON object from the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 0.0
    try:
        data = json.loads(match.group(0))
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def evaluate_one(store, item: dict) -> dict:
    """Run the pipeline on one question and score all three metrics."""
    question = item["question"]

    result = answer_question(store, question, k=5)
    answer = result["answer"]

    chunks = search(store, question, k=5)
    context = "\n\n".join(c.page_content for c in chunks)

    faithfulness = _judge(
        FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    )
    relevancy = _judge(
        RELEVANCY_PROMPT.format(question=question, answer=answer)
    )
    precision = _judge(
        PRECISION_PROMPT.format(question=question, context=context)
    )

    return {
        "question": question,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": precision,
    }


def run(document_path: str = "data/sample.docx", label: str = "Baseline RAG") -> dict:
    """Full evaluation across the golden set; prints and returns averages."""
    print("Building vector store...")
    chunks = ingest(Path(document_path))
    store = build_vector_store(chunks)
    print(f"Indexed {len(chunks)} chunks\n")

    print(f"Evaluating {len(GOLDEN_SET)} questions...\n")
    results = []
    for item in GOLDEN_SET:
        scored = evaluate_one(store, item)
        results.append(scored)
        print(
            f"  {scored['question'][:45]:<45} "
            f"F={scored['faithfulness']:.2f} "
            f"R={scored['answer_relevancy']:.2f} "
            f"P={scored['context_precision']:.2f}"
        )

    # Averages
    n = len(results)
    avg = {
        "faithfulness": sum(r["faithfulness"] for r in results) / n,
        "answer_relevancy": sum(r["answer_relevancy"] for r in results) / n,
        "context_precision": sum(r["context_precision"] for r in results) / n,
    }

    print("\n" + "=" * 55)
    print(f"RESULTS — {label}")
    print("=" * 55)
    print(f"  Faithfulness:      {avg['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {avg['answer_relevancy']:.3f}")
    print(f"  Context Precision: {avg['context_precision']:.3f}")
    print("=" * 55)

    return avg


if __name__ == "__main__":
    run()

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ingestion.pdf_loader import load_pdf
from ingestion.text_chunker import chunk_pages
from retriever.hybrid_retriever import HybridRetriever, tokenize


ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "raw_pdfs" / "AMSL_IEEE_Research_paper.pdf"
QUESTIONS_PATH = ROOT_DIR / "data" / "evaluation" / "retrieval_questions.json"

TOP_K = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def evaluate_method(
    method_name: str,
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    rankings: list[list[int]],
    latencies: list[float],
) -> dict[str, Any]:
    """
    Evaluate using page-level relevance labels.

    Recall@3 = fraction of relevant pages represented in top 3.
    MRR@3 = reciprocal rank of the first result from a relevant page.
    """
    recalls = []
    reciprocal_ranks = []

    for question, ranking in zip(questions, rankings):
        relevant_pages = set(question["relevant_pages"])
        top_results = ranking[:TOP_K]

        retrieved_pages = {
            chunks[index].get("metadata", {}).get("page")
            for index in top_results
        }

        page_hits = len(relevant_pages & retrieved_pages)
        recall = (
            page_hits / len(relevant_pages)
            if relevant_pages
            else 0.0
        )
        recalls.append(recall)

        first_relevant_rank = next(
            (
                rank
                for rank, index in enumerate(top_results, start=1)
                if chunks[index].get("metadata", {}).get("page")
                in relevant_pages
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

    return {
        "method": method_name,
        "recall_at_3": float(np.mean(recalls)),
        "mrr_at_3": float(np.mean(reciprocal_ranks)),
        "mean_latency_ms": float(np.mean(latencies) * 1000),
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Research paper not found: {PDF_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_PATH}"
        )

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print("Evaluation questions:", len(questions))
    print("Pages extracted:", len(pages))
    print("Chunks created:", len(chunks))
    print("Top K:", TOP_K)

    # Build the retriever once. Reuse its model, embeddings, and BM25 index
    # for the dense and sparse baselines to avoid duplicate model loading.
    hybrid = HybridRetriever(chunks, model_name=MODEL_NAME)
    model = hybrid.model
    embeddings = hybrid.embeddings
    bm25 = hybrid.bm25

    dense_rankings = []
    bm25_rankings = []
    hybrid_rankings = []

    dense_times = []
    bm25_times = []
    hybrid_times = []

    # Map each chunk to its original index.
    chunk_lookup = {
        (
            chunk["text"],
            chunk.get("metadata", {}).get("page"),
            chunk.get("metadata", {}).get("chunk_index"),
        ): index
        for index, chunk in enumerate(chunks)
    }

    for question_number, item in enumerate(questions, start=1):
        query = item["question"]

        # Dense retrieval
        start = time.perf_counter()
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        dense_scores = embeddings @ query_embedding
        dense_order = np.argsort(
            -dense_scores, kind="stable"
        ).tolist()
        dense_times.append(time.perf_counter() - start)
        dense_rankings.append(dense_order)

        # BM25 retrieval
        start = time.perf_counter()
        bm25_scores = np.asarray(
            bm25.get_scores(
                # Use the same tokenizer as the production retriever.
                tokenize(query)
            )
        )
        bm25_order = np.argsort(
            -bm25_scores, kind="stable"
        ).tolist()
        bm25_times.append(time.perf_counter() - start)
        bm25_rankings.append(bm25_order)

        # Hybrid retrieval
        start = time.perf_counter()
        hybrid_results = hybrid.retrieve(
            query,
            top_k=len(chunks),
            candidate_k=len(chunks),
        )
        hybrid_times.append(time.perf_counter() - start)

        hybrid_order = []
        for result in hybrid_results:
            metadata = result.get("metadata", {})
            key = (
                result["text"],
                metadata.get("page"),
                metadata.get("chunk_index"),
            )
            if key in chunk_lookup:
                hybrid_order.append(chunk_lookup[key])

        hybrid_rankings.append(hybrid_order)

        print(f"Evaluated {question_number}/{len(questions)}: {item['id']}")

    results = [
        evaluate_method(
            "Dense", questions, chunks, dense_rankings, dense_times
        ),
        evaluate_method(
            "BM25", questions, chunks, bm25_rankings, bm25_times
        ),
        evaluate_method(
            "Hybrid", questions, chunks, hybrid_rankings, hybrid_times
        ),
    ]

    print("\n=== RETRIEVAL EVALUATION ===")
    print(
        f"{'Method':<10} {'Recall@3':>10} "
        f"{'MRR@3':>10} {'Latency ms':>13}"
    )
    print("-" * 47)

    for result in results:
        print(
            f"{result['method']:<10} "
            f"{result['recall_at_3']:>10.3f} "
            f"{result['mrr_at_3']:>10.3f} "
            f"{result['mean_latency_ms']:>13.2f}"
        )

    output_path = (
        ROOT_DIR / "data" / "evaluation" / "retrieval_results.json"
    )
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")
    print(
        "\nNote: metrics use page-level relevance labels. "
        "They are provisional until supporting chunks are verified."
    )


if __name__ == "__main__":
    main()

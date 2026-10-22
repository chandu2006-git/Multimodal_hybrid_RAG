
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.document_ingestor import ingest_document
from ingestion.text_chunker import chunk_pages
from retriever.hybrid_retriever import HybridRetriever
from retriever.reranker import CrossEncoderReranker

from evaluation.retrieval.ground_truth import (
    load_evaluation_dataset,
    get_relevant_pages,
)
from evaluation.retrieval.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
)


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw_pdfs"
    / "AMSL_IEEE_Research_paper.pdf"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEBUG_RESULTS_PATH = RESULTS_DIR / "top10_retrieval_debug.md"

DEBUG_QUESTION_IDS = {
    f"Q{i:03d}" for i in range(1, 31)
}

TOP_K = 30
METRIC_K = 5


def calculate_metrics(
    results: list[dict[str, Any]],
    relevant_pages: set[int],
    question_id: str,
    category: str,
) -> dict[str, Any]:
    """Calculate page-level retrieval metrics."""
    # Evaluate only the top 5 ranked chunks.
    top_results = results[:METRIC_K]

    retrieved_pages = [
        result["metadata"].get("page")
        for result in top_results
        if result.get("metadata", {}).get("page") is not None
    ]

    return {
        "id": question_id,
        "category": category,
        "recall_at_5": recall_at_k(
            retrieved_pages, relevant_pages, METRIC_K
        ),
        "precision_at_5": precision_at_k(
            retrieved_pages, relevant_pages, METRIC_K
        ),
        "mrr_at_5": reciprocal_rank(
            retrieved_pages, relevant_pages, METRIC_K
        ),
    }


def add_results_to_debug(
    sections: list[str],
    title: str,
    results: list[dict[str, Any]],
    include_reranker_score: bool = False,
) -> None:
    """Append ranked chunk details to the Markdown report."""

    sections.extend([
        f"### {title}",
        "",
        f"Returned chunks: {len(results)}",
        "",
    ])

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})

        sections.extend([
            f"#### Rank {rank}",
            "",
            f"- Page: {metadata.get('page', 'Unknown')}",
            f"- Dense score: {result.get('dense_score', 'N/A')}",
            f"- BM25 score: {result.get('bm25_score', 'N/A')}",
            f"- RRF score: {result.get('rrf_score', 'N/A')}",
        ])

        if include_reranker_score:
            sections.append(
                f"- Reranker score: "
                f"{result.get('reranker_score', 'N/A')}"
            )

        sections.extend([
            "",
            "**Chunk content:**",
            "",
            result.get("text", ""),
            "",
            "---",
            "",
        ])


def build_report(
    retriever_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an aggregate evaluation report."""

    if not records:
        raise ValueError("No answerable questions to evaluate.")

    metric_names = (
        "recall_at_5",
        "precision_at_5",
        "mrr_at_5",
    )

    return {
        "dataset": "rag_evaluation_dataset.json",
        "retriever": retriever_name,
        "k": METRIC_K,
        "evaluated_questions": len(records),
        "metrics": {
            metric: mean(record[metric] for record in records)
            for metric in metric_names
        },
        "note": (
            "Preliminary page-level evaluation. Multiple chunks "
            "from the same page can occupy multiple ranks; "
            "chunk-level labels are needed for stronger evaluation."
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_evaluation_dataset()

    print("Loading and chunking document...")
    document = ingest_document(PDF_PATH)
    chunks = chunk_pages(document["text_pages"])

    for chunk in chunks:
        chunk["metadata"].setdefault("source", document["source"])

    print("Initializing hybrid retriever...")
    retriever = HybridRetriever(chunks)

    print("Initializing cross-encoder reranker...")
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-base"
    )

    baseline_records = []
    reranked_records = []

    debug_sections = [
        "# Hybrid Retrieval vs Cross-Encoder Reranking",
        "",
        "Original and reranked results for selected questions.",
        "",
    ]

    for item in dataset:
        relevant_pages = get_relevant_pages(item)

        if not relevant_pages:
            continue

        question_id = item["id"]
        question = item["question"]
        category = item["category"]

        print(f"Processing {question_id}: {question}")

        # 1. Retrieve candidate chunks once.
        original_results = retriever.retrieve(
            question,
            top_k=TOP_K,
        )

        # 2. Rerank those exact same candidates once.
        reranked_results = reranker.rerank(
            query=question,
            chunks=original_results,
            top_k=TOP_K,
        )

        # Select the top 5 chunks for the LLM stage.

        selected_chunks = reranked_results[:5]

        # 3. Calculate metrics for both rankings.
        baseline_records.append(
            calculate_metrics(
                original_results,
                relevant_pages,
                question_id,
                category,
            )
        )

        reranked_records.append(
            calculate_metrics(
                reranked_results,
                relevant_pages,
                question_id,
                category,
            )
        )

        # 4. Save both rankings for selected questions.
        if question_id in DEBUG_QUESTION_IDS:
            debug_sections.extend([
                f"## {question_id}: {question}",
                "",
                (
                    f"**Expected answer:** "
                    f"{item.get('reference_answer', 'Not provided')}"
                ),
                "",
                f"**Relevant pages:** {sorted(relevant_pages)}",
                "",
            ])

            add_results_to_debug(
                debug_sections,
                "Original Hybrid Ranking",
                original_results,
            )

            add_results_to_debug(
                debug_sections,
                "Cross-Encoder Reranked Results",
                reranked_results,
                include_reranker_score=True,
            )

    # 5. Write the Markdown file once, overwriting the previous run.
    DEBUG_RESULTS_PATH.write_text(
        "\n".join(debug_sections),
        encoding="utf-8",
    )

    baseline_report = build_report(
        "HybridRetriever",
        baseline_records,
    )

    reranked_report = build_report(
        "HybridRetriever + CrossEncoderReranker",
        reranked_records,
    )

    # 6. Save aggregate reports.
    (RESULTS_DIR / "retrieval_hybrid.json").write_text(
        json.dumps(baseline_report, indent=2),
        encoding="utf-8",
    )

    (RESULTS_DIR / "retrieval_reranked.json").write_text(
        json.dumps(reranked_report, indent=2),
        encoding="utf-8",
    )

    # 7. Save question-level comparison.
    comparison_path = RESULTS_DIR / "retrieval_comparison.csv"

    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "id",
            "category",
            "baseline_recall_at_5",
            "reranked_recall_at_5",
            "baseline_precision_at_5",
            "reranked_precision_at_5",
            "baseline_mrr_at_5",
            "reranked_mrr_at_5",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for baseline, reranked in zip(
            baseline_records,
            reranked_records,
        ):
            writer.writerow({
                "id": baseline["id"],
                "category": baseline["category"],
                "baseline_recall_at_5": baseline["recall_at_5"],
                "reranked_recall_at_5": reranked["recall_at_5"],
                "baseline_precision_at_5": baseline["precision_at_5"],
                "reranked_precision_at_5": reranked["precision_at_5"],
                "baseline_mrr_at_5": baseline["mrr_at_5"],
                "reranked_mrr_at_5": reranked["mrr_at_5"],
            })

    print("\n=== BASELINE METRICS ===")
    print(json.dumps(baseline_report, indent=2))

    print("\n=== RERANKED METRICS ===")
    print(json.dumps(reranked_report, indent=2))

    print(f"\nDebug report: {DEBUG_RESULTS_PATH}")
    print(f"Comparison CSV: {comparison_path}")


if __name__ == "__main__":
    main()

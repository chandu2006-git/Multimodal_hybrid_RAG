
"""Main evaluation runner for Multimodal Hybrid RAG."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import traceback
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag.pipeline import MultimodalRAGPipeline
from retriever.reranker import CrossEncoderReranker

from evaluation.quantitative_metrics import (
    ranked_unique_pages,
    retrieval_metrics,
    token_overlap_metrics,
)
from evaluation.qualitative_metrics import (
    QUALITATIVE_METRICS,
    judge_answer,
)

DATASET_PATH = ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
DEFAULT_PDF = ROOT / "data" / "raw" / "Time-Series_Forecasting_in_Industrial_Environments_A_Performance_Study_and_a_Novel_Late_Fusion_Framework"
RESULTS_DIR = ROOT / "evaluation" / "results_reranked_top20_top5"
RAW_DIR = RESULTS_DIR / "raw_results"
TOP_K = 5
CANDIDATE_K = 30


def load_dataset():
    """Load evaluation questions without modifying the dataset."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data if isinstance(data, list) else data.get(
        "questions", data.get("data", [])
    )

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            f"No evaluation questions found in {DATASET_PATH}"
        )

    return questions


def load_json(path):
    """Load JSON if it exists."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    """Write JSON safely and consistently."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)

class RetrievalOnlyAnswerGenerator:
    def generate(self, query, evidence):
        return {
            "answer": "",
            "citations": [],
            "abstained": True,
            "reason": "Quantitative retrieval-only evaluation",
        }


def build_pipeline(pdf_path, mode):
    """Create the pipeline with reranking enabled."""

    reranker = CrossEncoderReranker()

    if mode == "full":
        return MultimodalRAGPipeline(
            pdf_path=str(pdf_path),
            text_top_k=TOP_K,
            text_candidate_k=CANDIDATE_K,
            reranker=reranker,
            image_top_k=3,
            table_top_k=3,
        )

    return MultimodalRAGPipeline(
        pdf_path=str(pdf_path),
        text_top_k=TOP_K,
        text_candidate_k=CANDIDATE_K,
        reranker=reranker,
        image_top_k=3,
        table_top_k=3,
        answer_generator=RetrievalOnlyAnswerGenerator(),
    )


def evaluate_question(question_data, pipeline, mode, use_judge):
    """Run retrieval, optional generation, and per-question metrics."""
    question_id = str(question_data["id"])
    query = question_data["question"]

    result = {
        "id": question_id,
        "category": question_data.get("category"),
        "question": query,
        "answerable": question_data.get("answerable"),
        "mode": mode,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        retrieval = pipeline.retrieve(query)

        retrieved_pages = ranked_unique_pages(
            retrieval.get("text_results", []),
            TOP_K,
        )

        relevant_pages = question_data.get("relevant_pages", [])

        result["retrieved_pages"] = retrieved_pages
        result["relevant_pages"] = relevant_pages
        result["retrieval_metrics"] = retrieval_metrics(
            retrieved_pages,
            relevant_pages,
            TOP_K,
        )

        result["retrieved_text"] = retrieval.get("text_results", [])
        result["retrieved_images"] = retrieval.get("image_results", [])
        result["retrieved_tables"] = retrieval.get("table_results", [])
        result["evidence"] = retrieval.get("evidence", [])

        if mode == "full":
            generated = pipeline.answer_generator.generate(
                query=query,
                evidence=retrieval["evidence"],
            )

            result["answer"] = generated.get("answer", "")
            result["citations"] = generated.get("sources", [])
            result["model"] = generated.get("model")

            result["token_overlap_metrics"] = token_overlap_metrics(
                result["answer"],
                question_data.get("reference_answer", ""),
            )

            if use_judge:
                try:
                    result["qualitative_metrics"] = judge_answer(
                        question=query,
                        answer=result["answer"],
                        evidence=result["evidence"],
                        citations=result["citations"],
                        answerable=bool(question_data.get("answerable")),
                    )
                except Exception as exc:
                    result["judge_error"] = str(exc)

        result["status"] = "success"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc() 
    
    return result


def mean(values):
    """Average numeric values, excluding missing values."""
    valid = [
        value for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(valid) / len(valid) if valid else None


def aggregate_results(rows, questions, mode):
    """Create dataset-level metric summaries."""
    successful = [row for row in rows if row.get("status") == "success"]

    retrieval_keys = [
        f"recall@{TOP_K}",
        f"precision@{TOP_K}",
        f"hit_rate@{TOP_K}",
        f"mrr@{TOP_K}",
    ]

    retrieval_summary = {
        key: mean([
            (row.get("retrieval_metrics") or {}).get(key)
            for row in successful
        ])
        for key in retrieval_keys
    }

    retrieval_summary["evaluated_questions"] = sum(
        (row.get("retrieval_metrics") or {}).get(
            f"recall@{TOP_K}"
        ) is not None
        for row in successful
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset_path": str(DATASET_PATH),
        "total_questions": len(questions),
        "answerable_questions": sum(
            bool(item.get("answerable")) for item in questions
        ),
        "unanswerable_questions": sum(
            not bool(item.get("answerable")) for item in questions
        ),
        "successful_questions": len(successful),
        "failed_questions": len(rows) - len(successful),
        "retrieval_metrics": retrieval_summary,
    }

    if mode == "full":
        for metric in ("token_precision", "token_recall", "token_f1"):
            summary[metric] = mean([
                (row.get("token_overlap_metrics") or {}).get(metric)
                for row in successful
            ])

        summary["qualitative_metrics"] = {
            metric: mean([
                (row.get("qualitative_metrics") or {}).get(metric)
                for row in successful
            ])
            for metric in QUALITATIVE_METRICS
        }

    return summary


def write_csv(path, rows, columns):
    """Save selected fields to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            flat_row = {}

            for column in columns:
                value = row.get(column)

                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)

                flat_row[column] = value

            writer.writerow(flat_row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Multimodal Hybrid RAG pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["quantitative", "full", "metrics_only"],
        default="quantitative",
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable optional Gemini-based qualitative evaluation.",
    )

    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("Choose either --resume or --force, not both.")

    questions = load_dataset()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "metrics_only":
        rows = []

        for item in questions:
            path = RAW_DIR / f"{item['id']}.json"
            saved = load_json(path)

            if saved is not None:
                rows.append(saved)

        if not rows:
            raise SystemExit(
                "No saved results found. Run an evaluation first."
            )

        summary = aggregate_results(rows, questions, "full")
        save_json(RESULTS_DIR / "summary.json", summary)

        print(json.dumps(summary, indent=2))
        return

    if not args.pdf.exists():
        raise SystemExit(
            f"PDF not found: {args.pdf}\n"
            "Provide its actual location using --pdf."
        )

    pipeline = None
    rows = []

    for item in questions:
        question_id = str(item["id"])
        result_path = RAW_DIR / f"{question_id}.json"
        previous = load_json(result_path)

        if previous is not None and not args.force:
            same_mode = previous.get("mode") == args.mode
            has_answer = "answer" in previous

            if args.resume and same_mode:
                if args.mode != "full" or has_answer:
                    rows.append(previous)
                    print(f"[SKIP] {question_id}: saved result reused")
                    continue

            if not args.resume:
                raise SystemExit(
                    f"Results already exist for {question_id}. "
                    "Use --resume to reuse them or --force to rerun."
                )

        if pipeline is None:
            pipeline = build_pipeline(args.pdf, args.mode)

        print(f"[RUN] {question_id}: {item['question']}")

        result = evaluate_question(
            item,
            pipeline,
            args.mode,
            args.judge and args.mode == "full",
        )

        save_json(result_path, result)
        rows.append(result)

        print(f"[{result['status'].upper()}] {question_id}")

    summary = aggregate_results(rows, questions, args.mode)
    save_json(RESULTS_DIR / "summary.json", summary)

    write_csv(
        RESULTS_DIR / "quantitative_results.csv",
        rows,
        [
            "id",
            "category",
            "status",
            "retrieved_pages",
            "relevant_pages",
            "retrieval_metrics",
            "error",
        ],
    )

    if args.mode == "full":
        write_csv(
            RESULTS_DIR / "qualitative_results.csv",
            rows,
            [
                "id",
                "category",
                "status",
                "answer",
                "citations",
                "token_overlap_metrics",
                "qualitative_metrics",
                "judge_error",
                "error",
            ],
        )

        write_csv(
            RESULTS_DIR / "human_review.csv",
            rows,
            [
                "id",
                "question",
                "answer",
                "evidence",
                "citations",
                "qualitative_metrics",
                "judge_error",
            ],
        )

    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

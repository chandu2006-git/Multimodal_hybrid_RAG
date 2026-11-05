
"""Quantitative metrics for Multimodal Hybrid RAG evaluation."""

import re
from collections.abc import Iterable, Mapping
from typing import Any


def _normalise_page(value: Any) -> str | None:
    """Normalise page labels such as 3, '3', and 'page_3'."""
    if value is None or isinstance(value, bool):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"(?:page[_\s-]*)?(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else text.casefold()


def _get_page(result: Mapping[str, Any]) -> str | None:
    """Extract the page number from a retrieved text chunk."""
    metadata = result.get("metadata") or {}

    if not isinstance(metadata, Mapping):
        return None

    for key in ("page", "page_number", "page_num", "page_index", "source_page"):
        if key in metadata:
            page = _normalise_page(metadata[key])
            if page is not None:
                return page

    return None


def ranked_unique_pages(
    results: Iterable[Mapping[str, Any]],
    k: int = 5,
) -> list[str]:
    """Return the first k unique pages in retrieval rank order."""
    pages = []
    seen = set()

    for result in results:
        page = _get_page(result)

        if page is not None and page not in seen:
            seen.add(page)
            pages.append(page)

            if len(pages) >= k:
                break

    return pages


def retrieval_metrics(
    retrieved_pages: Iterable[Any],
    relevant_pages: Iterable[Any],
    k: int = 5,
) -> dict[str, float | int | None]:
    """Calculate page-level Recall@k, Precision@k, Hit Rate@k, and MRR@k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    retrieved = [_normalise_page(p) for p in retrieved_pages]
    retrieved = [p for p in retrieved if p is not None][:k]

    relevant = {_normalise_page(p) for p in relevant_pages}
    relevant.discard(None)

    # No relevance labels means the retrieval metrics are undefined.
    if not relevant:
        return {
            f"recall@{k}": None,
            f"precision@{k}": None,
            f"hit_rate@{k}": None,
            f"mrr@{k}": None,
            "relevant_page_count": 0,
            "retrieved_page_count": len(retrieved),
            "relevant_pages_found": 0,
        }

    found = len(set(retrieved) & relevant)

    first_rank = next(
        (
            rank
            for rank, page in enumerate(retrieved, start=1)
            if page in relevant
        ),
        None,
    )

    return {
        f"recall@{k}": found / len(relevant),
        f"precision@{k}": found / k,
        f"hit_rate@{k}": float(first_rank is not None),
        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
        "relevant_page_count": len(relevant),
        "retrieved_page_count": len(retrieved),
        "relevant_pages_found": found,
    }


def token_overlap_metrics(
    generated_answer: str,
    reference_answer: str,
) -> dict[str, float | None]:
    """Calculate token precision, recall, and F1 as lexical diagnostics."""
    tokenize = lambda text: re.findall(
        r"\b[\w'-]+\b", (text or "").casefold()
    )

    predicted = tokenize(generated_answer)
    reference = tokenize(reference_answer)

    if not predicted or not reference:
        return {
            "token_precision": None,
            "token_recall": None,
            "token_f1": None,
        }

    predicted_counts = {}
    reference_counts = {}

    for token in predicted:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1

    for token in reference:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )

    precision = overlap / len(predicted)
    recall = overlap / len(reference)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

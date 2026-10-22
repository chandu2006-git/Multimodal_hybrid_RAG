
from collections.abc import Sequence


def recall_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: set[int],
    k: int,
) -> float:
    """Fraction of relevant pages retrieved in the top k."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_pages:
        return 0.0

    retrieved = set(retrieved_pages[:k])
    return len(retrieved & relevant_pages) / len(relevant_pages)


def precision_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: set[int],
    k: int,
) -> float:
    """Fraction of the top k results that are relevant."""
    if k < 1:
        raise ValueError("k must be >= 1")

    results = retrieved_pages[:k]
    if not results:
        return 0.0

    return sum(page in relevant_pages for page in results) / len(results)


def reciprocal_rank(
    retrieved_pages: Sequence[int],
    relevant_pages: set[int],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant result within top k."""
    if k < 1:
        raise ValueError("k must be >= 1")

    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0

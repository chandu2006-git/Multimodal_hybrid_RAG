
from pathlib import Path
from typing import Any


def build_evidence(
    text_results: list[dict[str, Any]],
    image_results: list[dict[str, Any]],
    table_results: list[dict[str, Any]] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """
    Combine retrieved text, image, and table results into
    a consistent evidence format for the RAG answer generator.
    """
    evidence: list[dict[str, Any]] = []

    # 1. Text evidence
    for rank, result in enumerate(text_results, start=1):
        metadata = result.get("metadata") or {}

        evidence.append(
            {
                "modality": "text",
                "rank": rank,
                "content": result["text"],
                "source": source or metadata.get("source"),
                "page": metadata.get("page"),
                "metadata": metadata,
                "scores": {
                    "dense": result.get("dense_score"),
                    "bm25": result.get("bm25_score"),
                    "rrf": result.get("rrf_score"),
                },
            }
        )

    # 2. Image evidence
    for rank, result in enumerate(image_results, start=1):
        image_path = result.get("image_path")

        evidence.append(
            {
                "modality": "image",
                "rank": rank,
                "content": result.get("filename", "Unnamed image"),
                "source": source or result.get("source"),
                "page": result.get("page"),
                "image_path": (
                    str(Path(image_path)) if image_path else None
                ),
                "metadata": {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "image_path",
                        "filename",
                        "page",
                        "source",
                        "similarity",
                    }
                },
                "scores": {
                    "similarity": result.get("similarity"),
                },
            }
        )

    # 3. Table evidence
    table_results = table_results or []

    for rank, table in enumerate(table_results, start=1):
        metadata = table.get("metadata") or {}
        rows = table.get("rows") or []

        # Convert structured table rows into readable text.
        table_text = "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in rows
        )

        evidence.append(
            {
                "modality": "table",
                "rank": rank,
                "content": table_text,
                "source": source or metadata.get("source"),
                "page": metadata.get("page"),
                "table_id": table.get("table_id"),
                "metadata": metadata,
                "scores": table.get("scores", {}),
            }
        )

    return evidence

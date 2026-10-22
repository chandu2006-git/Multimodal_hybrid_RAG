
from typing import Any
from pathlib import Path


def get_document_title(metadata: dict[str, Any]) -> str:
    source = metadata.get("source", "Unknown document")
    return Path(source).stem.replace("_", " ").replace("-", " ")


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Args:
        pages: Output from the PDF loader.
        chunk_size: Maximum chunk size in words.
        chunk_overlap: Number of words shared between chunks.

    Returns:
        Text chunks with source, page, and chunk metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                continue

            chunks.append(
                {
                    "text": (
                        f"[Document: {get_document_title(page.get('metadata', {}))} | "
                        f"Page: {page.get('metadata', {}).get('page', 'Unknown')}]\n"
                        f"{' '.join(chunk_words)}"
                    ),
                    "metadata": {
                        **page.get("metadata", {}),
                        "chunk_index": len(chunks),
                        "start_word": start,
                        "end_word": start + len(chunk_words),
                    },
                }
            )

            if start + chunk_size >= len(words):
                break

    return chunks

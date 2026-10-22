
from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved chunks using a pretrained cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/BAAI/bge-reranker-base",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Reorder candidate chunks by query-passage relevance."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not chunks:
            return []

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive.")

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for chunk, score in zip(chunks, scores):
            result = chunk.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda chunk: chunk["reranker_score"],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

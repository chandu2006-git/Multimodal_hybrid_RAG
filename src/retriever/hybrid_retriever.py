
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    """Normalize text into tokens for BM25 keyword search."""
    return re.findall(r"\b[\w.%-]+\b", text.lower())


class HybridRetriever:
    """Combine dense semantic retrieval with BM25 keyword retrieval."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]

        # Sparse retrieval: BM25 index over document tokens.
        tokenized_corpus = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense retrieval: embed each chunk once.
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using dense search, BM25, and RRF."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive.")
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights cannot be negative.")
        if dense_weight + bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative.")

        # Embed the query and calculate dense similarity scores.
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        dense_scores = self.embeddings @ query_embedding

        # Calculate sparse BM25 scores.
        query_tokens = tokenize(query)
        bm25_scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=float,
        )

        # Rank candidates separately for each retrieval method.
        dense_order = np.argsort(-dense_scores, kind="stable")
        bm25_order = np.argsort(-bm25_scores, kind="stable")

        # Reciprocal Rank Fusion combines ranks, not raw scores.
        fused_scores: dict[int, float] = {}

        for weight, ranking in (
            (dense_weight, dense_order),
            (bm25_weight, bm25_order),
        ):
            for rank, index in enumerate(ranking[:candidate_k], start=1):
                chunk_index = int(index)
                fused_scores[chunk_index] = (
                    fused_scores.get(chunk_index, 0.0)
                    + weight / (rrf_k + rank)
                )

        ranked_indices = sorted(
            fused_scores,
            key=lambda index: (-fused_scores[index], index),
        )[:top_k]

        results = []
        for index in ranked_indices:
            results.append(
                {
                    "text": self.texts[index],
                    "metadata": self.chunks[index].get("metadata", {}),
                    "dense_score": float(dense_scores[index]),
                    "bm25_score": float(bm25_scores[index]),
                    "rrf_score": float(fused_scores[index]),
                }
            )

        return results

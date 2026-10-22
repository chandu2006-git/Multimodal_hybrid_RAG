
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b[\w.%-]+\b", text.lower())


class TableRetriever:
    """Retrieve structured PDF tables using BM25Plus."""

    def __init__(self, tables: list[dict[str, Any]]) -> None:
        if not tables:
            raise ValueError("At least one table is required.")

        self.tables = tables
        self.table_texts = [
            self._table_to_text(table)
            for table in tables
        ]

        tokenized_tables = [
            tokenize(text)
            for text in self.table_texts
        ]

        self.bm25 = BM25Plus(tokenized_tables)

    @staticmethod
    def _table_to_text(
        table: dict[str, Any],
    ) -> str:
        """Create searchable text without modifying the table."""
        metadata = table.get("metadata") or {}
        rows = table.get("rows") or []

        row_text = " ".join(
            " ".join(str(cell) for cell in row)
            for row in rows
        )

        return " ".join(
            str(value)
            for value in [
                table.get("table_id", ""),
                metadata.get("source", ""),
                metadata.get("page", ""),
                row_text,
            ]
            if value is not None
        )

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
    ) -> "TableRetriever":
        """Load extracted tables from a JSON file."""
        path = Path(json_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Table JSON file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            tables = json.load(file)

        if not isinstance(tables, list):
            raise ValueError(
                "Table JSON must contain a list of tables."
            )

        return cls(tables)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring tables for a query."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError(
                "Query must contain searchable tokens."
            )

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self.tables)),
            key=lambda index: (-scores[index], index),
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = dict(self.tables[index])
            result["scores"] = {
                "bm25": float(scores[index])
            }
            results.append(result)

        return results


from pathlib import Path

from retriever.table_retriever import TableRetriever


def main():
    table_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "tables"
        / "AMSL_IEEE_Research_paper_tables.json"
    )

    retriever = TableRetriever.from_json(table_path)

    query = "Which model achieved the highest accuracy?"
    results = retriever.retrieve(
        query,
        top_k=len(retriever.tables),
    )

    print(f"\nQuery: {query}")
    print("-" * 65)

    for rank, result in enumerate(results, start=1):
        table_id = result["table_id"]
        page = result["metadata"].get("page")
        score = result["scores"]["bm25"]

        print(
            f"{rank}. {table_id} | "
            f"page={page} | score={score:.4f}"
        )


if __name__ == "__main__":
    main()

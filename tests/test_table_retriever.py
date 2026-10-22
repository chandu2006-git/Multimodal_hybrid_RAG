
import json

import pytest

from retriever.table_retriever import TableRetriever


@pytest.fixture
def sample_tables():
    return [
        {
            "table_id": "page_3_table_1",
            "rows": [
                ["Parameter", "Value"],
                ["Total Images", "2379"],
                ["Genuine Images", "204"],
                ["Morph Images", "2175"],
            ],
            "metadata": {
                "source": "research.pdf",
                "page": 3,
                "content_type": "table",
            },
        },
        {
            "table_id": "page_7_table_4",
            "rows": [
                ["Model", "Accuracy", "F1 Score"],
                ["MobileNetV2", "96.93%", "0.7840"],
                ["InceptionV3", "97.44%", "0.8283"],
            ],
            "metadata": {
                "source": "research.pdf",
                "page": 7,
                "content_type": "table",
            },
        },
    ]


def test_retrieves_relevant_table(sample_tables):
    retriever = TableRetriever(sample_tables)

    results = retriever.retrieve(
        "Which model achieved the highest accuracy?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["table_id"] == "page_7_table_4"
    assert results[0]["metadata"]["page"] == 7
    assert "InceptionV3" in str(results[0]["rows"])
    assert isinstance(results[0]["scores"]["bm25"], float)


def test_relevant_table_ranks_above_dataset_table(sample_tables):
    retriever = TableRetriever(sample_tables)

    results = retriever.retrieve(
        "model accuracy F1 score",
        top_k=2,
    )

    assert results[0]["table_id"] == "page_7_table_4"


def test_rejects_empty_query(sample_tables):
    retriever = TableRetriever(sample_tables)

    with pytest.raises(
        ValueError,
        match="Query must not be empty",
    ):
        retriever.retrieve("   ")


def test_rejects_query_without_searchable_tokens(sample_tables):
    retriever = TableRetriever(sample_tables)

    with pytest.raises(
        ValueError,
        match="searchable tokens",
    ):
        retriever.retrieve("!!!")


def test_rejects_invalid_top_k(sample_tables):
    retriever = TableRetriever(sample_tables)

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        retriever.retrieve("accuracy", top_k=0)


def test_loads_tables_from_json(tmp_path, sample_tables):
    json_path = tmp_path / "tables.json"
    json_path.write_text(
        json.dumps(sample_tables),
        encoding="utf-8",
    )

    retriever = TableRetriever.from_json(json_path)

    assert len(retriever.tables) == 2
    assert retriever.tables[0]["table_id"] == "page_3_table_1"


def test_rejects_missing_json_file(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Table JSON file not found",
    ):
        TableRetriever.from_json(tmp_path / "missing.json")


def test_preserves_original_table_rows(sample_tables):
    original_rows = sample_tables[1]["rows"]

    retriever = TableRetriever(sample_tables)
    results = retriever.retrieve("InceptionV3 accuracy", top_k=1)

    assert results[0]["rows"] == original_rows
    assert results[0]["metadata"]["source"] == "research.pdf"

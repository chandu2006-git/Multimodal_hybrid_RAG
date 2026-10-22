
from pathlib import Path

import pytest

from ingestion.table_extractor import extract_tables


PDF_PATH = Path("data/raw_pdfs/AMSL_IEEE_Research_paper.pdf")


@pytest.mark.skipif(
    not PDF_PATH.exists(),
    reason="Real AMSL research paper is not available",
)
def test_real_paper_table_regression():
    tables = extract_tables(PDF_PATH)

    # Locate tables by their source page.
    page_3 = [
        table for table in tables
        if table["metadata"]["page"] == 3
    ]
    page_5 = [
        table for table in tables
        if table["metadata"]["page"] == 5
    ]
    page_7 = [
        table for table in tables
        if table["metadata"]["page"] == 7
    ]

    assert len(page_3) == 1
    assert len(page_5) == 1
    assert len(page_7) == 2

    dataset_table = page_3[0]["rows"]
    config_table = page_5[0]["rows"]

    # Header plus 12 dataset parameters.
    assert len(dataset_table) == 13
    assert all(len(row) == 2 for row in dataset_table)

    dataset_values = {
        row[0]: row[1]
        for row in dataset_table[1:]
    }
    assert dataset_values["Total Images"] == "2379"

    # Header plus 13 training parameters.
    assert len(config_table) == 14
    assert all(len(row) == 2 for row in config_table)

    # Identify the performance table by its header.
    performance_tables = [
        table["rows"]
        for table in page_7
        if table["rows"]
        and table["rows"][0][0] == "Model"
    ]

    assert len(performance_tables) == 1

    performance_table = performance_tables[0]

    # Header plus four models; six columns per row.
    assert len(performance_table) == 5
    assert all(len(row) == 6 for row in performance_table)

    performance_by_model = {
        row[0]: row[1:]
        for row in performance_table[1:]
    }

    assert performance_by_model["MobileNetV2"][0] == "96.93"
    assert performance_by_model["InceptionV3"][0] == "97.44"

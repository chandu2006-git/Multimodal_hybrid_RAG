
import pytest

from ingestion.table_extractor import (
    _repair_columnar_table,
    extract_tables,
)


def test_repairs_multiline_two_column_table():
    rows = [
        ["Parameter", "Value"],
        [
            "Batch Size\nOptimizer\nEpochs",
            "32\nAdam\n20",
        ],
    ]

    result = _repair_columnar_table(rows)

    assert result == [
        ["Parameter", "Value"],
        ["Batch Size", "32"],
        ["Optimizer", "Adam"],
        ["Epochs", "20"],
    ]


def test_repairs_multiline_performance_table():
    rows = [
        ["Model", "Accuracy", "Precision"],
        [
            "CNN\nMobileNetV2\nInceptionV3",
            "91.34\n96.93\n97.44",
            "N/A\n1.0000\n1.0000",
        ],
    ]

    result = _repair_columnar_table(rows)

    assert result[1] == ["CNN", "91.34", "N/A"]
    assert result[2] == ["MobileNetV2", "96.93", "1.0000"]
    assert result[3] == ["InceptionV3", "97.44", "1.0000"]
    assert len(result) == 4


def test_preserves_already_structured_table():
    rows = [
        ["Model", "Accuracy"],
        ["CNN", "91.34"],
        ["InceptionV3", "97.44"],
    ]

    assert _repair_columnar_table(rows) == rows


def test_does_not_guess_when_column_lengths_differ():
    rows = [
        ["Model", "Accuracy"],
        ["CNN\nInceptionV3", "91.34\n97.44\n88.00"],
    ]

    assert _repair_columnar_table(rows) == rows


def test_normalizes_whitespace_and_empty_cells():
    rows = [
        [" Parameter ", " Value "],
        [" Batch Size \n Optimizer ", " 32 \n Adam "],
    ]

    assert _repair_columnar_table(rows) == [
        ["Parameter", "Value"],
        ["Batch Size", "32"],
        ["Optimizer", "Adam"],
    ]


def test_extract_tables_rejects_missing_pdf(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_tables(missing_pdf)


def test_extract_tables_rejects_non_pdf(tmp_path):
    text_file = tmp_path / "paper.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extract_tables(text_file)

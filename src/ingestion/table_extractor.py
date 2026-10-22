
from pathlib import Path
from typing import Any

import pymupdf


def _clean_cell(value: Any) -> str:
    """Normalize a cell while preserving meaningful internal line breaks."""
    if value is None:
        return ""

    lines = [line.strip() for line in str(value).splitlines()]
    return "\n".join(line for line in lines if line)


def _repair_columnar_table(rows: list[list[Any]]) -> list[list[str]]:
    """
    Repair tables where the extractor puts multiple logical rows
    into each cell of a single data row.

    Repair is applied only when every data cell has the same
    number of non-empty lines. Otherwise, preserve the original
    structure rather than guessing.
    """
    cleaned = [
        [_clean_cell(cell) for cell in row]
        for row in rows
    ]

    if len(cleaned) != 2:
        return cleaned

    headers, values = cleaned

    if not headers or len(headers) != len(values):
        return cleaned

    split_columns = [
        [line.strip() for line in cell.splitlines() if line.strip()]
        for cell in values
    ]

    lengths = [len(column) for column in split_columns]

    if not lengths or max(lengths) <= 1:
        return cleaned

    # Every column must align. Do not infer missing values.
    if len(set(lengths)) != 1:
        return cleaned

    data_rows = [
        [split_columns[column][row] for column in range(len(headers))]
        for row in range(lengths[0])
    ]

    return [headers, *data_rows]


def extract_tables(pdf_path: str | Path) -> list[dict]:
    """
    Extract tables from a PDF using PyMuPDF.

    Returns one record per detected table, including its rows
    and source/page metadata.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    extracted_tables = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            finder = page.find_tables()

            for table_number, table in enumerate(finder.tables, start=1):
                raw_rows = table.extract()

                if not raw_rows:
                    continue

                rows = _repair_columnar_table(raw_rows)

                # Skip tables containing no meaningful cell content.
                if not any(cell for row in rows for cell in row):
                    continue

                extracted_tables.append(
                    {
                        "table_id": f"page_{page_number}_table_{table_number}",
                        "rows": rows,
                        "metadata": {
                            "source": path.name,
                            "page": page_number,
                            "content_type": "table",
                        },
                    }
                )

    return extracted_tables

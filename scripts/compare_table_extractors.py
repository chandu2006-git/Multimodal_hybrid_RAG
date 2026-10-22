
from pathlib import Path

import pdfplumber

from ingestion.table_extractor import extract_tables


PDF_PATH = Path("data/raw_pdfs/AMSL_IEEE_Research_paper.pdf")


def compare_pymupdf():
    print("\n" + "=" * 70)
    print("PYMUPDF — CURRENT IMPLEMENTATION")
    print("=" * 70)

    tables = extract_tables(PDF_PATH)

    for table in tables:
        print(
            f"\nPage {table['metadata']['page']} | "
            f"{table['table_id']}"
        )

        for row in table["rows"][:5]:
            print(row)


def compare_pdfplumber():
    print("\n" + "=" * 70)
    print("PDFPLUMBER — ALTERNATIVE IMPLEMENTATION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table_number, table in enumerate(tables, start=1):
                if not table:
                    continue

                print(
                    f"\nPage {page_number} | "
                    f"Table {table_number} | "
                    f"Rows: {len(table)}"
                )

                for row in table[:5]:
                    print(row)


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    compare_pymupdf()
    compare_pdfplumber()

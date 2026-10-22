
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images


def ingest_document(
    pdf_path: str | Path,
    image_root: str | Path = "data/images",
    table_root: str | Path = "data/processed/tables",
) -> dict:
    """Extract text, tables, embedded images, and page renders."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    image_dir = Path(image_root) / pdf_path.stem
    table_dir = Path(table_root)
    table_dir.mkdir(parents=True, exist_ok=True)

    text_pages = load_pdf(pdf_path)
    tables = extract_tables(pdf_path)
    images = extract_images(pdf_path, image_dir)

    # Save tables as JSON for inspection and later processing.
    import json

    table_output = table_dir / f"{pdf_path.stem}_tables.json"
    table_output.write_text(
        json.dumps(tables, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "source": pdf_path.name,
        "text_pages": text_pages,
        "tables": tables,
        "images": images,
        "table_output": str(table_output),
    }

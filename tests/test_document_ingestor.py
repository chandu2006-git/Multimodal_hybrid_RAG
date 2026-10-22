
import json

import pymupdf

from ingestion.document_ingestor import ingest_document


def create_sample_pdf(pdf_path):
    """Create a small PDF for isolated testing."""
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Multimodal RAG ingestion test")
        document.save(pdf_path)


def test_ingestion_extracts_text_and_creates_outputs(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_sample_pdf(pdf_path)

    result = ingest_document(
        pdf_path,
        image_root=tmp_path / "images",
        table_root=tmp_path / "tables",
    )

    assert result["source"] == "sample.pdf"
    assert len(result["text_pages"]) == 1
    assert "Multimodal RAG ingestion test" in result["text_pages"][0]["text"]
    assert result["text_pages"][0]["metadata"]["page"] == 1

    assert result["tables"] == []
    assert (tmp_path / "tables" / "sample_tables.json").exists()

    saved_tables = json.loads(
        (tmp_path / "tables" / "sample_tables.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_tables == []


def test_ingestion_missing_pdf_raises_error(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    try:
        ingest_document(
            missing_pdf,
            image_root=tmp_path / "images",
            table_root=tmp_path / "tables",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")

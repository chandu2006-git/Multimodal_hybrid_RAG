
import pytest
import pymupdf

from ingestion.pdf_loader import load_pdf


def test_missing_pdf_raises_error():
    with pytest.raises(FileNotFoundError):
        load_pdf("does_not_exist.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        load_pdf(text_file)


def test_load_pdf_extracts_text_and_page_metadata(tmp_path):
    # Create a tiny PDF specifically for testing.
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Multimodal RAG test document")
        document.save(pdf_path)

    # Run our actual PDF loader.
    pages = load_pdf(pdf_path)

    # Verify the extracted content and metadata.
    assert len(pages) == 1
    assert "Multimodal RAG test document" in pages[0]["text"]
    assert pages[0]["metadata"]["source"] == "sample.pdf"
    assert pages[0]["metadata"]["page"] == 1

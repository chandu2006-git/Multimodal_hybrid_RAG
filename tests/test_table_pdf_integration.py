
import pymupdf

from ingestion.table_extractor import extract_tables


def test_extract_tables_from_synthetic_pdf(tmp_path):
    pdf_path = tmp_path / "synthetic_table.pdf"

    # Create a real PDF containing a simple grid table.
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)

    rows = [
        ["Model", "Accuracy"],
        ["CNN", "91.34"],
        ["InceptionV3", "97.44"],
    ]

    table_data = [
        [pymupdf.Rect(50 + col * 200, 80 + row * 50,
                      250 + col * 200, 130 + row * 50),
         value]
        for row, row_values in enumerate(rows)
        for col, value in enumerate(row_values)
    ]

    for rect, value in table_data:
        page.draw_rect(rect, width=1)
        page.insert_textbox(
            pymupdf.Rect(
                rect.x0 + 5,
                rect.y0 + 10,
                rect.x1 - 5,
                rect.y1 - 5,
            ),
            value,
            fontsize=11,
        )

    doc.save(pdf_path)
    doc.close()

    tables = extract_tables(pdf_path)

    assert len(tables) >= 1

    extracted_rows = tables[0]["rows"]

    assert extracted_rows[0] == ["Model", "Accuracy"]
    assert ["CNN", "91.34"] in extracted_rows
    assert ["InceptionV3", "97.44"] in extracted_rows

    assert tables[0]["metadata"]["source"] == pdf_path.name
    assert tables[0]["metadata"]["page"] == 1
    assert tables[0]["metadata"]["content_type"] == "table"

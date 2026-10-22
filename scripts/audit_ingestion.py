
from pathlib import Path

import pymupdf

from ingestion.pdf_loader import load_pdf
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images


PDF_PATH = Path("data/raw_pdfs/AMSL_IEEE_Research_paper.pdf")
OUTPUT_DIR = Path("data/images") / PDF_PATH.stem
TABLE_DIR = Path("data/processed/tables")


def audit_text():
    print("\n" + "=" * 55)
    print("1. TEXT EXTRACTION AUDIT")
    print("=" * 55)

    pages = load_pdf(PDF_PATH)

    print(f"Pages containing text: {len(pages)}")

    total_chars = 0
    for page in pages:
        text = page["text"]
        total_chars += len(text)

        print(
            f"Page {page['metadata']['page']}: "
            f"{len(text)} characters"
        )

    print(f"Total extracted characters: {total_chars}")

    return pages


def audit_tables():
    print("\n" + "=" * 55)
    print("2. TABLE EXTRACTION AUDIT")
    print("=" * 55)

    tables = extract_tables(PDF_PATH)
    print(f"Tables detected: {len(tables)}")

    for table in tables:
        rows = table["rows"]
        widths = [len(row) for row in rows]

        print(f"\n{table['table_id']}")
        print(f"Page: {table['metadata']['page']}")
        print(f"Rows extracted: {len(rows)}")
        print(f"Columns per row: {widths}")

        for row in rows[:3]:
            print("  ", row)

    print("\nExpected values to inspect manually:")
    print("- Dataset total: 2379")
    print("- InceptionV3 accuracy: 97.44")
    print("- MobileNetV2 accuracy: 96.93")

    return tables


def audit_images():
    print("\n" + "=" * 55)
    print("3. IMAGE AND PAGE-RENDER AUDIT")
    print("=" * 55)

    assets = extract_images(
        PDF_PATH,
        OUTPUT_DIR,
        render_pages=True,
        dpi=150,
    )

    embedded = [
        asset for asset in assets
        if asset["metadata"]["content_type"] == "image"
    ]
    renders = [
        asset for asset in assets
        if asset["metadata"]["content_type"] == "page_render"
    ]

    print(f"Embedded image assets: {len(embedded)}")
    print(f"Page renders: {len(renders)}")
    print(f"Total output assets: {len(assets)}")

    for asset in assets:
        path = Path(asset["path"])
        print(
            f"Page {asset['metadata']['page']}: "
            f"{asset['asset_id']} | "
            f"{path.stat().st_size:,} bytes | "
            f"{path.name}"
        )

    return assets


def audit_pdf_pages():
    print("\n" + "=" * 55)
    print("4. FIGURE / GRAPH INSPECTION")
    print("=" * 55)

    with pymupdf.open(PDF_PATH) as document:
        print(f"Total PDF pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):
            images = page.get_images(full=True)
            drawings = page.get_drawings()
            text = page.get_text("text")

            print(
                f"Page {page_number}: "
                f"embedded images={len(images)}, "
                f"vector drawings={len(drawings)}, "
                f"text chars={len(text)}"
            )

    print(
        "\nNote: image and drawing counts do not directly equal "
        "the number of scientific figures or graphs."
    )


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Research paper not found: {PDF_PATH}"
        )

    audit_text()
    audit_tables()
    audit_images()
    audit_pdf_pages()

    print("\n" + "=" * 55)
    print("AUDIT COMPLETE")
    print("=" * 55)
    print("Counts show extraction output, not guaranteed accuracy.")


if __name__ == "__main__":
    main()

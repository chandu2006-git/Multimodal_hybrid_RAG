
import pymupdf

from ingestion.image_extractor import extract_images


def test_page_render_is_created(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "images"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Image extraction test")
        document.save(pdf_path)

    assets = extract_images(
        pdf_path,
        output_dir,
        render_pages=True,
        dpi=100,
    )

    renders = [
        asset for asset in assets
        if asset["metadata"]["content_type"] == "page_render"
    ]

    assert len(renders) == 1

    image_path = output_dir / "page_1_render.png"
    assert image_path.exists()
    assert image_path.stat().st_size > 0

    # Verify the PNG can actually be opened.
    pixmap = pymupdf.Pixmap(str(image_path))
    assert pixmap.width > 0
    assert pixmap.height > 0

    assert renders[0]["metadata"]["page"] == 1
    assert renders[0]["metadata"]["source"] == "sample.pdf"


from pathlib import Path

import pymupdf


def extract_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    render_pages: bool = True,
    dpi: int = 150,
) -> list[dict]:
    """Extract embedded images and optionally render PDF pages."""
    path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            # Embedded raster images.
            for image_index, image_info in enumerate(
                page.get_images(full=True), start=1
            ):
                xref = image_info[0]
                image_data = document.extract_image(xref)

                extension = image_data["ext"]
                image_path = (
                    output_dir
                    / f"page_{page_number}_image_{image_index}.{extension}"
                )

                image_path.write_bytes(image_data["image"])

                extracted.append(
                    {
                        "asset_id": f"page_{page_number}_image_{image_index}",
                        "path": str(image_path),
                        "metadata": {
                            "source": path.name,
                            "page": page_number,
                            "content_type": "image",
                            "extraction_method": "embedded",
                        },
                    }
                )

            # Full-page rendering captures vector graphics too.
            if render_pages:
                page_path = (
                    output_dir / f"page_{page_number}_render.png"
                )
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                    alpha=False,
                )
                pixmap.save(page_path)

                extracted.append(
                    {
                        "asset_id": f"page_{page_number}_render",
                        "path": str(page_path),
                        "metadata": {
                            "source": path.name,
                            "page": page_number,
                            "content_type": "page_render",
                            "extraction_method": "page_render",
                            "dpi": dpi,
                        },
                    }
                )

    return extracted

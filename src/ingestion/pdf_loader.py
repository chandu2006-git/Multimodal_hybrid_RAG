
from pathlib import Path

import pymupdf


def load_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract PDF text and layout information.

    Preserves:
        - Page text
        - Page number
        - Text blocks and lines
        - Font names and sizes
        - Bold/italic flags
        - Bounding boxes
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages = []

    with pymupdf.open(path) as document:

        for page_number, page in enumerate(document, start=1):

            # Extract ordinary text.
            text = page.get_text("text").strip()

            # Extract structured layout.
            page_dict = page.get_text("dict")

            layout_blocks = []

            for block in page_dict.get("blocks", []):

                if block.get("type") != 0:
                    continue

                lines = []

                for line in block.get("lines", []):

                    spans = []

                    for span in line.get("spans", []):

                        span_text = span.get("text", "").strip()

                        if not span_text:
                            continue

                        spans.append({
                            "text": span_text,
                            "font": span.get("font", ""),
                            "font_size": round(
                                span.get("size", 0), 2
                            ),
                            "flags": span.get("flags", 0),
                            "bbox": span.get("bbox"),
                        })

                    if not spans:
                        continue

                    line_text = " ".join(
                        span["text"] for span in spans
                    )

                    lines.append({
                        "text": line_text,
                        "spans": spans,
                        "bbox": line.get("bbox"),
                    })

                if lines:
                    layout_blocks.append({
                        "lines": lines,
                        "bbox": block.get("bbox"),
                    })

            pages.append({
                "text": text,
                "layout": {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": layout_blocks,
                },
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                },
            })

    return pages

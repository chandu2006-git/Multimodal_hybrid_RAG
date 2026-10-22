
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from rag.pipeline import MultimodalRAGPipeline


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    pipeline = MultimodalRAGPipeline(
        pdf_path=PROJECT_ROOT / "data" / "raw_pdfs"
        / "AMSL_IEEE_Research_paper.pdf",
        image_index_dir=PROJECT_ROOT / "data" / "processed"
        / "image_index",
        image_root=PROJECT_ROOT / "data" / "images",
        table_root=PROJECT_ROOT / "data" / "processed" / "tables",
    )

    query = (
        "Which model achieved the highest accuracy, "
        "and what was its reported F1 score?"
    )

    result = pipeline.retrieve(query)

    print(f"\nQuery: {query}")

    for item in result["evidence"]:
        print("\n" + "=" * 60)
        print(f"Modality: {item['modality']}")
        print(f"Source: {item.get('source')}")
        print(f"Page: {item.get('page')}")
        print(f"Content:\n{item.get('content', '')[:1800]}")

if __name__ == "__main__":
    main()

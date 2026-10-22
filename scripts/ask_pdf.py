
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from rag.pipeline import MultimodalRAGPipeline


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "raw_pdfs"
        / "AMSL_IEEE_Research_paper.pdf"
    )

    image_index_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "image_index"
    )

    pipeline = MultimodalRAGPipeline(
        pdf_path=pdf_path,
        image_index_dir=image_index_dir,
        image_root=PROJECT_ROOT / "data" / "images",
        table_root=PROJECT_ROOT / "data" / "processed" / "tables",
    )

    print("\nMultimodal RAG — Research Paper Assistant")
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            result = pipeline.ask(query)

            print("\nAnswer:")
            print(result["answer"])

            print("\nRetrieved sources:")
            for source in result["sources"]:
                print(
                    f"- {source.get('source') or 'Unknown source'}"
                    f", page {source.get('page') or 'unknown'}"
                )

            print(f"\nModel: {result['model']}\n")

        except Exception as exc:
            print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    main()

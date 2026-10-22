import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.image_retriever import ImageRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "AMSL_IEEE_Research_paper"
)


def main() -> None:
    retriever = ImageRetriever(IMAGE_DIR)

    print(f"\nIndexed images: {len(retriever.records)}")
    print(f"Device: {retriever.device}")

    queries = [
        "A confusion matrix showing face morph detection results",
        "A diagram of a deep learning neural network architecture",
        "A chart comparing accuracy of CNN models",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['filename']} | "
                f"page={result['page']} | "
                f"similarity={result['similarity']:.4f}"
            )
            print(f"   Path: {result['image_path']}")


if __name__ == "__main__":
    main()

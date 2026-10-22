
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.image_retriever import ImageRetriever

IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "AMSL_IEEE_Research_paper"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "image_index"


def main():
    retriever = ImageRetriever(IMAGE_DIR, build_index=False)
    retriever.load_index(INDEX_DIR)

    query = "A chart comparing accuracy of CNN models"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['filename']} | "
            f"page={result['page']} | "
            f"similarity={result['similarity']:.4f}"
        )


if __name__ == "__main__":
    main()

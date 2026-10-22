
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.image_retriever import ImageRetriever

IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "AMSL_IEEE_Research_paper"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "image_index"


def main():
    retriever = ImageRetriever(IMAGE_DIR)

    print(f"Indexed images: {len(retriever.records)}")
    print(f"Device: {retriever.device}")

    retriever.save_index(INDEX_DIR)


if __name__ == "__main__":
    main()

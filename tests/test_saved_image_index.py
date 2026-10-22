
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.image_retriever import ImageRetriever

IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "AMSL_IEEE_Research_paper"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "image_index"


def main():
    # Initialize the model, but don't index the image directory.
    retriever = ImageRetriever.__new__(ImageRetriever)
    retriever.image_dir = IMAGE_DIR
    retriever.device = "cpu"
    retriever.model = None
    retriever.processor = None
    retriever.records = []
    retriever.image_embeddings = None

    retriever.load_index(INDEX_DIR)

    print("Embedding shape:", tuple(retriever.image_embeddings.shape))
    print("Metadata records:", len(retriever.records))
    print("First image:", retriever.records[0]["filename"])


if __name__ == "__main__":
    main()

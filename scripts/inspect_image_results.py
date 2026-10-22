from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.image_retriever import ImageRetriever

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "AMSL_IEEE_Research_paper"
)


def main() -> None:
    retriever = ImageRetriever(IMAGE_DIR)

    queries = [
        "A confusion matrix showing face morph detection results",
        "A diagram of a deep learning neural network architecture",
        "A chart comparing accuracy of CNN models",
    ]

    for query in queries:
        results = retriever.retrieve(query, top_k=3)

        figure, axes = plt.subplots(1, len(results), figsize=(15, 6))
        if len(results) == 1:
            axes = [axes]

        figure.suptitle(query, fontsize=12)

        for axis, result in zip(axes, results):
            with Image.open(result["image_path"]) as image:
                axis.imshow(image.convert("RGB"))

            axis.set_title(
                f"{result['filename']}\n"
                f"Page: {result['page']} | "
                f"Similarity: {result['similarity']:.3f}",
                fontsize=9,
            )
            axis.axis("off")

        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from rag.answer_generator import AnswerGenerator


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    generator = AnswerGenerator()

    result = generator.generate(
        query="What is retrieval-augmented generation?",
        evidence=[
            {
                "modality": "text",
                "content": (
                    "Retrieval-augmented generation (RAG) "
                    "combines information retrieval with "
                    "language-model generation."
                ),
                "source": "test_document.pdf",
                "page": 1,
            }
        ],
    )

    print("Gemini connection successful!")
    print("\nAnswer:")
    print(result["answer"])


if __name__ == "__main__":
    main()

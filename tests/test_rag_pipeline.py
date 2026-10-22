
from pathlib import Path

import pytest

from rag.pipeline import MultimodalRAGPipeline


class FakeTextRetriever:
    def retrieve(self, query, top_k=3):
        return [
            {
                "text": "InceptionV3 achieved 97.44% accuracy.",
                "metadata": {
                    "source": "research.pdf",
                    "page": 7,
                },
                "dense_score": 0.9,
                "bm25_score": 3.0,
                "rrf_score": 0.03,
            }
        ]


class FakeImageRetriever:
    def retrieve(self, query, top_k=3):
        return [
            {
                "filename": "page_7_image_1.png",
                "image_path": "data/images/page_7_image_1.png",
                "source": "research",
                "page": 7,
                "similarity": 0.3,
            }
        ]


class FakeTableRetriever:
    def retrieve(self, query, top_k=3):
        return [
            {
                "table_id": "page_7_table_2",
                "rows": [
                    ["Model", "Accuracy"],
                    ["InceptionV3", "97.44%"],
                ],
                "metadata": {
                    "source": "research.pdf",
                    "page": 7,
                },
                "scores": {"bm25": 2.5},
            }
        ]


class FakeAnswerGenerator:
    def generate(self, query, evidence):
        assert query == "Which model had the highest accuracy?"
        assert {item["modality"] for item in evidence} == {
            "text",
            "image",
            "table",
        }

        return {
            "answer": (
                "InceptionV3 achieved 97.44% accuracy "
                "[Source: research.pdf, page: 7]."
            ),
            "sources": [
                {"source": "research.pdf", "page": 7}
            ],
            "model": "test-model",
        }


def test_ask_integrates_retrieval_and_generation(tmp_path):
    # Build without running the expensive constructor.
    pipeline = MultimodalRAGPipeline.__new__(
        MultimodalRAGPipeline
    )

    pipeline.pdf_path = tmp_path / "research.pdf"
    pipeline.text_top_k = 3
    pipeline.image_top_k = 3
    pipeline.table_top_k = 3

    pipeline.document = {
        "source": "research.pdf",
        "text_pages": [],
        "tables": [],
        "images": [],
        "table_output": "",
    }

    pipeline.text_retriever = FakeTextRetriever()
    pipeline.image_retriever = FakeImageRetriever()
    pipeline.table_retriever = FakeTableRetriever()
    pipeline.answer_generator = FakeAnswerGenerator()

    result = pipeline.ask(
        "Which model had the highest accuracy?"
    )

    assert "InceptionV3" in result["answer"]
    assert len(result["evidence"]) == 3
    assert result["model"] == "test-model"
    assert result["sources"][0]["page"] == 7


def test_retrieve_rejects_empty_query():
    pipeline = MultimodalRAGPipeline.__new__(
        MultimodalRAGPipeline
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        pipeline.retrieve("   ")

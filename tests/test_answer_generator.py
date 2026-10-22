
import pytest

from rag.answer_generator import AnswerGenerator


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        AnswerGenerator(api_key=None)


def test_rejects_empty_query():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate("  ", [{"modality": "text"}])


def test_returns_fallback_when_evidence_is_empty():
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.model_name = "test-model"

    result = generator.generate(
        "What is the model accuracy?",
        [],
    )

    assert "couldn't find relevant evidence" in result["answer"]
    assert result["sources"] == []
    assert result["model"] == "test-model"


def test_formats_evidence_with_page_reference():
    evidence = [
        {
            "modality": "table",
            "content": "InceptionV3 | 97.44%",
            "source": "research.pdf",
            "page": 7,
        }
    ]

    formatted = AnswerGenerator._format_evidence(evidence)

    assert "InceptionV3 | 97.44%" in formatted
    assert "Source: research.pdf, page: 7" in formatted


from rag.evidence_builder import build_evidence


def test_build_evidence_combines_text_and_images():
    text_results = [
        {
            "text": "InceptionV3 achieved 97.44% accuracy.",
            "metadata": {
                "page": 7,
                "source": "research.pdf",
            },
            "dense_score": 0.82,
            "bm25_score": 4.1,
            "rrf_score": 0.03,
        }
    ]

    image_results = [
        {
            "filename": "page_7_image_1.png",
            "image_path": "data/images/page_7_image_1.png",
            "page": 7,
            "source": "research",
            "similarity": 0.31,
        }
    ]

    evidence = build_evidence(
        text_results,
        image_results,
        source="research.pdf",
    )

    assert len(evidence) == 2

    assert evidence[0]["modality"] == "text"
    assert evidence[0]["page"] == 7
    assert evidence[0]["source"] == "research.pdf"

    assert evidence[1]["modality"] == "image"
    assert evidence[1]["source"] == "research.pdf"
    assert evidence[1]["scores"]["similarity"] == 0.31


def test_build_evidence_includes_structured_tables():
    table_results = [
        {
            "table_id": "page_3_table_1",
            "rows": [
                ["Parameter", "Value"],
                ["Total Images", "2379"],
            ],
            "metadata": {
                "source": "AMSL_IEEE_Research_paper.pdf",
                "page": 3,
                "content_type": "table",
            },
        }
    ]

    evidence = build_evidence(
        text_results=[],
        image_results=[],
        table_results=table_results,
    )

    assert len(evidence) == 1
    assert evidence[0]["modality"] == "table"
    assert evidence[0]["page"] == 3
    assert evidence[0]["table_id"] == "page_3_table_1"
    assert "Total Images | 2379" in evidence[0]["content"]
    assert evidence[0]["source"] == "AMSL_IEEE_Research_paper.pdf"


def test_build_evidence_handles_empty_results():
    evidence = build_evidence(
        text_results=[],
        image_results=[],
    )

    assert evidence == []


def test_build_evidence_handles_empty_table_results():
    evidence = build_evidence(
        text_results=[],
        image_results=[],
        table_results=[],
    )

    assert evidence == []

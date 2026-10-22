
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_evaluation_dataset.json"
)


def load_evaluation_dataset(
    path: Path = DATASET_PATH,
) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    if not questions:
        raise ValueError("Evaluation dataset is empty")

    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found")

    return questions


def get_relevant_pages(question: dict) -> set[int]:
    return set(question.get("relevant_pages", []))

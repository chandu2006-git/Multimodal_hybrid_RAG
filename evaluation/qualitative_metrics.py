
"""Optional LLM-as-judge metrics for Multimodal Hybrid RAG."""

import json
import os


QUALITATIVE_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "citation_correctness",
    "correct_abstention",
]


def judge_answer(
    *,
    question: str,
    answer: str,
    evidence,
    citations,
    answerable: bool,
    model_name: str | None = None,
) -> dict:
    """Score a generated answer using Gemini, when explicitly requested."""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to enable the judge."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Install google-genai to enable the qualitative judge."
        ) from exc

    model = model_name or os.getenv(
        "RAG_EVAL_JUDGE_MODEL",
        "gemini-2.5-flash",
    )

    prompt = f"""
Evaluate the following Retrieval-Augmented Generation answer.

Return ONLY a valid JSON object with numeric scores from 0 to 1 for:
faithfulness, answer_relevancy, citation_correctness, correct_abstention.

Scoring criteria:
1. faithfulness: Are the answer's factual claims supported by the evidence?
2. answer_relevancy: Does the answer directly address the question?
3. citation_correctness: Do the cited sources support the answer?
   If citations are absent, normally score 0.
4. correct_abstention:
   - If answerable is false, score 1 when the system clearly abstains
     because the evidence is insufficient, without inventing facts.
   - If answerable is true, score 1 when the system attempts a grounded
     answer, and 0 when it unnecessarily refuses.
   - Use partial scores when appropriate.

Question:
{question}

Dataset answerable: {answerable}

Generated answer:
{answer}

Citations:
{json.dumps(citations, ensure_ascii=False, default=str)}

Retrieved evidence:
{json.dumps(evidence, ensure_ascii=False, default=str)[:12000]}
""".strip()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    response_text = (response.text or "").strip()

    if response_text.startswith("```"):
        response_text = response_text.strip("`").strip()

        if response_text.lower().startswith("json"):
            response_text = response_text[4:].strip()

    try:
        scores = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge returned invalid JSON: {response_text[:300]}"
        ) from exc

    result = {}

    for metric in QUALITATIVE_METRICS:
        value = scores.get(metric)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[metric] = max(0.0, min(1.0, float(value)))
        else:
            result[metric] = None

    result["judge_model"] = model

    return result

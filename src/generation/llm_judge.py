
import json
import os
import re
from typing import Any

from google import genai


class LLMJudge:
    """
    Evaluate a RAG answer against the question and retrieved evidence.

    Scores each dimension from 1 to 5:
    - Faithfulness
    - Answer relevance
    - Completeness
    - Citation accuracy
    """

    DIMENSIONS = (
        "faithfulness",
        "answer_relevance",
        "completeness",
        "citation_accuracy",
    )

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Set it in your .env file."
            )

        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _format_evidence(
        evidence: list[dict[str, Any]],
    ) -> str:
        """Format retrieved evidence with its source metadata."""
        if not evidence:
            return "No evidence was retrieved."

        sections = []

        for index, item in enumerate(evidence, start=1):
            source = item.get("source") or "Unknown source"
            page = item.get("page")
            modality = item.get("modality", "unknown")
            content = item.get("content", "")

            location = f"Source: {source}"
            if page is not None:
                location += f", page: {page}"

            sections.append(
                f"[Evidence {index}]\n"
                f"Modality: {modality}\n"
                f"{location}\n"
                f"Content: {content}"
            )

        return "\n\n".join(sections)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON even if the model wraps it in a code fence."""
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The judge returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "The judge response must be a JSON object."
            )

        return result

    @staticmethod
    def _validate_result(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate score ranges and required evaluation fields."""
        required = (
            "faithfulness",
            "answer_relevance",
            "completeness",
            "citation_accuracy",
        )

        for dimension in required:
            item = result.get(dimension)

            if not isinstance(item, dict):
                raise ValueError(
                    f"Missing evaluation dimension: {dimension}"
                )

            score = item.get("score")

            # bool is a subclass of int, so reject it explicitly.
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 1 <= score <= 5
            ):
                raise ValueError(
                    f"{dimension}.score must be between 1 and 5."
                )

            item["score"] = float(score)
            item["reason"] = str(item.get("reason", "")).strip()

        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]

        result["issues"] = [str(issue) for issue in issues]

        result["overall_assessment"] = str(
            result.get("overall_assessment", "")
        ).strip()

        return result

    def evaluate(
        self,
        query: str,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate one generated answer.

        Returns validated scores, explanations, issues, and
        an overall assessment.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not evidence:
            raise ValueError(
                "Cannot evaluate the answer without retrieved evidence."
            )

        formatted_evidence = self._format_evidence(evidence)

        prompt = f"""
You are an independent evaluator for a scientific-paper RAG system.

Evaluate the generated answer ONLY against the user's question
and the supplied retrieved evidence. Do not use outside knowledge
to treat unsupported claims as correct.

Score each dimension from 1 to 5.

SCORING RUBRIC

1. faithfulness:
   5 = All factual claims are supported by the evidence.
   4 = Mostly supported; only minor unsupported details.
   3 = A mixture of supported and unsupported claims.
   2 = Several important claims lack support or contradict evidence.
   1 = Mostly unsupported or contradicted by the evidence.

2. answer_relevance:
   5 = Directly and precisely answers the question.
   4 = Relevant, with minor unnecessary content.
   3 = Partially answers the question.
   2 = Mostly misses the user's intent.
   1 = Does not answer the question.

3. completeness:
   5 = Addresses every part of the question.
   4 = Omits only a minor detail.
   3 = Leaves out at least one meaningful part.
   2 = Leaves out most requested information.
   1 = Essentially incomplete.

4. citation_accuracy:
   5 = Citations match the supplied evidence's source and page.
   4 = Citations are mostly accurate with minor omissions.
   3 = Some citations are missing, ambiguous, or mismatched.
   2 = Multiple citation errors.
   1 = Citations are absent when needed or materially incorrect.

IMPORTANT RULES
- Do not assume a claim is true merely because it sounds plausible.
- Check numerical claims against the exact evidence values.
- Treat missing evidence as missing; do not invent support.
- Check citation filenames and page numbers against the evidence.
- If the answer contains no explicit citations, score citation
  accuracy according to the rubric and explain the omission.
- The evidence itself may be incomplete; do not penalize an answer
  for clearly acknowledging that the evidence is insufficient.
- Give concise, specific reasons for each score.
- List concrete unsupported claims, missing information, or
  citation problems in "issues".
- Return ONLY valid JSON. Do not use Markdown fences.

Required JSON structure:
{{
  "faithfulness": {{
    "score": 1,
    "reason": "Explanation"
  }},
  "answer_relevance": {{
    "score": 1,
    "reason": "Explanation"
  }},
  "completeness": {{
    "score": 1,
    "reason": "Explanation"
  }},
  "citation_accuracy": {{
    "score": 1,
    "reason": "Explanation"
  }},
  "issues": [
    "Specific issue, if any"
  ],
  "overall_assessment": "Brief overall evaluation"
}}

USER QUESTION:
{query.strip()}

GENERATED ANSWER:
{answer.strip()}

RETRIEVED EVIDENCE:
{formatted_evidence}
""".strip()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        response_text = getattr(response, "text", None)

        if not response_text or not response_text.strip():
            raise RuntimeError(
                "Gemini returned an empty judge evaluation."
            )

        evaluation = self._parse_json(response_text)
        evaluation = self._validate_result(evaluation)

        scores = [
            evaluation[dimension]["score"]
            for dimension in self.DIMENSIONS
        ]

        evaluation["overall_score"] = round(
            sum(scores) / len(scores),
            2,
        )
        evaluation["model"] = self.model_name

        return evaluation


import os
from typing import Any

from google import genai


class AnswerGenerator:
    """Generate answers grounded in retrieved multimodal evidence."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Set it in your .env file."
            )

        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _format_evidence(
        evidence: list[dict[str, Any]],
    ) -> str:
        """Format evidence into a source-aware prompt."""
        if not evidence:
            return "No relevant evidence was retrieved."

        sections = []

        for index, item in enumerate(evidence, start=1):
            modality = item.get("modality", "unknown")
            source = item.get("source") or "Unknown source"
            page = item.get("page")
            content = item.get("content", "")

            location = f"Source: {source}"
            if page is not None:
                location += f", page: {page}"

            section = (
                f"[Evidence {index}]\n"
                f"Modality: {modality}\n"
                f"{location}\n"
                f"Content: {content}"
            )

            if item.get("image_path"):
                section += (
                    f"\nImage path: {item['image_path']}"
                )

            sections.append(section)

        return "\n\n".join(sections)

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a grounded answer with source references."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if not evidence:
            return {
                "answer": (
                    "I couldn't find relevant evidence in "
                    "the supplied document to answer this question."
                ),
                "sources": [],
                "model": self.model_name,
            }

        formatted_evidence = self._format_evidence(evidence)

        
        prompt = f"""
You are a research-paper assistant.

Answer the user's question using only the retrieved evidence below.

Instructions:
1. Answer every part of the question. If it asks for multiple
   facts, provide each fact explicitly.
2. For model comparisons, identify the winning model and report
   every requested metric, including exact numerical values.
3. Prefer structured table evidence for exact numerical values.
   Use text evidence to add context or cross-check results.
4. Do not invent facts or unsupported conclusions.
5. If a requested fact is missing, state that it could not
   be verified from the supplied evidence.
6. Cite each important factual claim using
   [Source: filename, page: N].
7. Distinguish reported results from your interpretation.
8. Keep the answer concise and directly address the question.

Question:
{query.strip()}

Retrieved evidence:
{formatted_evidence}

Answer every part of the question:
""".strip()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if not answer or not answer.strip():
            raise RuntimeError(
                "Gemini returned an empty answer."
            )

        sources = []
        seen = set()

        for item in evidence:
            source = item.get("source")
            page = item.get("page")
            key = (source, page)

            if key not in seen and (source or page is not None):
                seen.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "model": self.model_name,
        }

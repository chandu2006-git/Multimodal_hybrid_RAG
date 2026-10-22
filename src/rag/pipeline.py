
from pathlib import Path
from typing import Any

from ingestion.document_ingestor import ingest_document
from ingestion.text_chunker import chunk_pages
from rag.answer_generator import AnswerGenerator
from rag.evidence_builder import build_evidence
from retriever.hybrid_retriever import HybridRetriever
from retriever.image_retriever import ImageRetriever
from retriever.table_retriever import TableRetriever


class MultimodalRAGPipeline:
    """End-to-end RAG pipeline for a scientific PDF."""

    def __init__(
        self,
        pdf_path: str | Path,
        image_index_dir: str | Path = "data/processed/image_index",
        image_root: str | Path = "data/images",
        table_root: str | Path = "data/processed/tables",
        text_top_k: int = 3,
        image_top_k: int = 3,
        table_top_k: int = 3,
        answer_generator: Any | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.table_top_k = table_top_k

        # 1. Ingest the PDF.
        self.document = ingest_document(
            pdf_path=self.pdf_path,
            image_root=image_root,
            table_root=table_root,
        )

        # 2. Split extracted page text into chunks.
        chunks = chunk_pages(self.document["text_pages"])

        if not chunks:
            raise ValueError(
                "No text chunks were extracted from the PDF."
            )

        # Ensure the source filename is available for citations.
        for chunk in chunks:
            chunk["metadata"].setdefault(
                "source", self.document["source"]
            )

        # 3. Initialize text retrieval.
        self.text_retriever = HybridRetriever(chunks)

        # 4. Load the persisted image index.
        self.image_retriever = ImageRetriever(
            image_dir=Path(image_root) / self.pdf_path.stem,
            build_index=False,
        )
        self.image_retriever.load_index(image_index_dir)

        # 5. Initialize table retrieval.
        if self.document["tables"]:
            self.table_retriever = TableRetriever(
                self.document["tables"]
            )
        else:
            self.table_retriever = None

        # 6. Set up answer generation.
        self.answer_generator = (
            answer_generator or AnswerGenerator()
        )

    def retrieve(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Retrieve evidence from all available modalities."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        text_results = self.text_retriever.retrieve(
            query,
            top_k=self.text_top_k,
        )

        image_results = self.image_retriever.retrieve(
            query,
            top_k=self.image_top_k,
        )

        table_results = (
            self.table_retriever.retrieve(
                query,
                top_k=self.table_top_k,
            )
            if self.table_retriever is not None
            else []
        )

        evidence = build_evidence(
            text_results=text_results,
            image_results=image_results,
            table_results=table_results,
            source=self.document["source"],
        )

        return {
            "query": query,
            "evidence": evidence,
            "text_results": text_results,
            "image_results": image_results,
            "table_results": table_results,
        }

    def ask(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Retrieve evidence and generate a grounded answer."""
        retrieval = self.retrieve(query)

        answer = self.answer_generator.generate(
            query=query,
            evidence=retrieval["evidence"],
        )

        return {
            **retrieval,
            "answer": answer["answer"],
            "sources": answer["sources"],
            "model": answer["model"],
        }

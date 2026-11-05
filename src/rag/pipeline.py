
from pathlib import Path
from typing import Any

from ingestion.document_ingestor import ingest_document
from chunking.text_chunker import chunk_pages
from generation.answer_generator import AnswerGenerator
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
        text_candidate_k: int | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        if text_top_k < 1:
            raise ValueError("text_top_k must be positive.")

        if image_top_k < 1 or table_top_k < 1:
            raise ValueError(
                "image_top_k and table_top_k must be positive."
            )

        if text_candidate_k is not None and text_candidate_k < 1:
            raise ValueError("text_candidate_k must be positive.")

        self.text_top_k = text_top_k
        self.text_candidate_k = max(
            text_top_k,
            text_candidate_k or text_top_k,
        )
        self.reranker = reranker
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
                "source",
                self.document["source"],
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
        """Retrieve multimodal evidence and optionally rerank text."""

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        # 1. Retrieve text candidates.
        candidate_k = (
            self.text_candidate_k
            if self.reranker is not None
            else self.text_top_k
        )

        text_candidates = self.text_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        # 2. Optionally rerank candidates and select final Top-K.
        if self.reranker is not None:
            text_results = self.reranker.rerank(
                query=query,
                chunks=text_candidates,
                top_k=self.text_top_k,
            )
        else:
            text_results = text_candidates

        # 3. Retrieve images.
        image_results = self.image_retriever.retrieve(
            query,
            top_k=self.image_top_k,
        )

        # 4. Retrieve tables.
        table_results = (
            self.table_retriever.retrieve(
                query,
                top_k=self.table_top_k,
            )
            if self.table_retriever is not None
            else []
        )

        # 5. Build evidence using the final selected text results.
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
            "text_candidates": text_candidates,
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

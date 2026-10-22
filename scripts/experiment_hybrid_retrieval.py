
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.text_chunker import chunk_pages
from retrieval.hybrid_retriever import HybridRetriever


ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "raw_pdfs" / "AMSL_IEEE_Research_paper.pdf"


def main() -> None:
    # 1. Load and chunk the paper.
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(
        pages,
        chunk_size=300,
        chunk_overlap=50,
    )

    print("Pages extracted:", len(pages))
    print("Chunks created:", len(chunks))

    # 2. Build the dense + sparse retriever.
    retriever = HybridRetriever(chunks)

    # 3. Retrieve evidence for a question.
    query = "Which model achieved the highest accuracy in face morph detection?"
    results = retriever.retrieve(
        query,
        top_k=3,
        candidate_k=10,
    )

    print("\nQuestion:", query)
    print("\nTop 3 hybrid retrieval results:\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"--- Rank {rank} | "
            f"RRF: {result['rrf_score']:.5f} | "
            f"Dense: {result['dense_score']:.4f} | "
            f"BM25: {result['bm25_score']:.4f} ---"
        )
        print("Metadata:", result["metadata"])
        print(result["text"][:1000])
        print()


if __name__ == "__main__":
    main()

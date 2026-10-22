
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ingestion.pdf_loader import load_pdf
from ingestion.text_chunker import chunk_pages


ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "raw_pdfs" / "AMSL_IEEE_Research_paper.pdf"


def main() -> None:
    # 1. Extract text from the research paper.
    pages = load_pdf(PDF_PATH)
    print("Pages extracted:", len(pages))

    # 2. Split the pages into overlapping chunks.
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=100,
    )
    print("Chunks created:", len(chunks))

    if not chunks:
        print("No text chunks were created.")
        return

    # 3. Generate an embedding for every chunk.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Embedding matrix shape:", embeddings.shape)

    # 4. Embed a question.
    question = "Which model achieved the highest accuracy in face morph detection?"

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    # 5. Rank chunks by cosine similarity.
    scores = embeddings @ question_embedding

    ranked_indices = scores.argsort()[::-1][:3]

    print("\nQuestion:", question)
    print("\nTop 3 retrieved chunks:\n")

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[int(index)]

        print(f"--- Rank {rank} | Score: {scores[index]:.4f} ---")
        print("Metadata:", chunk["metadata"])
        print(chunk["text"][:1000])
        print()


if __name__ == "__main__":
    main()

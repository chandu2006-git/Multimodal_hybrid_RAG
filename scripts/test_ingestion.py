from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_chunker import chunk_pages

pages = load_pdf("data/raw_pdfs/AMSL_IEEE_Research_paper.pdf")
chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

print("Total chunks:", len(chunks))
print("First chunk:\n", chunks[0]["text"][:300])
print("Metadata:", chunks[0]["metadata"])

from pathlib import Path

from ingestion.document_ingestor import ingest_document


pdf_path = Path("data/raw_pdfs/AMSL_IEEE_Research_paper.pdf")

result = ingest_document(pdf_path)

print(f"\nSource: {result['source']}")
print(f"Text pages extracted: {len(result['text_pages'])}")
print(f"Tables extracted: {len(result['tables'])}")
print(f"Visual assets extracted: {len(result['images'])}")
print(f"Table JSON: {result['table_output']}")

print("\n--- TABLE PREVIEW ---")
for table in result["tables"][:3]:
    print(f"\n{table['table_id']} | Page {table['metadata']['page']}")
    for row in table["rows"][:5]:
        print(row)

print("\n--- IMAGE / GRAPH ASSET PREVIEW ---")
for asset in result["images"][:10]:
    print(
        asset["asset_id"],
        "|",
        asset["metadata"]["content_type"],
        "|",
        asset["path"],
    )

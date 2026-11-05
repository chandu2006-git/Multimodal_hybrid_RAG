from pathlib import Path
import faulthandler

faulthandler.enable()

from evaluation.run_evaluation import load_dataset, build_pipeline

pdf = Path(
    "data/raw_pdfs/Time-Series_Forecasting_in_Industrial_Environments_A_Performance_Study_and_a_Novel_Late_Fusion_Framework.pdf"
)

print("STEP 1: Loading dataset", flush=True)
questions = load_dataset()
question = questions[0]

print("STEP 2: Building pipeline", flush=True)
pipeline = build_pipeline(pdf, "quantitative")

print("STEP 3: Retrieving Q001", flush=True)
result = pipeline.retrieve(question["question"])

print("STEP 4: Retrieval completed", flush=True)
print("Text results:", len(result.get("text_results", [])), flush=True)
print("Image results:", len(result.get("image_results", [])), flush=True)
print("Table results:", len(result.get("table_results", [])), flush=True)

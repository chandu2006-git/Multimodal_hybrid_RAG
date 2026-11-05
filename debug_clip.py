import faulthandler
faulthandler.enable()

import torch
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"

print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

print("1. Loading CLIP processor", flush=True)
processor = CLIPProcessor.from_pretrained(model_name)

print("2. Loading CLIP model on CPU", flush=True)
model = CLIPModel.from_pretrained(model_name).to("cpu")
model.eval()

print("3. Preparing query", flush=True)
inputs = processor(
    text=["What is the full title of the paper?"],
    return_tensors="pt",
    padding=True,
    truncation=True,
)

print("4. Running CLIP text encoder", flush=True)
with torch.inference_mode():
    features = model.get_text_features(**inputs)

print("5. CLIP succeeded:", tuple(features.shape), flush=True)

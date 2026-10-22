
from pathlib import Path
from typing import Any
import hashlib
import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


class ImageRetriever:
    """Retrieve images using CLIP text-image similarity."""

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        image_dir: str | Path,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        build_index: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.image_dir.is_dir():
            raise NotADirectoryError(
                f"Expected an image directory: {self.image_dir}"
            )

        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.records: list[dict[str, Any]] = []
        self.image_embeddings: torch.Tensor | None = None

        if build_index:
            self._build_index()

    @staticmethod
    def _page_number(path: Path) -> int | None:
        """Extract page numbers from filenames such as page_3_image_1.png."""
        match = re.search(r"page_(\d+)", path.stem)
        return int(match.group(1)) if match else None

    def _build_index(self) -> None:
        """Encode supported images and retain their provenance."""
        paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        embeddings: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []

        for path in paths:
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                print(f"Skipping unreadable image {path}: {exc}")
                continue

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)

                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output

                features = F.normalize(features.float(), dim=-1)

            embeddings.append(features.cpu().squeeze(0))
            records.append(
                {
                    "image_path": str(path.resolve()),
                    "filename": path.name,
                    "page": self._page_number(path),
                    "source": path.parent.name,
                }
            )

        if not embeddings:
            raise ValueError(
                f"No readable images found in {self.image_dir}"
            )

        self.image_embeddings = torch.stack(embeddings)
        self.records = records

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return images ranked by text-image cosine similarity."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if self.image_embeddings is None:
            raise RuntimeError(
                "Image index is not loaded. Call load_index() first "
                "or initialize with build_index=True."
            )

        inputs = self.processor(
            text=[query.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)

            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

            query_embedding = F.normalize(
                features.float(), dim=-1
            ).cpu()

        scores = (
            query_embedding @ self.image_embeddings.T
        ).squeeze(0)

        k = min(top_k, len(self.records))
        indices = torch.argsort(scores, descending=True)[:k].tolist()

        return [
            {
                **self.records[index],
                "similarity": float(scores[index]),
            }
            for index in indices
        ]

    def save_index(self, index_dir: str | Path) -> None:
        """Save image embeddings, metadata, and a validation manifest."""
        if self.image_embeddings is None:
            raise ValueError("No image embeddings available to save.")

        if len(self.records) != self.image_embeddings.shape[0]:
            raise ValueError(
                "Embedding and metadata counts do not match."
            )

        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        embeddings = (
            self.image_embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        embeddings_path = index_path / "embeddings.npy"
        metadata_path = index_path / "metadata.json"
        manifest_path = index_path / "manifest.json"

        np.save(embeddings_path, embeddings)

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=2)

        checksum = hashlib.sha256(
            embeddings_path.read_bytes()
        ).hexdigest()

        manifest = {
            "model_name": self.model_name,
            "embedding_count": len(self.records),
            "embedding_dimension": int(embeddings.shape[1]),
            "embeddings_sha256": checksum,
        }

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        print(
            f"Saved {len(self.records)} image embeddings to {index_path}"
        )

    def load_index(self, index_dir: str | Path) -> None:
        """Load and validate a previously saved image index."""
        index_path = Path(index_dir)

        embeddings_path = index_path / "embeddings.npy"
        metadata_path = index_path / "metadata.json"
        manifest_path = index_path / "manifest.json"

        for path in (
            embeddings_path,
            metadata_path,
            manifest_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required index file is missing: {path}"
                )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if manifest.get("model_name") != self.model_name:
            raise ValueError(
                "Index model does not match the configured model. "
                f"Index: {manifest.get('model_name')!r}; "
                f"configured: {self.model_name!r}"
            )

        checksum = hashlib.sha256(
            embeddings_path.read_bytes()
        ).hexdigest()

        if checksum != manifest.get("embeddings_sha256"):
            raise ValueError(
                "Embedding checksum mismatch; index may be corrupted."
            )

        embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )
        records = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if not isinstance(records, list):
            raise ValueError("Metadata must contain a JSON list.")

        if len(records) != embeddings.shape[0]:
            raise ValueError(
                "Embedding and metadata counts do not match."
            )

        if len(records) != manifest.get("embedding_count"):
            raise ValueError(
                "Manifest embedding count does not match."
            )

        if embeddings.shape[1] != manifest.get(
            "embedding_dimension"
        ):
            raise ValueError(
                "Manifest embedding dimension does not match."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values.")

        self.image_embeddings = torch.from_numpy(
            embeddings.astype(np.float32, copy=False)
        )
        self.records = records

        print(
            f"Loaded {len(self.records)} image embeddings from {index_path}"
        )

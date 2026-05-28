import numpy as np
from pathlib import Path
from PIL import Image
from config import SIGLIP_MODEL, CACHE_DIR


class VisualEmbedder:
    def __init__(self, model_name=SIGLIP_MODEL):
        self._processor = None
        self._model = None
        self.model_name = model_name

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, AutoModel
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        import torch
        image_path = Path(image_path)
        cache_file = CACHE_DIR / f"emb_{image_path.stem}.npy"

        if cache_file.exists():
            return np.load(str(cache_file))

        self._load()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")

        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            emb = feats.squeeze().numpy()

        emb = emb / (np.linalg.norm(emb) + 1e-8)
        np.save(str(cache_file), emb)
        return emb

    def embed_batch(self, image_paths: list, progress_cb=None) -> np.ndarray:
        embs = []
        for i, p in enumerate(image_paths):
            embs.append(self.embed_image(p))
            if progress_cb:
                progress_cb(i + 1, len(image_paths))
        return np.array(embs)

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            emb = feats.squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

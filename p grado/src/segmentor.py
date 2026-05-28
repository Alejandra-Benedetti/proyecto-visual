import cv2
import numpy as np
import pickle
from pathlib import Path
from config import SAM_MODEL, CACHE_DIR


class ImageSegmentor:
    def __init__(self, model_path=SAM_MODEL):
        self._model = None
        self.model_path = model_path

    @property
    def model(self):
        if self._model is None:
            from ultralytics import SAM
            self._model = SAM(self.model_path)
        return self._model

    def segment(self, image_path: str | Path, bboxes: list | None = None) -> dict:
        image_path = Path(image_path)
        suffix = "_bbox" if bboxes else "_auto"
        cache_file = CACHE_DIR / f"seg_{image_path.stem}{suffix}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        if bboxes:
            results = self.model(str(image_path), bboxes=bboxes, verbose=False)
        else:
            results = self.model(str(image_path), verbose=False)

        masks_data = []
        for r in results:
            if r.masks is not None:
                for mask in r.masks.data:
                    m = mask.cpu().numpy().astype(bool)
                    masks_data.append({"mask": m, "area": int(m.sum())})

        result = {
            "image": str(image_path),
            "num_masks": len(masks_data),
            "masks": masks_data,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    @staticmethod
    def overlay_masks(image: np.ndarray, masks_data: list) -> np.ndarray:
        overlay = image.copy()
        palette = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (255, 80, 255), (80, 255, 255),
        ]
        h, w = image.shape[:2]
        for i, md in enumerate(masks_data):
            mask = md["mask"]
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            color = np.array(palette[i % len(palette)], dtype=np.uint8)
            overlay[mask] = (overlay[mask] * 0.45 + color * 0.55).astype(np.uint8)
        return overlay

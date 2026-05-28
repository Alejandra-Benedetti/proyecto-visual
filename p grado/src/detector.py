import cv2
import numpy as np
import json
from pathlib import Path
from config import YOLO_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, CACHE_DIR


class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
        self._model = None
        self.model_path = model_path
        self.conf = conf
        self.iou = iou

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        cache_file = CACHE_DIR / f"det_{image_path.stem}.json"

        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        results = self.model(
            str(image_path),
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                    "bbox_norm": box.xywhn[0].tolist(),
                })

        result = {
            "image": str(image_path),
            "num_detections": len(detections),
            "detections": detections,
        }
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def detect_batch(self, image_paths: list, progress_cb=None) -> list:
        results = []
        for i, p in enumerate(image_paths):
            results.append(self.detect(p))
            if progress_cb:
                progress_cb(i + 1, len(image_paths))
        return results

    def draw_detections(self, image: np.ndarray, det: dict) -> np.ndarray:
        img = image.copy()
        for d in det["detections"]:
            x1, y1, x2, y2 = map(int, d["bbox"])
            label = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 60), 2)
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 60), 1)
        return img

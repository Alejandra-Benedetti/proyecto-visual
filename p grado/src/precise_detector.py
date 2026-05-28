"""
Módulo de detección precisa para imágenes de góndola/estantería.

Estrategias implementadas:
  1. SAHI  — divide la imagen en tiles superpuestos antes de inferencia,
             ideal para objetos pequeños/densos. Usa NMM (Non-Maximum Merging)
             en vez de NMS para no eliminar cajas válidas.
  2. Watershed — separa objetos adyacentes dentro de una máscara SAM2
             que agrupa múltiples productos como uno solo.
  3. Filtros geométricos — elimina máscaras que son fondo (muy grandes),
             ruido (muy pequeñas), formas extremas o poco sólidas.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from scipy import ndimage as ndi
from config import CACHE_DIR


# ── SAHI detector ─────────────────────────────────────────────────────────────

class SahiDetector:
    """YOLOv11 con SAHI tiling para detección densa de alta precisión."""

    def __init__(self, model_path="yolo11s.pt", conf=0.10,
                 slice_size=512, overlap=0.2):
        self._model = None
        self.model_path = model_path
        self.conf = conf
        self.slice_size = slice_size
        self.overlap = overlap

    @property
    def model(self):
        if self._model is None:
            from sahi import AutoDetectionModel
            self._model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=self.conf,
                device="cpu",
            )
        return self._model

    def detect(self, image_path: str | Path) -> dict:
        from sahi.predict import get_sliced_prediction
        image_path = Path(image_path)
        cache_file = CACHE_DIR / f"sahi_{image_path.stem}.json"

        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        result = get_sliced_prediction(
            str(image_path),
            self.model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="NMM",          # Non-Maximum Merging: preserva cajas válidas
            postprocess_match_metric="IOS",  # Intersection over Smaller area
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            box = pred.bbox.to_xyxy()
            detections.append({
                "class_id": pred.category.id or 0,
                "class_name": pred.category.name or "objeto",
                "confidence": float(pred.score.value),
                "bbox": [float(v) for v in box],
                "bbox_norm": [],
            })

        output = {
            "image": str(image_path),
            "num_detections": len(detections),
            "detections": detections,
            "mode": "SAHI",
        }
        cache_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output

    def detect_batch(self, image_paths: list, progress_cb=None) -> list:
        results = []
        for i, p in enumerate(image_paths):
            results.append(self.detect(p))
            if progress_cb:
                progress_cb(i + 1, len(image_paths))
        return results

    def draw(self, image: np.ndarray, det: dict) -> np.ndarray:
        img = image.copy()
        for d in det["detections"]:
            x1, y1, x2, y2 = map(int, d["bbox"])
            label = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
        return img


# ── Watershed split ───────────────────────────────────────────────────────────

def watershed_split(mask: np.ndarray, min_distance: int = 22) -> list[np.ndarray]:
    """
    Intenta separar una máscara binaria en sub-máscaras usando watershed.
    Si la máscara contiene un solo objeto, devuelve [mask] sin cambios.
    """
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        return [mask]

    m = mask.astype(np.uint8)
    dist = ndi.distance_transform_edt(m)

    coords = peak_local_max(dist, min_distance=min_distance, labels=m)
    if len(coords) <= 1:
        return [mask]

    markers = np.zeros_like(m, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1

    labels = watershed(-dist, markers, mask=m)

    subs = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        sub = (labels == lbl)
        if sub.sum() > 80:      # mínimo 80 px para no contar ruido
            subs.append(sub)

    return subs if subs else [mask]


# ── Geometric filters ─────────────────────────────────────────────────────────

def filter_sam_masks(
    masks_data: list,
    image_shape: tuple,
    min_area_ratio: float = 0.0008,   # mínimo 0.08% del área total
    max_area_ratio: float = 0.18,     # máximo 18% (evita detectar el fondo)
    min_aspect: float = 0.10,         # no demasiado achatado
    max_aspect: float = 10.0,
    min_solidity: float = 0.30,       # solidez mínima (no formas muy irregulares)
    watershed_threshold: float = 0.04, # máscaras >4% se intentan separar
) -> list:
    """
    Filtra y post-procesa las máscaras SAM2 para conteo preciso de productos.

    Pasos por máscara:
      1. Descartar si área fuera del rango [min_area_ratio, max_area_ratio]
      2. Descartar si aspecto (w/h) fuera de rango
      3. Descartar si solidez < min_solidity (eliminates background artifacts)
      4. Si la máscara es grande (> watershed_threshold), aplicar watershed
         para separar productos que SAM2 agrupó juntos
    """
    h, w = image_shape[:2]
    total_px = h * w
    result = []

    for md in masks_data:
        mask = md["mask"]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        area = int(mask.sum())
        ratio = area / total_px

        # ── Filtro 1: área ────────────────────────────────────────────────────
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue

        # ── Filtro 2: bounding box aspect ratio ───────────────────────────────
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        bh = int(ys.max() - ys.min()) + 1
        bw = int(xs.max() - xs.min()) + 1
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # ── Filtro 3: solidez (convex hull) ───────────────────────────────────
        m_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            if hull_area > 0 and (area / hull_area) < min_solidity:
                continue

        # ── Filtro 4: watershed en máscaras grandes ────────────────────────────
        if ratio > watershed_threshold:
            subs = watershed_split(mask, min_distance=int(min(bh, bw) * 0.25))
            for sub in subs:
                sub_area = int(sub.sum())
                sub_ratio = sub_area / total_px
                if min_area_ratio <= sub_ratio <= max_area_ratio:
                    result.append({"mask": sub, "area": sub_area})
        else:
            result.append({"mask": mask, "area": area})

    return result

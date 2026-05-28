import cv2
import numpy as np
import json
from pathlib import Path
from scipy import stats as scipy_stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from PIL import Image
import pandas as pd
from config import HISTOGRAM_BINS, CACHE_DIR


class StatisticalExtractor:

    def extract(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        cache_file = CACHE_DIR / f"stats_{image_path.stem}.json"

        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        img_bgr = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        h, w = img_gray.shape
        features: dict = {"image_name": image_path.name, "width": w, "height": h}
        features.update(self._color_stats(img_rgb))
        features.update(self._intensity_stats(img_gray))
        features.update(self._hsv_stats(img_hsv))
        features.update(self._texture_features(img_gray))
        features.update(self._edge_features(img_gray))
        features.update(self._spatial_features(img_rgb))

        cache_file.write_text(json.dumps(features), encoding="utf-8")
        return features

    def _color_stats(self, img_rgb: np.ndarray) -> dict:
        feat = {}
        for i, ch in enumerate(["r", "g", "b"]):
            c = img_rgb[:, :, i].flatten().astype(float)
            feat[f"mean_{ch}"] = float(np.mean(c))
            feat[f"std_{ch}"] = float(np.std(c))
            feat[f"skew_{ch}"] = float(scipy_stats.skew(c))
            feat[f"kurt_{ch}"] = float(scipy_stats.kurtosis(c))

        total = feat["mean_r"] + feat["mean_g"] + feat["mean_b"] + 1e-6
        feat["ratio_r"] = feat["mean_r"] / total
        feat["ratio_g"] = feat["mean_g"] / total
        feat["ratio_b"] = feat["mean_b"] / total
        return feat

    def _intensity_stats(self, img_gray: np.ndarray) -> dict:
        flat = img_gray.flatten().astype(float)
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        return {
            "mean_intensity": mean,
            "std_intensity": std,
            "median_intensity": float(np.median(flat)),
            "min_intensity": float(np.min(flat)),
            "max_intensity": float(np.max(flat)),
            "range_intensity": float(np.max(flat) - np.min(flat)),
            "skew_intensity": float(scipy_stats.skew(flat)),
            "kurt_intensity": float(scipy_stats.kurtosis(flat)),
            "entropy": float(shannon_entropy(img_gray)),
            "p25_intensity": float(np.percentile(flat, 25)),
            "p75_intensity": float(np.percentile(flat, 75)),
            "iqr_intensity": float(np.percentile(flat, 75) - np.percentile(flat, 25)),
            "cv_intensity": std / (mean + 1e-6),
        }

    def _hsv_stats(self, img_hsv: np.ndarray) -> dict:
        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        return {
            "mean_hue": float(np.mean(h)),
            "std_hue": float(np.std(h)),
            "mean_saturation": float(np.mean(s)),
            "std_saturation": float(np.std(s)),
            "mean_value": float(np.mean(v)),
            "std_value": float(np.std(v)),
            "colorfulness": float(np.mean(s) * np.mean(v) / 255.0),
        }

    def _texture_features(self, img_gray: np.ndarray) -> dict:
        h, w = img_gray.shape
        small = cv2.resize(img_gray, (w // 4, h // 4))
        quantized = (small // 4).astype(np.uint8)
        glcm = graycomatrix(
            quantized, distances=[1], angles=[0, np.pi / 2],
            levels=64, symmetric=True, normed=True,
        )
        return {
            "texture_contrast": float(graycoprops(glcm, "contrast").mean()),
            "texture_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
            "texture_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            "texture_energy": float(graycoprops(glcm, "energy").mean()),
            "texture_correlation": float(graycoprops(glcm, "correlation").mean()),
            "texture_asm": float(graycoprops(glcm, "ASM").mean()),
        }

    def _edge_features(self, img_gray: np.ndarray) -> dict:
        sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        edges = cv2.Canny(img_gray, 50, 150)
        return {
            "edge_density": float(np.mean(edges > 0)),
            "mean_gradient": float(np.mean(mag)),
            "std_gradient": float(np.std(mag)),
            "max_gradient": float(np.max(mag)),
        }

    def _spatial_features(self, img_rgb: np.ndarray) -> dict:
        h, w = img_rgb.shape[:2]
        grid_means = []
        for row in range(3):
            for col in range(3):
                r1, r2 = row * h // 3, (row + 1) * h // 3
                c1, c2 = col * w // 3, (col + 1) * w // 3
                region = img_rgb[r1:r2, c1:c2]
                grid_means.append(float(np.mean(region)))

        feat = {f"region_{i}_mean": v for i, v in enumerate(grid_means)}
        feat["spatial_variance"] = float(np.var(grid_means))
        feat["aspect_ratio"] = float(w / h)
        return feat

    def extract_batch(self, image_paths: list, progress_cb=None) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(image_paths):
            rows.append(self.extract(p))
            if progress_cb:
                progress_cb(i + 1, len(image_paths))
        return pd.DataFrame(rows)

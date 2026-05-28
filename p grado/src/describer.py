import base64
import json
import requests
from pathlib import Path
from config import QWEN_MODEL, OLLAMA_URL, CACHE_DIR


class ImageDescriber:
    def __init__(self, model: str = QWEN_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                r = requests.get(f"{self.url}/api/tags", timeout=2)
                models = [m["name"] for m in r.json().get("models", [])]
                self._available = any(self.model.split(":")[0] in m for m in models)
            except Exception:
                self._available = False
        return self._available

    def describe(self, image_path: str | Path, stats: dict | None = None) -> str:
        image_path = Path(image_path)
        cache_file = CACHE_DIR / f"desc_{image_path.stem}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        desc = (
            self._ollama_describe(image_path, stats)
            if self.is_available
            else self._template_describe(image_path, stats)
        )
        cache_file.write_text(desc, encoding="utf-8")
        return desc

    def _ollama_describe(self, image_path: Path, stats: dict | None) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        stats_str = ""
        if stats:
            stats_str = (
                f"\n\nEstadísticas cuantitativas:\n"
                f"- Intensidad media: {stats.get('mean_intensity', 0):.1f}\n"
                f"- Entropía: {stats.get('entropy', 0):.3f}\n"
                f"- Densidad de bordes: {stats.get('edge_density', 0):.3f}\n"
                f"- Colorfulness: {stats.get('colorfulness', 0):.3f}"
            )

        prompt = (
            "Analiza esta imagen de una góndola/estantería de ferretería. "
            "Describe: 1) productos visibles y marcas principales, "
            "2) organización y densidad del estante, "
            "3) estado visual y limpieza, "
            "4) características de color dominante."
            f"{stats_str}"
        )

        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt,
                      "images": [img_b64], "stream": False},
                timeout=90,
            )
            return r.json().get("response", self._template_describe(image_path, stats))
        except Exception:
            return self._template_describe(image_path, stats)

    @staticmethod
    def _template_describe(image_path: Path, stats: dict | None) -> str:
        if not stats:
            return f"Imagen: {image_path.name}."

        intensity = stats.get("mean_intensity", 128)
        brightness = "alta" if intensity > 170 else "baja" if intensity < 85 else "media"
        entropy = stats.get("entropy", 4)
        complexity = "alta" if entropy > 5 else "baja" if entropy < 3 else "media"
        edge_d = stats.get("edge_density", 0.1)
        detail = "alto" if edge_d > 0.2 else "bajo" if edge_d < 0.05 else "moderado"
        colorfulness = stats.get("colorfulness", 0)
        color_desc = "muy colorida" if colorfulness > 50 else "poco colorida" if colorfulness < 15 else "moderadamente colorida"

        return (
            f"'{image_path.name}': Luminosidad {brightness} (intensidad: {intensity:.1f}). "
            f"Complejidad visual {complexity} (entropía: {entropy:.3f}). "
            f"Nivel de detalle {detail} (bordes: {edge_d:.3f}). "
            f"Imagen {color_desc} (colorfulness: {colorfulness:.1f}). "
            f"Canales RGB — R:{stats.get('mean_r',0):.1f} "
            f"G:{stats.get('mean_g',0):.1f} B:{stats.get('mean_b',0):.1f}."
        )

    def describe_batch(
        self, image_paths: list, stats_list: list | None = None, progress_cb=None
    ) -> list[str]:
        results = []
        for i, p in enumerate(image_paths):
            s = stats_list[i] if stats_list and i < len(stats_list) else None
            results.append(self.describe(p, s))
            if progress_cb:
                progress_cb(i + 1, len(image_paths))
        return results

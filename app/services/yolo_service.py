"""
yolo_service.py
Servicio de detección de objetos con YOLO en imágenes de estanterías.
"""
import os, io, time, base64
from PIL import Image, ImageDraw
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_DISPONIBLE = True
except ImportError:
    YOLO_DISPONIBLE = False
    print("[WARN] ultralytics no instalado. Usando modo simulación.")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
CONFIANZA_MIN   = float(os.getenv("YOLO_CONFIDENCE", "0.35"))

COLORES = {
    "propio":      "#00D4AA",
    "competencia": "#FF6B6B",
    "desconocido": "#FFD93D",
}

_modelo_yolo = None

def get_modelo():
    global _modelo_yolo
    if not YOLO_DISPONIBLE:
        return None
    if _modelo_yolo is None:
        print(f"[YOLO] Cargando modelo: {YOLO_MODEL_PATH}")
        _modelo_yolo = YOLO(YOLO_MODEL_PATH)
    return _modelo_yolo


def imagen_a_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _simular_detecciones(image: Image.Image) -> list:
    w, h = image.size
    np.random.seed(42)
    n = np.random.randint(4, 12)
    detecciones = []
    for i in range(n):
        x1 = int(np.random.uniform(0.05, 0.7) * w)
        y1 = int(np.random.uniform(0.05, 0.7) * h)
        x2 = min(x1 + int(np.random.uniform(0.08, 0.2) * w), w - 5)
        y2 = min(y1 + int(np.random.uniform(0.1, 0.3) * h), h - 5)
        es_comp = np.random.random() < 0.35
        detecciones.append({
            "id":        i,
            "clase":     "producto",
            "confianza": round(float(np.random.uniform(0.45, 0.97)), 3),
            "bbox":      [x1, y1, x2, y2],
            "es_propio": not es_comp,
            "tipo":      "competencia" if es_comp else "propio",
        })
    return detecciones


def dibujar_detecciones(image: Image.Image, detecciones: list) -> Image.Image:
    img_draw = image.copy().convert("RGBA")
    overlay  = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)
    for det in detecciones:
        x1, y1, x2, y2 = det["bbox"]
        tipo      = det.get("tipo", "desconocido")
        color_hex = COLORES.get(tipo, COLORES["desconocido"])
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 45), outline=(r, g, b, 230), width=2)
        label = f"{tipo[:4].upper()} {det['confianza']:.0%}"
        lx, ly = x1 + 3, max(y1 - 18, 0)
        draw.rectangle([lx - 2, ly - 1, lx + len(label) * 7, ly + 14], fill=(r, g, b, 200))
        draw.text((lx, ly), label, fill=(255, 255, 255, 255))
    resultado = Image.alpha_composite(img_draw, overlay)
    return resultado.convert("RGB")


def procesar_imagen_yolo(image: Image.Image) -> dict:
    tic    = time.perf_counter()
    modelo = get_modelo()

    if modelo is None or not YOLO_DISPONIBLE:
        detecciones = _simular_detecciones(image)
        modo = "simulacion"
    else:
        results = modelo.predict(source=image, conf=CONFIANZA_MIN, verbose=False)
        detecciones = []
        for r in results:
            for i, box in enumerate(r.boxes):
                clase = modelo.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                es_propio = (i % 2 == 0)
                detecciones.append({
                    "id":        i,
                    "clase":     clase,
                    "confianza": round(conf, 3),
                    "bbox":      [x1, y1, x2, y2],
                    "es_propio": es_propio,
                    "tipo":      "propio" if es_propio else "competencia",
                })
        modo = "yolo"

    imagen_anotada = dibujar_detecciones(image, detecciones)
    tiempo = round(time.perf_counter() - tic, 3)

    total       = len(detecciones)
    propios     = sum(1 for d in detecciones if d["tipo"] == "propio")
    competencia = sum(1 for d in detecciones if d["tipo"] == "competencia")
    conf_prom   = round(np.mean([d["confianza"] for d in detecciones]), 3) if detecciones else 0
    alta   = sum(1 for d in detecciones if d["confianza"] >= 0.75)
    media  = sum(1 for d in detecciones if 0.5 <= d["confianza"] < 0.75)
    baja   = sum(1 for d in detecciones if d["confianza"] < 0.5)

    return {
        "modo":            modo,
        "tiempo_s":        tiempo,
        "total_productos": total,
        "propios":         propios,
        "competencia":     competencia,
        "confianza_prom":  conf_prom,
        "distribucion_confianza": {"alta": alta, "media": media, "baja": baja},
        "detecciones":     detecciones,
        "imagen_anotada_b64": imagen_a_base64(imagen_anotada),
    }
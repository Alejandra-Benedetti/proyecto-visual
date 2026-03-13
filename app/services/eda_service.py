"""
eda_service.py
Análisis Exploratorio de Datos para imágenes de estanterías.
"""
import os, json
from collections import Counter
from PIL import Image
import numpy as np


def analizar_imagen_eda(image: Image.Image, nombre_archivo: str = "imagen") -> dict:
    w, h     = image.size
    modo     = image.mode
    img_array = np.array(image.convert("RGB"))
    brillo_promedio = float(np.mean(img_array))
    canal_r = float(np.mean(img_array[:, :, 0]))
    canal_g = float(np.mean(img_array[:, :, 1]))
    canal_b = float(np.mean(img_array[:, :, 2]))

    if brillo_promedio < 60:
        calidad_brillo = "MUY_OSCURA"
    elif brillo_promedio < 100:
        calidad_brillo = "OSCURA"
    elif brillo_promedio > 220:
        calidad_brillo = "SOBREEXPUESTA"
    elif brillo_promedio > 180:
        calidad_brillo = "CLARA"
    else:
        calidad_brillo = "NORMAL"

    aspect_ratio = round(w / h, 3)
    orientacion  = "LANDSCAPE" if w > h else "PORTRAIT" if h > w else "CUADRADA"
    megapixeles  = round((w * h) / 1_000_000, 2)

    if megapixeles < 0.5:
        resolucion_cat = "BAJA"
    elif megapixeles < 2:
        resolucion_cat = "MEDIA"
    elif megapixeles < 8:
        resolucion_cat = "ALTA"
    else:
        resolucion_cat = "MUY_ALTA"

    return {
        "archivo":         nombre_archivo,
        "dimensiones":     {"ancho": w, "alto": h},
        "megapixeles":     megapixeles,
        "resolucion_cat":  resolucion_cat,
        "modo_color":      modo,
        "orientacion":     orientacion,
        "aspect_ratio":    aspect_ratio,
        "brillo_promedio": round(brillo_promedio, 2),
        "calidad_brillo":  calidad_brillo,
        "canales_rgb":     {
            "rojo":  round(canal_r, 2),
            "verde": round(canal_g, 2),
            "azul":  round(canal_b, 2)
        },
    }


def calcular_eda_detecciones(yolo_resultado: dict, ollama_resultado: dict) -> dict:
    detecciones    = yolo_resultado.get("detecciones", [])
    precios_ollama = ollama_resultado.get("precios", [])

    total   = yolo_resultado.get("total_productos", 0)
    propios = yolo_resultado.get("propios", 0)
    comp    = yolo_resultado.get("competencia", 0)
    share_of_shelf = round((propios / total * 100), 1) if total > 0 else 0

    precios_validos = [
        p for p in precios_ollama
        if p.get("precio") and p["precio"] not in ("ilegible", "NOT VISIBLE", "")
    ]
    n_legibles  = len(precios_validos)
    n_ilegibles = len(precios_ollama) - n_legibles
    conf_precios = Counter(p.get("confianza", "BAJA") for p in precios_ollama)
    dist_conf    = yolo_resultado.get("distribucion_confianza", {})

    return {
        "resumen": {
            "total_productos":    total,
            "propios":            propios,
            "competencia":        comp,
            "share_of_shelf":     share_of_shelf,
            "precios_legibles":   n_legibles,
            "precios_ilegibles":  n_ilegibles,
            "tipo_estanteria":    ollama_resultado.get("tipo_estanteria", "—"),
            "calidad_imagen":     ollama_resultado.get("calidad_imagen", "—"),
            "confianza_prom_yolo": yolo_resultado.get("confianza_prom", 0),
            "tiempo_yolo_s":      yolo_resultado.get("tiempo_s", 0),
            "tiempo_ollama_s":    ollama_resultado.get("_tiempo_inferencia_s", 0),
        },
        "graficas": {
            "productos": {
                "labels":  ["Propios", "Competencia"],
                "valores": [propios, comp],
                "colores": ["#00D4AA", "#FF6B6B"]
            },
            "confianza": {
                "labels":  ["Alta (≥75%)", "Media (50-75%)", "Baja (<50%)"],
                "valores": [dist_conf.get("alta", 0), dist_conf.get("media", 0), dist_conf.get("baja", 0)],
                "colores": ["#6BCB77", "#FFD93D", "#FF6B6B"]
            },
            "precios": {
                "labels":  ["Legibles", "Ilegibles"],
                "valores": [n_legibles, n_ilegibles],
                "colores": ["#4D96FF", "#C9C9C9"]
            },
            "conf_precios": {
                "labels":  list(conf_precios.keys()),
                "valores": list(conf_precios.values()),
                "colores": ["#6BCB77", "#FFD93D", "#FF6B6B"]
            },
        },
        "tabla_precios":        precios_ollama,
        "observaciones_vision": ollama_resultado.get("observaciones", ""),
    }


def guardar_resultado_json(resultado: dict, carpeta: str = "../../data/resultados") -> str:
    import datetime
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(carpeta, f"analisis_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        resultado_limpio = {k: v for k, v in resultado.items() if "b64" not in k}
        json.dump(resultado_limpio, f, ensure_ascii=False, indent=2, default=str)
    return path
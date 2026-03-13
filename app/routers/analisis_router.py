"""
analisis_router.py
Endpoints REST para el dashboard de análisis de estanterías.
"""
import io, base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from services.ollama_service import (
    analizar_imagen_ollama,
    generar_analisis_eda,
    verificar_conexion
)
from services.yolo_service import procesar_imagen_yolo, imagen_a_base64
from services.eda_service import (
    analizar_imagen_eda,
    calcular_eda_detecciones,
    guardar_resultado_json
)

router = APIRouter(prefix="/api", tags=["analisis"])

FORMATOS_PERMITIDOS = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 20


def _cargar_imagen(file: UploadFile) -> Image.Image:
    if file.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(400, f"Formato no soportado: {file.content_type}")
    contenido = file.file.read()
    if len(contenido) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Imagen demasiado grande. Máximo {MAX_SIZE_MB}MB.")
    try:
        img = Image.open(io.BytesIO(contenido))
        img.load()
        return img
    except Exception as e:
        raise HTTPException(400, f"No se pudo abrir la imagen: {e}")


@router.get("/health")
def health():
    return {"status": "ok", "servicio": "proyecto-visual"}


@router.get("/ollama/status")
def ollama_status():
    return verificar_conexion()


@router.post("/analizar")
async def analizar(file: UploadFile = File(...)):
    imagen = _cargar_imagen(file)
    nombre = file.filename or "imagen_subida"
    img_original_b64 = imagen_a_base64(imagen)
    eda_imagen = analizar_imagen_eda(imagen, nombre)
    yolo_resultado = procesar_imagen_yolo(imagen)
    ollama_resultado = analizar_imagen_ollama(imagen)
    datos_para_ia = {
        "total_productos":    yolo_resultado["total_productos"],
        "propios":            yolo_resultado["propios"],
        "competencia":        yolo_resultado["competencia"],
        "tipo_estanteria":    ollama_resultado.get("tipo_estanteria", "otro"),
        "n_precios_legibles": ollama_resultado.get("n_precios_legibles", 0),
        "calidad_imagen":     ollama_resultado.get("calidad_imagen", "REGULAR"),
        "precios":            ollama_resultado.get("precios", []),
        "observaciones":      ollama_resultado.get("observaciones", ""),
    }
    analisis_ia = generar_analisis_eda(datos_para_ia)
    eda_combinado = calcular_eda_detecciones(yolo_resultado, ollama_resultado)
    respuesta = {
        "archivo":             nombre,
        "imagen_original_b64": img_original_b64,
        "imagen_anotada_b64":  yolo_resultado.pop("imagen_anotada_b64", ""),
        "eda_imagen":          eda_imagen,
        "yolo":                {k: v for k, v in yolo_resultado.items() if k != "detecciones"},
        "ollama":              {k: v for k, v in ollama_resultado.items() if not k.startswith("_")},
        "eda":                 eda_combinado,
        "analisis_ia":         analisis_ia,
    }
    try:
        guardar_resultado_json(respuesta)
    except Exception:
        pass
    return JSONResponse(content=respuesta)


@router.post("/analizar/solo-yolo")
async def analizar_yolo(file: UploadFile = File(...)):
    imagen = _cargar_imagen(file)
    resultado = procesar_imagen_yolo(imagen)
    return JSONResponse(content=resultado)


@router.post("/analizar/solo-ollama")
async def analizar_ollama(file: UploadFile = File(...)):
    imagen = _cargar_imagen(file)
    resultado = analizar_imagen_ollama(imagen)
    return JSONResponse(content=resultado)
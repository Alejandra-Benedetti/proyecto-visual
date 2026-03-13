"""
ollama_service.py
Servicio de análisis de imágenes con Ollama + Qwen Vision
"""
import os, io, re, json, base64, time, functools
from PIL import Image
from openai import OpenAI

OLLAMA_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
MODEL_NAME  = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
MAX_TOKENS  = int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

def my_timer(orig_func):
    @functools.wraps(orig_func)
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        value = orig_func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed = round(toc - tic, 3)
        print(f"[TIMER] {orig_func.__name__}: {elapsed}s")
        if isinstance(value, dict):
            value["_tiempo_inferencia_s"] = elapsed
        return value
    return wrapper

PROMPT_ANALISIS = """Tarea: Análisis de Precios en Estanterías de Supermercado.
Analiza la imagen de una estantería. Identifica productos y lee etiquetas de precio.
Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{
  "n_productos_detectados": <int>,
  "n_precios_legibles": <int>,
  "precios": [
    {"producto": "<descripcion>", "precio": "<valor o ilegible>", "confianza": "ALTA|MEDIA|BAJA", "es_competencia": false}
  ],
  "tipo_estanteria": "<bebidas|snacks|lacteos|limpieza|otro>",
  "calidad_imagen": "BUENA|REGULAR|MALA",
  "observaciones": "<texto breve con hallazgos relevantes>"
}"""

PROMPT_EDA = """Eres un analista experto en retail y distribución de productos.
Se te proporcionan los resultados de una detección automática de productos en una estantería.
Genera un análisis interpretativo breve (máximo 150 palabras) que incluya:
1. Conclusión principal sobre la distribución de productos
2. Oportunidades de mejora en el planograma
3. Observación sobre presencia de competencia (si aplica)
Datos de entrada: {datos}
Responde en español, en párrafos cortos, sin listas ni bullets."""


def encode_image(image: Image.Image, max_size: int = 1024) -> str:
    buf = io.BytesIO()
    if image.mode in ("P", "RGBA"):
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def limpiar_json(content: str) -> str:
    c = re.sub(r"```(?:json)?", "", content).strip()
    c = re.sub(r"```", "", c).strip()
    match = re.search(r"\{.*\}", c, re.DOTALL)
    return match.group(0) if match else ""


@my_timer
def analizar_imagen_ollama(image: Image.Image) -> dict:
    try:
        b64 = encode_image(image)
    except Exception as e:
        return {"error": f"Error codificando imagen: {e}", "precios": []}
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_ANALISIS},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            max_tokens=MAX_TOKENS,
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip()
        cleaned = limpiar_json(raw)
        if not cleaned:
            return {"error": "Respuesta vacía del modelo", "precios": [], "raw": raw}
        resultado = json.loads(cleaned)
        resultado.setdefault("precios", [])
        resultado.setdefault("n_productos_detectados", 0)
        resultado.setdefault("n_precios_legibles", 0)
        resultado.setdefault("tipo_estanteria", "otro")
        resultado.setdefault("calidad_imagen", "REGULAR")
        resultado.setdefault("observaciones", "")
        return resultado
    except json.JSONDecodeError as e:
        return {"error": f"JSON inválido: {e}", "precios": []}
    except Exception as e:
        return {"error": str(e)[:200], "precios": []}


@my_timer
def generar_analisis_eda(datos: dict) -> str:
    try:
        datos_str = json.dumps(datos, ensure_ascii=False, indent=2)
        prompt = PROMPT_EDA.format(datos=datos_str)
        resp = client.chat.completions.create(
            model=os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5:3b"),
            messages=[
                {"role": "system", "content": "Eres un analista experto en retail. Responde siempre en español."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[No se pudo generar análisis IA: {e}]"


def verificar_conexion() -> dict:
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5:3b"),
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            max_tokens=10
        )
        return {"status": "ok", "modelo": MODEL_NAME, "respuesta": resp.choices[0].message.content.strip()}
    except Exception as e:
        return {"status": "error", "detalle": str(e)}
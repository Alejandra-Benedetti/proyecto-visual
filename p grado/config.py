from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"

for d in [INPUT_DIR, CACHE_DIR, MODELS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Models
YOLO_MODEL = "yolo11s.pt"       # small: mejor precisión en CPU vs nano
SAM_MODEL = "sam2.1_t.pt"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
QWEN_MODEL = "qwen2.5-vl:7b"
OLLAMA_URL = "http://localhost:11434"

# Detection — umbral bajo para capturar productos en góndola
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.40

# FAISS
EMBEDDING_DIM = 768
TOP_K_SIMILAR = 6

# Stats
HISTOGRAM_BINS = 64

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

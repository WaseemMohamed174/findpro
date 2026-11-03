import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
STATIC_DIR = DATA_DIR / "static"
RUNTIME_DIR = DATA_DIR / "runtime"

MODELS_CATALOG = STATIC_DIR / "catalog" / "models.json"
SEED_FILE = STATIC_DIR / "workers_data.xlsx"

DB_PATH = RUNTIME_DIR / "workers.db"
MODELS_DIR = RUNTIME_DIR / "models"
EMBEDDER_DIR = MODELS_DIR / "embedder"
LLM_DIR = MODELS_DIR / "llm"
VENV_DIR = RUNTIME_DIR / "venv"
LOGS_DIR = RUNTIME_DIR / "logs"
SELECTION_FILE = RUNTIME_DIR / "selection.json"

WORKERS_DIR = BASE_DIR / "models"
EMBEDDER_WORKER = WORKERS_DIR / "embedder_py.py"
LLM_WORKER = WORKERS_DIR / "llm_py.py"

WEB_DIR = BASE_DIR / "web"


def venv_python(name):
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV_DIR / name / sub


def ensure_runtime_dirs():
    for d in (RUNTIME_DIR, MODELS_DIR, EMBEDDER_DIR, LLM_DIR, VENV_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

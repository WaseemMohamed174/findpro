import json
import os
import re
import secrets
from pathlib import Path

from . import paths


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _embedder_entry(sel, cat):
    chosen = sel.get("embedder", {})
    if chosen.get("source") == "local":
        return chosen
    key = chosen.get("key") or cat.get("defaults", {}).get("embedder")
    embedders = cat.get("embedders", [])
    for e in embedders:
        if e["key"] == key:
            return e
    return embedders[0] if embedders else {}


def _llm_entry(sel, cat):
    chosen = sel.get("llm", {})
    if chosen.get("source") == "local":
        return chosen
    if chosen.get("enabled", True) is False:
        return {}
    name = chosen.get("name") or cat.get("defaults", {}).get("llm", "")
    for m in cat.get("llms", []):
        if m["name"] == name:
            return {**m, "filename": chosen.get("filename") or m.get("filename")}
    return {}


def load():
    g = globals()
    sel = _read(paths.SELECTION_FILE, {})
    cat = _read(paths.MODELS_CATALOG, {"embedders": [], "llms": [], "defaults": {}})
    emb = _embedder_entry(sel, cat)
    llm = _llm_entry(sel, cat)

    g["EMBEDDER_KEY"] = emb.get("key", "")
    g["EMBEDDER_REPO"] = emb.get("repo", "")
    g["EMBEDDER_DIM"] = int(emb.get("dim", 384))
    g["EMBEDDER_PASSAGE_PREFIX"] = emb.get("passage_prefix", "")
    g["EMBEDDER_QUERY_PREFIX"] = emb.get("query_prefix", "")
    g["EMBEDDER_FILES"] = emb.get("files", [])
    g["EMBEDDER_IGNORE"] = emb.get("ignore", [])
    g["EMBEDDER_DIR_PATH"] = (
        Path(emb["path"]) if emb.get("source") == "local"
        else paths.EMBEDDER_DIR / g["EMBEDDER_KEY"]
    )

    enabled = bool(llm.get("enabled", True) and (llm.get("filename") or llm.get("path")))
    g["LLM_ENABLED"] = enabled
    g["LLM_NAME"] = llm.get("name", "")
    g["LLM_REPO"] = llm.get("repo", "")
    g["LLM_FILENAME"] = llm.get("filename")
    g["LLM_PATH"] = (
        (Path(llm["path"]) if llm.get("source") == "local" else paths.LLM_DIR / llm["filename"])
        if enabled else None
    )
    g["LLM_N_CTX"] = int(llm.get("ctx", 4096))
    g["LLM_N_GPU_LAYERS"] = int(sel.get("llm", {}).get("n_gpu_layers", -1))
    g["DEVICE"] = sel.get("device", "auto")
    g["CUDA_TAG"] = sel.get("cuda_tag")


def selection():
    return {
        "device": DEVICE,
        "cuda_tag": CUDA_TAG,
        "embedder": {"key": EMBEDDER_KEY, "repo": EMBEDDER_REPO, "dim": EMBEDDER_DIM},
        "llm": {
            "enabled": LLM_ENABLED,
            "name": LLM_NAME,
            "repo": LLM_REPO,
            "filename": LLM_FILENAME,
            "n_gpu_layers": LLM_N_GPU_LAYERS,
        },
    }


def save_selection(data):
    paths.SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.SELECTION_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def reload():
    load()


load()

LLM_N_BATCH = 256
LLM_MAX_TOKENS = 400
LLM_NLU_TOKENS = 300
LLM_TEMP_NLU = 0.0
LLM_TEMP_REPLY = 0.7
LLM_REPEAT_PENALTY = 1.15

CPU_THREADS = max(2, (os.cpu_count() or 4) // 2)
EMBED_BATCH = 16

TOP_K = 5
COMPARE_MAX = 3
FUZZY_CUTOFF = 0.80
FUZZY_NAME_CUTOFF = 0.70          # per word of a typed name (see text.name_score)
FUZZY_NAME_JOINED_CUTOFF = 0.82   # whole name with spaces ignored
FUZZY_COMPARE_CUTOFF = 0.50
SERVICE_SIM_MIN = 0.80
def _env_float(name, default, low=0.0, high=1.0):
    """Read a number from the environment; fail loudly on junk instead of silently ignoring it."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


# Final ranking of already-filtered candidates = RANK_SIM_WEIGHT * semantic match with the
# user's message + (1 - RANK_SIM_WEIGHT) * rating. Unknown rating counts as RATING_DEFAULT / 5.
RANK_SIM_WEIGHT = _env_float("FINDPRO_RANK_SIM_WEIGHT", 0.35)
RATING_DEFAULT = 3.0
# Name lookup by embedding is only trusted above this cosine similarity; below it the answer is
# "not found" instead of a guess. Override with FINDPRO_NAME_SIM_MIN (see eval/README.md).
NAME_SIM_MIN = _env_float("FINDPRO_NAME_SIM_MIN", 0.90, low=-1.0)

HISTORY_LIMIT = 16
NLU_HISTORY = 4
CATALOG_LIMIT = 400

ADMIN_TOKEN = os.environ.get("FINDPRO_ADMIN_TOKEN", "")
HOST = os.environ.get("FINDPRO_HOST", "127.0.0.1")
DEBUG = os.environ.get("FINDPRO_DEBUG") == "1"
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
MAX_UPLOAD = 50 * 1024 * 1024
MAX_TEXT = 2000

# Shared secret between the gateway and its model workers. The gateway generates a random
# one at start and passes it to the worker processes through the environment.
WORKER_TOKEN = os.environ.get("FINDPRO_WORKER_TOKEN") or secrets.token_urlsafe(24)
WORKER_HEALTH_TIMEOUT = 5.0
WORKER_CALL_TIMEOUT = 90.0
WORKER_COOLDOWN = 20.0
WORKER_MAX_RESTARTS = 5


def is_loopback(host):
    return (host or "").strip().lower() in LOOPBACK_HOSTS


_SELECT_TOP = {"device", "cuda_tag", "embedder", "llm"}
_SELECT_LLM = {"enabled", "name", "filename", "n_gpu_layers"}


def validate_selection(data):
    """Check a selection sent through the API. Raises ValueError with a clear message.

    Only models from the catalog are accepted: a local file path (source=local) can only be
    chosen from the terminal wizard, never over HTTP.
    """
    if not isinstance(data, dict):
        raise ValueError("selection must be an object")
    extra = set(data) - _SELECT_TOP
    if extra:
        raise ValueError(f"unknown keys: {sorted(extra)}")
    cat = _read(paths.MODELS_CATALOG, {"embedders": [], "llms": []})
    clean = {}
    if "device" in data:
        if data["device"] not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be auto, cpu or gpu")
        clean["device"] = data["device"]
    if "cuda_tag" in data:
        tag = data["cuda_tag"]
        if tag is not None and not (isinstance(tag, str) and tag.replace("cu", "").isdigit()):
            raise ValueError("cuda_tag must look like cu121")
        clean["cuda_tag"] = tag
    if "embedder" in data:
        emb = data["embedder"]
        keys = {e["key"] for e in cat.get("embedders", [])}
        if not isinstance(emb, dict) or set(emb) - {"key"} or emb.get("key") not in keys:
            raise ValueError(f"embedder must be {{\"key\": one of {sorted(keys)}}}")
        clean["embedder"] = {"key": emb["key"]}
    if "llm" in data:
        llm = data["llm"]
        if not isinstance(llm, dict) or set(llm) - _SELECT_LLM:
            raise ValueError(f"llm accepts only {sorted(_SELECT_LLM)}")
        if llm.get("enabled") is False:
            clean["llm"] = {"enabled": False}
        else:
            models = {m["name"]: m for m in cat.get("llms", [])}
            if llm.get("name") not in models:
                raise ValueError(f"llm.name must be one of {sorted(models)}")
            entry = {"name": llm["name"]}
            if "filename" in llm:
                fn = llm["filename"]
                if fn is not None and not (isinstance(fn, str) and re.fullmatch(r"[\w.\-]+\.gguf", fn)):
                    raise ValueError("llm.filename must be a plain .gguf file name")
                entry["filename"] = fn
            if "n_gpu_layers" in llm:
                n = llm["n_gpu_layers"]
                if isinstance(n, bool) or not isinstance(n, int) or not -1 <= n <= 200:
                    raise ValueError("n_gpu_layers must be an integer between -1 and 200")
                entry["n_gpu_layers"] = n
            clean["llm"] = entry
    return clean

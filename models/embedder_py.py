import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src import settings


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True, name="embedder-load").start()
    yield


app = FastAPI(title="FindPro Embedder", lifespan=lifespan)

_lock = threading.Lock()
_state = {
    "state": "starting",
    "progress": 0,
    "message": "initializing",
    "memory_mb": 0,
    "model_key": settings.EMBEDDER_KEY,
    "dim": settings.EMBEDDER_DIM,
}
_model = None


def _set(state, message="", progress=None):
    with _lock:
        _state["state"] = state
        if message:
            _state["message"] = message
        if progress is not None:
            _state["progress"] = progress
        try:
            import resource
            _state["memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            pass


def _load():
    global _model
    try:
        _set("loading", "importing torch", 10)
        from sentence_transformers import SentenceTransformer

        _set("loading", "loading model", 30)
        path = settings.EMBEDDER_DIR_PATH
        if not Path(path).exists():
            print(f"[embedder] model not found: {path}", flush=True)
            _set("error", f"model not found: {path}")
            return
        _model = SentenceTransformer(str(path), device="cpu")
        try:
            import torch
            torch.set_num_threads(settings.CPU_THREADS)
        except Exception:
            pass
        _set("ready", "ready", 100)
    except BaseException as e:  # MemoryError must be reported too, not swallowed
        traceback.print_exc()
        print(f"[embedder] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        _set("error", f"load failed: {type(e).__name__}: {e}")




@app.get("/health")
def health():
    with _lock:
        return dict(_state)


class EmbedIn(BaseModel):
    texts: list[str]
    kind: str = "passage"


@app.post("/embed")
def embed(body: EmbedIn, x_worker_token: str | None = Header(default=None)):
    if x_worker_token != settings.WORKER_TOKEN:
        raise HTTPException(401, "unauthorized")
    with _lock:
        if _state["state"] != "ready" or _model is None:
            raise HTTPException(503, f"embedder not ready: {_state['state']}")
    prefix = settings.EMBEDDER_PASSAGE_PREFIX if body.kind == "passage" else settings.EMBEDDER_QUERY_PREFIX
    vectors = _model.encode(
        [prefix + t for t in body.texts],
        batch_size=settings.EMBED_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {"vectors": [v.tolist() for v in vectors]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("EMBEDDER_PORT", "9000")))

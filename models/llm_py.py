import argparse
import json
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src import settings

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True, name="llm-load").start()
    yield


app = FastAPI(title="FindPro LLM", lifespan=lifespan)

_lock = threading.Lock()
_state = {
    "state": "starting",
    "progress": 0,
    "message": "initializing",
    "memory_mb": 0,
    "model": settings.LLM_NAME,
}
_llm = None
_run_lock = threading.Lock()


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
    global _llm
    try:
        if not settings.LLM_ENABLED or not settings.LLM_PATH:
            _set("error", "LLM disabled")
            return
        if not settings.LLM_PATH.exists():
            _set("error", f"model not found: {settings.LLM_PATH}")
            return
        _set("loading", "importing llama_cpp", 10)
        from llama_cpp import Llama

        _set("loading", "loading model", 30)
        _llm = Llama(
            model_path=str(settings.LLM_PATH),
            n_gpu_layers=settings.LLM_N_GPU_LAYERS,
            n_ctx=settings.LLM_N_CTX,
            n_threads=settings.CPU_THREADS,
            n_batch=settings.LLM_N_BATCH,
            verbose=False,
        )
        _set("ready", "ready", 100)
    except BaseException as e:
        traceback.print_exc()
        hint = ""
        if "Failed to load model" in str(e):
            hint = " (usually not enough free RAM, or an incomplete/corrupt GGUF file)"
        print(f"[llm] LOAD FAILED: {type(e).__name__}: {e}{hint}", flush=True)
        _set("error", f"load failed: {type(e).__name__}: {e}{hint}")




def _auth(token):
    if token != settings.WORKER_TOKEN:
        raise HTTPException(401, "unauthorized")


@app.get("/health")
def health():
    with _lock:
        return dict(_state)


@app.get("/v1/models")
def models(x_worker_token: str | None = Header(default=None)):
    _auth(x_worker_token)
    return {"object": "list", "data": [{"id": settings.LLM_NAME, "object": "model"}]}


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = 400
    temperature: float = 0.2
    stream: bool = False
    response_format: dict | None = None
    repeat_penalty: float = 1.0


@app.post("/v1/chat/completions")
def completions(body: ChatIn, x_worker_token: str | None = Header(default=None)):
    _auth(x_worker_token)
    with _lock:
        if _state["state"] != "ready" or _llm is None:
            raise HTTPException(503, f"LLM not ready: {_state['state']}")
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    extra = {}
    if body.response_format and body.response_format.get("type") == "json_object":
        extra["response_format"] = {"type": "json_object"}

    if body.stream:
        def gen():
            with _run_lock:
                try:
                    for chunk in _llm.create_chat_completion(
                        messages=messages, max_tokens=body.max_tokens,
                        temperature=body.temperature, repeat_penalty=body.repeat_penalty,
                        stream=True, **extra,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode("utf-8")

        return StreamingResponse(gen(), media_type="text/event-stream")

    with _run_lock:
        try:
            return _llm.create_chat_completion(
                messages=messages, max_tokens=body.max_tokens,
                temperature=body.temperature, repeat_penalty=body.repeat_penalty, **extra,
            )
        except Exception as e:
            raise HTTPException(500, f"LLM call failed: {e}")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

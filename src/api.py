import asyncio
import hmac
import json
import queue
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Path,
                     Request, UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import chat, ingest, paths, settings
from .db import Busy, store
from .runtime import bus, supervisor
from .text import split_list

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


SESSION_ID = r"^[A-Za-z0-9_-]{8,64}$"


@asynccontextmanager
async def lifespan(app):
    if not settings.ADMIN_TOKEN and not settings.is_loopback(settings.HOST):
        raise RuntimeError(
            f"Refusing to listen on {settings.HOST} without FINDPRO_ADMIN_TOKEN: "
            "set a token or bind to 127.0.0.1."
        )
    paths.ensure_runtime_dirs()
    store.init_schema()
    store.recover_stale_jobs()
    supervisor.start_all(llm_enabled=settings.LLM_ENABLED)
    bus.publish("boot", "gateway", {"step": "ready"})
    yield
    supervisor.stop_all()


app = FastAPI(title="FindPro", version="1.1.0", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)
    text: str = Field(max_length=settings.MAX_TEXT)


def require_admin(request: Request, x_admin_token: str | None = Header(default=None)):
    """Guard for everything that changes data, runs installs or shows debug output.

    With FINDPRO_ADMIN_TOKEN set, the X-Admin-Token header must match it. Without a token the
    call is accepted only from the same machine and only when it is not a cross-site browser
    request (a web page you visit cannot reach your local server through this guard).
    """
    if settings.ADMIN_TOKEN:
        if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
            raise HTTPException(401, "missing or wrong X-Admin-Token")
        return
    host = request.client.host if request.client else ""
    if not settings.is_loopback(host):
        raise HTTPException(403, "admin actions from other machines need FINDPRO_ADMIN_TOKEN")
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host", ""):
        raise HTTPException(403, "cross-site request refused")


ADMIN = [Depends(require_admin)]


class WorkerIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    city: str = Field(min_length=2, max_length=60)
    service: str = Field(min_length=2, max_length=60)
    gender: str | None = Field(default=None, pattern="^(male|female)$")
    age: int | None = Field(default=None, ge=14, le=90)
    nationality: str | None = None
    experience_years: int | None = Field(default=None, ge=0, le=70)
    rating: float | None = Field(default=None, ge=0, le=5)
    hourly_rate: int | None = Field(default=None, ge=0)
    completed_projects: int | None = Field(default=None, ge=0)
    phone: str | None = None
    skills: str = ""


def _text(body):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    return text


@app.post("/api/chat")
def chat_once(body: ChatIn):
    done = chat.handle(body.session_id, _text(body))
    return {
        "reply": done.get("text", ""),
        "workers": done.get("workers", []),
        "intent": done.get("intent", ""),
        "message_id": done.get("message_id", 0),
        "stages": done.get("stages", []),
        "latency_ms": done.get("latency_ms", 0),
        "lang": done.get("lang", "en"),
        "dir": done.get("dir", "ltr"),
    }


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn):
    text = _text(body)

    def gen():
        for event in chat.handle_stream(body.session_id, text):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)


@app.get("/api/history/{session_id}")
def history(session_id: str = Path(pattern=SESSION_ID), limit: int = 100):
    return {"items": store.history(session_id, min(max(limit, 1), 300))}


@app.post("/api/explain", dependencies=ADMIN)
def explain(body: ChatIn):
    return chat.explain(_text(body), body.session_id)


@app.get("/api/workers")
def list_workers(q: str = "", city_id: int | None = None, service_id: int | None = None,
                 limit: int = 200, offset: int = 0):
    items, total = store.search_workers(q, city_id, service_id, min(limit, 500), offset)
    return {"items": items, "total": total}


@app.get("/api/workers/{worker_id}")
def get_worker(worker_id: int):
    worker = store.worker(worker_id)
    if not worker:
        raise HTTPException(404, "Worker not found")
    return worker


def _save_worker(body, worker_id=None):
    rec = body.model_dump()
    rec["city_id"] = store.entity_id("city", body.city, create=True)
    rec["service_id"] = store.entity_id("service", body.service, create=True)
    rec["skills"] = split_list(body.skills)
    if worker_id is None:
        worker_id = store.insert_worker(rec)
    elif not store.update_worker(worker_id, rec):
        raise HTTPException(404, "Worker not found")
    ingest.embed_worker_later(worker_id)
    return store.worker(worker_id)


@app.post("/api/workers", dependencies=ADMIN)
def add_worker(body: WorkerIn):
    return _save_worker(body)


@app.put("/api/workers/{worker_id}", dependencies=ADMIN)
def edit_worker(worker_id: int, body: WorkerIn):
    return _save_worker(body, worker_id)


@app.delete("/api/workers/{worker_id}", dependencies=ADMIN)
def delete_worker(worker_id: int):
    if not store.delete_worker(worker_id):
        raise HTTPException(404, "Worker not found")
    return {"ok": True}


@app.delete("/api/workers", dependencies=ADMIN)
def clear_workers():
    return {"ok": True, "deleted": store.clear_workers()}


def _begin(kind, label, total):
    try:
        return store.job_begin(kind, label, total)
    except Busy as e:
        raise HTTPException(409, str(e))


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    return {"items": store.jobs_recent(limit)}


@app.post("/api/reindex", dependencies=ADMIN)
def reindex(background: BackgroundTasks):
    job = _begin("reindex", "Reindex", 1)
    background.add_task(ingest.run_reindex, job["job_id"])
    return {"ok": True, "job_id": job["job_id"]}


@app.post("/api/seed_demo", dependencies=ADMIN)
def seed_demo(background: BackgroundTasks):
    job = _begin("seed_demo", "Import demo data", 1)
    background.add_task(ingest.run_seed, job["job_id"])
    return {"ok": True, "job_id": job["job_id"]}


@app.post("/api/upload", dependencies=ADMIN)
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    name = file.filename or ""
    if not name.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(400, "An .xlsx file is required")
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD:
        raise HTTPException(413, "File is too large")
    try:
        records, report = await run_in_threadpool(ingest.parse_excel, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    job = _begin("upload", f"Upload {name}", len(records))
    background.add_task(ingest.run_import, job["job_id"], records, True)
    return {"ok": True, "job_id": job["job_id"], "rows": len(records),
            "invalid_rows": report["invalid_rows"]}


@app.get("/api/template")
def template():
    return Response(
        ingest.template_bytes(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="workers_template.xlsx"'},
    )


@app.get("/api/meta")
def meta():
    return store.meta()


@app.get("/api/system/status")
def system_status():
    return supervisor.status()


@app.post("/api/system/restart/{name}", dependencies=ADMIN)
def restart_worker(name: str):
    if name not in ("embedder", "llm"):
        raise HTTPException(400, "worker name must be embedder or llm")
    if not supervisor.restart(name):
        raise HTTPException(404, f"worker {name} not found")
    return {"ok": True}


@app.get("/api/system/hardware")
def hardware():
    from .hardware import detect_hardware
    return detect_hardware().to_dict()


@app.get("/api/system/selection")
def selection():
    return settings.selection()


@app.get("/api/system/events")
async def system_events(request: Request):
    q, history = bus.subscribe()

    def frame(event):
        return f"event: {event.kind}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

    async def gen():
        try:
            for event in history:
                if await request.is_disconnected():
                    return
                yield frame(event)
            loop = asyncio.get_running_loop()
            while not await request.is_disconnected():
                try:
                    event = await loop.run_in_executor(None, lambda: q.get(timeout=15.0))
                    yield frame(event)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)


@app.post("/api/admin/install/{component}", dependencies=ADMIN)
def admin_install(component: str):
    from . import installer
    if component == "embedder":
        installer.install_embedder_deps()
    elif component == "llm":
        installer.install_llm_deps(settings.DEVICE, settings.CUDA_TAG)
    else:
        raise HTTPException(400, "component must be embedder or llm")
    return {"ok": True, "component": component}


@app.post("/api/admin/download/{component}", dependencies=ADMIN)
def admin_download(component: str):
    from . import installer
    if component == "embedder":
        installer.download_embedder()
    elif component == "llm":
        if not settings.LLM_REPO:
            raise HTTPException(400, "LLM disabled in selection")
        installer.download_llm(settings.LLM_REPO, settings.LLM_FILENAME)
    else:
        raise HTTPException(400, "component must be embedder or llm")
    return {"ok": True, "component": component}


@app.post("/api/admin/select", dependencies=ADMIN)
def admin_select(body: dict):
    try:
        clean = settings.validate_selection(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    settings.save_selection(clean)
    settings.reload()
    return {"ok": True, "selection": settings.selection()}


@app.get("/health")
def health():
    try:
        count = store.count_all()
    except Exception:
        count = 0
    return {"api": "ok", "version": "1.1.0", "workers_count": count,
            "selection": settings.selection()}


@app.get("/", response_class=HTMLResponse)
def index():
    page = paths.WEB_DIR / "index.html"
    if not page.exists():
        return "<h1>FindPro</h1>"
    return page.read_text(encoding="utf-8")


for _name in ("css", "js"):
    if (paths.WEB_DIR / _name).exists():
        app.mount(f"/{_name}", StaticFiles(directory=str(paths.WEB_DIR / _name)), name=_name)

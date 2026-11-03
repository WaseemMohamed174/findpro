import io
import json

from fastapi.testclient import TestClient

from src import api
from tests.conftest import ROWS, add_workers, spec_json


def client(host="127.0.0.1"):
    return TestClient(api.app, client=(host, 50000))


def test_health_meta_and_workers_crud(env):
    store, *_ = env
    c = client()
    assert c.get("/health").json()["workers_count"] == 0
    body = {"name": "Alex Morgan", "city": "Cairo", "service": "Plumbing", "gender": "male",
            "skills": "a, b", "rating": 4.2}
    w = c.post("/api/workers", json=body).json()
    assert w["skills"] == ["a", "b"] and w["city_name"] == "Cairo"
    assert c.get(f"/api/workers/{w['worker_id']}").status_code == 200
    body["name"] = "Alex M"
    assert c.put(f"/api/workers/{w['worker_id']}", json=body).json()["name"] == "Alex M"
    assert c.get("/api/workers?q=alex").json()["total"] == 1
    meta = c.get("/api/meta").json()
    assert meta["workers_count"] == 1 and meta["cities"][0]["workers"] == 1
    assert c.post("/api/workers", json={**body, "gender": "x"}).status_code == 422
    assert c.delete(f"/api/workers/{w['worker_id']}").status_code == 200
    assert c.get("/api/workers/999").status_code == 404
    assert c.delete("/api/workers/999").status_code == 404


def test_chat_endpoints(seeded):
    store, llm, *_ = seeded
    c = client()
    llm.nlu_reply = spec_json(city="Cairo", service="Plumbing")
    r = c.post("/api/chat", json={"session_id": "session-a1", "text": "plumber cairo"}).json()
    assert r["intent"] == "search" and len(r["workers"]) == 2 and r["dir"] == "ltr"
    assert c.post("/api/chat", json={"session_id": "session-a1", "text": "  "}).status_code == 400
    with c.stream("POST", "/api/chat/stream", json={"session_id": "session-b1", "text": "hi"}) as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert json.loads(lines[-1][5:])["type"] == "done"
    e = c.post("/api/explain", json={"session_id": "session-a1", "text": "plumber cairo"}).json()
    assert e["spec"]["city"] == "Cairo" and len(e["workers"]) == 2


def test_template_and_upload_flow(env):
    store, llm, emb = env
    c = client()
    tpl = c.get("/api/template")
    assert tpl.status_code == 200 and tpl.content[:2] == b"PK"
    r = c.post("/api/upload", files={"file": ("w.txt", b"x")})
    assert r.status_code == 400
    r = c.post("/api/upload", files={"file": ("w.xlsx", b"not excel")})
    assert r.status_code == 400
    r = c.post("/api/upload", files={"file": ("w.xlsx", tpl.content)})
    assert r.status_code == 200 and r.json()["rows"] == 2
    jobs = c.get("/api/jobs").json()["items"]
    assert jobs[0]["status"] == "done" and store.count_all() == 2


def test_busy_job_returns_409(env):
    store, *_ = env
    store.job_begin("upload", "x", 1)
    assert client().post("/api/reindex").status_code == 409


def test_reindex_and_seed_jobs(seeded):
    c = client()
    assert c.post("/api/reindex").status_code == 200
    assert c.get("/api/jobs").json()["items"][0]["status"] == "done"


def test_system_and_admin_routes(env):
    c = client()
    assert c.post("/api/system/restart/foo").status_code == 400
    assert c.post("/api/system/restart/llm").status_code == 404
    assert "workers" in c.get("/api/system/status").json()
    assert "device" in c.get("/api/system/selection").json()
    assert c.post("/api/admin/install/foo").status_code == 400
    assert c.post("/api/admin/download/foo").status_code == 400


def test_index_served(env):
    r = client().get("/")
    assert r.status_code == 200 and "FindPro" in r.text


def test_history_endpoint(env):
    c = client()
    c.post("/api/chat", json={"session_id": "hist-0001", "text": "hello"})
    items = c.get("/api/history/hist-0001").json()["items"]
    assert [m["role"] for m in items] == ["user", "assistant"]
    assert c.get("/api/history/nobody-0").json()["items"] == []

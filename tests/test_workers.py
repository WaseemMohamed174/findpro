import importlib.util
import json
import sys
import textwrap
import time

from fastapi.testclient import TestClient

from src import paths, settings
from src.runtime import ManagedProcess, State


def load(name):
    spec = importlib.util.spec_from_file_location(name, paths.WORKERS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeModel:
    def encode(self, texts, **kw):
        import numpy as np
        return np.ones((len(texts), 4), dtype="float32")


def test_embedder_worker_endpoints(monkeypatch):
    mod = load("embedder_py")
    c = TestClient(mod.app)
    h = {"X-Worker-Token": settings.WORKER_TOKEN}
    assert c.get("/health").json()["state"] == "starting"
    assert c.post("/embed", json={"texts": ["a"]}, headers=h).status_code == 503
    mod._model = FakeModel()
    mod._state["state"] = "ready"
    monkeypatch.setattr(settings, "EMBEDDER_QUERY_PREFIX", "query: ")
    r = c.post("/embed", json={"texts": ["a", "b"], "kind": "query"}, headers=h)
    assert r.status_code == 200 and len(r.json()["vectors"]) == 2
    assert c.post("/embed", json={"texts": ["a"]}).status_code == 401


class FakeLlama:
    def create_chat_completion(self, messages, stream=False, **kw):
        if stream:
            return iter([{"choices": [{"delta": {"content": "he"}}]},
                         {"choices": [{"delta": {"content": "llo"}}]}])
        return {"choices": [{"message": {"content": "hi"}}], "kw": kw}


def test_llm_worker_endpoints():
    mod = load("llm_py")
    c = TestClient(mod.app)
    h = {"X-Worker-Token": settings.WORKER_TOKEN}
    body = {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}
    assert c.post("/v1/chat/completions", json=body, headers=h).status_code == 503
    mod._llm = FakeLlama()
    mod._state["state"] = "ready"
    r = c.post("/v1/chat/completions", json=body, headers=h).json()
    assert r["choices"][0]["message"]["content"] == "hi"
    assert r["kw"]["response_format"] == {"type": "json_object"}
    assert c.get("/v1/models").status_code == 401
    with c.stream("POST", "/v1/chat/completions", json={**body, "stream": True}, headers=h) as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[0][5:])["choices"][0]["delta"]["content"] == "he"


def test_clients_stream_and_chat_against_fake_server(tmp_path, monkeypatch):
    import socket
    import subprocess
    from src import clients
    from src.runtime import supervisor

    script = tmp_path / "srv.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"state": "ready"}).encode())
            def do_POST(self):
                n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
                self.send_response(200); self.end_headers()
                if body.get("stream"):
                    for t in ("a", "b"):
                        self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": t}}]}) + "\\n\\n").encode())
                    self.wfile.write(b"data: [DONE]\\n\\n")
                elif self.path == "/embed":
                    self.wfile.write(json.dumps({"vectors": [[1, 2]]}).encode())
                else:
                    self.wfile.write(json.dumps({"choices": [{"message": {"content": " " + str(body["temperature"]) + " "}}]}).encode())
        HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
    """))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, str(script), str(port)])
    try:
        mp = ManagedProcess("llm", lambda: [], port)
        mp.status.state = State.READY
        emb = ManagedProcess("embedder", lambda: [], port)
        emb.status.state = State.READY
        monkeypatch.setitem(supervisor._workers, "llm", mp)
        monkeypatch.setitem(supervisor._workers, "embedder", emb)
        time.sleep(0.5)
        assert clients.llm.ready()
        assert clients.llm.chat([{"role": "user", "content": "x"}], temperature=0.3) == "0.3"
        assert "".join(clients.llm.stream([{"role": "user", "content": "x"}])) == "ab"
        assert clients.embedder.embed(["x"]) == [[1, 2]]
        mp.status.state = State.STOPPED
        assert not clients.llm.ready()
        try:
            clients.llm.chat([])
            assert False
        except clients.LLMError:
            pass
    finally:
        proc.kill()


def test_managed_process_ready_and_crash_recovery(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(settings, "WORKER_COOLDOWN", 0.1)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    script = tmp_path / "w.py"
    script.write_text(textwrap.dedent(f"""
        import json, os, sys, time, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        marker = {str(tmp_path / 'started')!r}
        first = not os.path.exists(marker)
        open(marker, "w").write("x")
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({{"state": "ready", "memory_mb": 5}}).encode())
        if first:
            threading.Timer(1.5, lambda: os._exit(3)).start()
        HTTPServer(("127.0.0.1", {port}), H).serve_forever()
    """))
    mp = ManagedProcess("t", lambda: [sys.executable, str(script)], port)
    mp.start()
    try:
        for _ in range(60):
            if mp.is_ready():
                break
            time.sleep(0.2)
        assert mp.is_ready() and mp.status.memory_mb == 5
        for _ in range(100):
            if mp.status.restarted_count >= 1 and mp.is_ready():
                break
            time.sleep(0.2)
        assert mp.status.restarted_count == 1 and mp.is_ready()
        assert mp._monitor.is_alive()
    finally:
        mp.stop()
    assert mp.status.state == State.STOPPED


def test_llm_memory_preflight(tmp_path, monkeypatch):
    from src import runtime, settings
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0" * 1024 * 1024)
    monkeypatch.setattr(settings, "LLM_PATH", gguf)

    class HW:
        ram_avail = 0.2
    monkeypatch.setattr(runtime, "detect_hardware", lambda: HW)
    ok, need, avail = runtime.llm_memory_check()
    assert not ok and avail == 0.2 and need >= 0.6

    HW.ram_avail = 8.0
    assert runtime.llm_memory_check()[0] is True

    def boom():
        raise RuntimeError("no psutil")
    monkeypatch.setattr(runtime, "detect_hardware", boom)
    assert runtime.llm_memory_check()[0] is True  # unknown RAM must not block startup


def test_supervisor_blocks_llm_when_ram_low(monkeypatch):
    from src import runtime
    sup = runtime.Supervisor()
    launched = []
    monkeypatch.setattr(sup, "_launch", lambda name, b: launched.append(name))
    monkeypatch.setattr(runtime, "llm_memory_check", lambda: (False, 3.1, 1.0))
    sup._launch_llm()
    assert launched == []
    st = sup.get_worker("llm").status
    assert st.state == runtime.State.FAILED and "RAM" in st.message


def test_llm_launches_only_after_embedder_settles(monkeypatch):
    from src import runtime
    sup = runtime.Supervisor()
    sup._started = True
    emb = runtime.ManagedProcess("embedder", lambda: [], 1)
    emb.status.state = runtime.State.LOADING
    sup._workers["embedder"] = emb
    launched = []
    monkeypatch.setattr(sup, "_launch_llm", lambda: launched.append("llm"))
    import threading, time
    t = threading.Thread(target=sup._launch_llm_after_embedder, args=(5.0,))
    t.start()
    time.sleep(1.0)
    assert launched == []          # embedder still loading -> LLM must wait
    emb.status.state = runtime.State.READY
    t.join(3)
    assert launched == ["llm"]

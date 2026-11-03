import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass

from . import log, paths, settings
from .hardware import Hardware, detect_hardware


@dataclass
class Event:
    kind: str
    source: str
    payload: dict
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()

    def to_dict(self):
        return asdict(self)


logger = log.get("supervisor")


class EventBus:
    def __init__(self, max_queue=200, max_history=50):
        self._subscribers = []
        self._history = []
        self._lock = threading.Lock()
        self._max_queue = max_queue
        self._max_history = max_history

    def publish(self, kind, source, payload=None):
        if not isinstance(payload, dict):
            payload = {"value": payload}
        event = Event(kind=kind, source=source, payload=payload)
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self):
        q = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subscribers.append(q)
            history = list(self._history)
        return q, history

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


bus = EventBus()


class State:
    NOT_INSTALLED = "NOT_INSTALLED"
    STARTING = "STARTING"
    LOADING = "LOADING"
    READY = "READY"
    CRASHED = "CRASHED"
    BACKOFF = "BACKOFF"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class Status:
    name: str
    state: str = State.NOT_INSTALLED
    progress: int = 0
    message: str = ""
    pid: int | None = None
    port: int | None = None
    memory_mb: float = 0.0
    last_crash: str = ""
    last_crash_reason: str = ""
    restarted_count: int = 0

    def to_dict(self):
        return asdict(self)


class ManagedProcess:
    def __init__(self, name, cmd_builder, port):
        self.name = name
        self.port = port
        self.status = Status(name=name, port=port)
        self._cmd_builder = cmd_builder
        self._health_url = f"http://127.0.0.1:{port}/health"
        self._proc = None
        self._monitor = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._restarts = 0

    def _set_state(self, state, message="", progress=None):
        if self.status.state == state and (not message or message == self.status.message):
            return
        self.status.state = state
        if message:
            self.status.message = message
        if progress is not None:
            self.status.progress = progress
        if state in (State.FAILED, State.CRASHED):
            logger.warning("%s %s: %s", self.name, state, message)
        bus.publish("worker_state", self.name, self.status.to_dict())

    def start(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._stop.clear()
            self._set_state(State.STARTING, "starting process")
            try:
                env = dict(os.environ)
                env["NO_PROXY"] = "127.0.0.1,localhost"
                env["TOKENIZERS_PARALLELISM"] = "false"
                env["FINDPRO_WORKER_TOKEN"] = settings.WORKER_TOKEN
                env["PYTHONPATH"] = str(paths.BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
                paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
                log = open(paths.LOGS_DIR / f"{self.name}.log", "a", encoding="utf-8")
                self._proc = subprocess.Popen(
                    self._cmd_builder(), stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(paths.BASE_DIR), env=env,
                )
                self.status.pid = self._proc.pid
            except Exception as e:
                self._set_state(State.FAILED, f"start failed: {e}")
                return
            if self._monitor is None or not self._monitor.is_alive():
                self._monitor = threading.Thread(
                    target=self._watch, daemon=True, name=f"monitor-{self.name}"
                )
                self._monitor.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    time.sleep(1)
                    if self._proc.poll() is None:
                        self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self.status.pid = None
        self._set_state(State.STOPPED, "stopped")

    def restart(self):
        self.stop()
        time.sleep(0.5)
        self._restarts = 0
        self.start()

    def _watch(self):
        loading = announced = False
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self._stop.is_set():
            time.sleep(0.5)
            if self._proc is None or self._proc.poll() is not None:
                code = self._proc.poll() if self._proc else -1
                if not self._recover(f"process exited (code={code})"):
                    return
                loading = announced = False
                continue
            try:
                with opener.open(self._health_url, timeout=settings.WORKER_HEALTH_TIMEOUT) as resp:
                    health = json.loads(resp.read().decode("utf-8"))
                state = health.get("state", "loading")
                progress = health.get("progress", 0)
                if state == "ready":
                    if not announced:
                        self._set_state(State.READY, "ready", progress=100)
                        announced = True
                        self._restarts = 0
                elif state in ("loading", "starting"):
                    if not loading:
                        self._set_state(State.LOADING, "loading model", progress=progress)
                        loading = True
                    else:
                        self.status.progress = progress
                elif state == "error":
                    self._set_state(State.FAILED, health.get("message", "error"))
                if "memory_mb" in health:
                    self.status.memory_mb = float(health["memory_mb"])
            except Exception:
                if self.status.state == State.STARTING:
                    self._set_state(State.LOADING, "loading")

    def _recover(self, reason):
        self.status.last_crash = str(int(time.time()))
        self.status.last_crash_reason = reason
        self.status.restarted_count += 1
        self._set_state(State.CRASHED, reason)
        if self._restarts >= settings.WORKER_MAX_RESTARTS:
            self._set_state(State.FAILED, f"max restarts ({settings.WORKER_MAX_RESTARTS}) exceeded")
            return False
        backoff = settings.WORKER_COOLDOWN * (2 ** min(self._restarts, 4))
        self._restarts += 1
        self._set_state(State.BACKOFF, f"retry in {backoff:.0f}s")
        if self._stop.wait(backoff):
            return False
        self.start()
        return True

    def is_ready(self):
        return self.status.state == State.READY


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _python(name):
    py = paths.venv_python(name)
    return str(py) if py.exists() else sys.executable


def _embedder_cmd(port):
    return [
        _python("embedder"), "-m", "uvicorn", "embedder_py:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--app-dir", str(paths.WORKERS_DIR),
    ]


def _llm_cmd(port):
    return [_python("llm"), str(paths.LLM_WORKER), "--host", "127.0.0.1", "--port", str(port)]


def llm_memory_check():
    """Return (ok, need_gb, avail_gb). Never raises; unknown RAM counts as ok."""
    try:
        size = settings.LLM_PATH.stat().st_size / 1024 ** 3 if settings.LLM_PATH else 0.0
        need = size * 1.2 + 0.6  # weights + KV cache / buffers
        avail = detect_hardware().ram_avail
    except Exception:
        return True, 0.0, 0.0
    return (avail <= 0 or avail >= need), round(need, 2), round(avail, 2)


class Supervisor:
    def __init__(self):
        self._workers = {}
        self._lock = threading.Lock()
        self._hardware = None
        self._started = False

    def start_all(self, llm_enabled=True):
        with self._lock:
            if self._started:
                return
            self._started = True
        bus.publish("boot", "supervisor", {"step": "starting_workers"})
        self._launch("embedder", _embedder_cmd)
        if llm_enabled and settings.LLM_ENABLED:
            # Load one model at a time: torch + llama.cpp loading together is the RAM peak.
            threading.Thread(target=self._launch_llm_after_embedder, daemon=True,
                             name="llm-launcher").start()
        else:
            bus.publish("boot", "llm", {"step": "disabled"})

    def _launch_llm_after_embedder(self, max_wait=300.0):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            emb = self.get_worker("embedder")
            if emb is None or emb.status.state in (State.READY, State.FAILED, State.STOPPED):
                break
            time.sleep(0.5)
        if self._started:
            self._launch_llm()

    def _launch_llm(self):
        ok, need, avail = llm_memory_check()
        if ok:
            self._launch("llm", _llm_cmd)
            return
        process = self._register("llm", _llm_cmd)
        process._set_state(
            State.FAILED,
            f"not enough free RAM for the LLM: needs ~{need} GB, {avail} GB available. "
            "Free memory or choose a smaller model, then click to restart.",
        )

    def _register(self, name, builder):
        port = _free_port()
        process = ManagedProcess(name, lambda: builder(port), port)
        with self._lock:
            self._workers[name] = process
        return process

    def _launch(self, name, builder):
        self._register(name, builder).start()

    def get_worker(self, name):
        with self._lock:
            return self._workers.get(name)

    def restart(self, name):
        process = self.get_worker(name)
        if not process and name == "llm" and settings.LLM_ENABLED:
            threading.Thread(target=self._launch_llm, daemon=True, name="launch-llm").start()
            return True
        if not process:
            return False
        threading.Thread(target=process.restart, daemon=True, name=f"restart-{name}").start()
        return True

    def status(self):
        with self._lock:
            workers = {n: p.status.to_dict() for n, p in self._workers.items()}
        if self._hardware is None:
            try:
                self._hardware = detect_hardware()
            except Exception:
                self._hardware = Hardware()
        return {
            "workers": workers,
            "hardware": self._hardware.to_dict(),
            "selection": settings.selection(),
            "booted": self._started,
        }

    def stop_all(self):
        with self._lock:
            processes = list(self._workers.values())
        for p in processes:
            try:
                p.stop()
            except Exception:
                pass
        self._started = False


supervisor = Supervisor()

import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src import paths, settings
from src.cli import ui

HOST = os.environ.get("FINDPRO_HOST", "127.0.0.1")
PORT = int(os.environ.get("FINDPRO_PORT", "8000"))
URL = f"http://{HOST}:{PORT}/"

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _create_venv(name):
    py = paths.venv_python(name)
    if py.exists():
        return py
    subprocess.check_call([sys.executable, "-m", "venv", str(paths.VENV_DIR / name)])
    for pkg in ("pip", "wheel", "setuptools"):
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", pkg],
                       capture_output=True, check=False)
    return py


def _ensure_gateway_venv():
    py = paths.venv_python("gateway")
    if Path(sys.executable).absolute() == py.absolute():
        return
    paths.VENV_DIR.mkdir(parents=True, exist_ok=True)
    if not py.exists():
        ui.info("Creating the gateway environment (first run only) ...")
    _create_venv("gateway")
    os.execv(str(py), [str(py), __file__] + sys.argv[1:])


def _start_gateway():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    log = open(paths.LOGS_DIR / "gateway.log", "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "uvicorn", "src.api:app",
           "--host", HOST, "--port", str(PORT), "--log-level", "info"]
    return subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT, env=env)


def _wait_ready(proc, max_wait=60):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if proc.poll() is not None:
            return False
        try:
            with _NO_PROXY.open(URL + "health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    if not settings.ADMIN_TOKEN and not settings.is_loopback(HOST):
        print(f"Refusing to listen on {HOST} without FINDPRO_ADMIN_TOKEN.\n"
              "Set a token (export FINDPRO_ADMIN_TOKEN=...) or use FINDPRO_HOST=127.0.0.1.")
        sys.exit(2)
    _ensure_gateway_venv()

    from src.cli import deps, select

    ui.step(1, "Libraries")
    paths.ensure_runtime_dirs()
    install_log = paths.LOGS_DIR / "install.log"
    auto = "--auto" in sys.argv or os.environ.get("FINDPRO_AUTO") == "1"
    no_llm = "--no-llm" in sys.argv

    ui.section("Checking environments")
    pending = []
    for venv_name, specs in (
        ("gateway", deps.GATEWAY),
        ("embedder", deps.EMBEDDER),
        ("llm", deps.LLM_BASE),
    ):
        py = _create_venv(venv_name)
        missing = deps.missing_by_name(py, [s for s, _ in specs])
        for spec, extra in specs:
            if spec in missing:
                pending.append({
                    "spec": spec, "extra": extra, "py": py,
                    "lib": deps.base_name(spec),
                    "component": venv_name.capitalize(), "venv": venv_name,
                })

    if pending:
        deps.show_missing(pending)
        mode, central = deps.choose_source()
        deps.install_all(pending, mode, central, install_log)
    else:
        ui.ok("All required libraries are present")

    ui.step(2, "Model selection")
    selection = select.run(no_llm=no_llm, auto=auto)

    ui.step(3, "Download models")
    from src import installer, settings
    settings.reload()
    installer.set_install_mode("online", None)
    try:
        installer.ensure_all(selection)
    except SystemExit:
        raise
    except Exception as e:
        ui.warn(f"Model download failed: {e}")
        ui.warn("Continuing without models. Download later from /api/admin/download")
    settings.reload()

    ui.step(4, "Launch")
    ui.info("Starting gateway ...")
    proc = _start_gateway()

    def shutdown(*_):
        ui.info("Stopping ...")
        try:
            if proc.poll() is None:
                proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()
        except Exception:
            pass
        sys.exit(0)

    if not _wait_ready(proc):
        ui.fail("Gateway failed to start. Last log lines:")
        try:
            lines = (paths.LOGS_DIR / "gateway.log").read_text(
                encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-25:]))
        except Exception:
            pass
        shutdown()

    ui.ok(f"Gateway is up: {URL}")
    try:
        webbrowser.open(URL)
    except Exception:
        pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    ui.info("Running. Press Ctrl+C to stop.")
    try:
        while proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    shutdown()


if __name__ == "__main__":
    main()

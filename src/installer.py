from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import paths, settings
from .cli import deps, ui
from .runtime import bus


_install_mode: str = "online"
_central_path: str | None = None


def set_install_mode(mode: str, central_path: str | None = None) -> None:
    global _install_mode, _central_path
    _install_mode = mode
    _central_path = central_path


def _venv_py(name: str) -> Path:
    return paths.venv_python(name)


def _install_log() -> Path:
    return paths.LOGS_DIR / "install.log"


def ensure_all(selection: dict) -> None:

    llm = selection.get("llm") or {}
    if llm.get("enabled"):
        ui.section("Inference engine", note="llama-cpp-python")
        _ensure_llama_build(selection)

    emb = selection.get("embedder") or {}
    ui.section("Embedding model", note=emb.get("key", ""))
    if emb.get("source") == "local":
        ui.ok(f"Using local embedder: {emb.get('path')}")
        bus.publish("installer", "embedder", {"step": "local_skip"})
    else:
        download_embedder()

    ui.section("Language model", note=llm.get("name") or "")
    if not llm.get("enabled"):
        ui.ok("LLM disabled - nothing to download")
        bus.publish("installer", "llm", {"step": "skipped", "reason": "disabled"})
        return
    if llm.get("source") == "local":
        ui.ok(f"Using local LLM: {llm.get('path')}")
        bus.publish("installer", "llm", {"step": "local_skip"})
        return
    download_llm(llm["repo"], llm.get("filename"))


def _ensure_llama_build(selection: dict) -> None:
    device = selection.get("device", "cpu")
    cuda_tag = selection.get("cuda_tag")
    state_file = paths.RUNTIME_DIR / "deps.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        state = {}

    want = f"cuda-{cuda_tag}" if (device == "gpu" and cuda_tag) else "cpu"
    if state.get("llama_cpp_build") == want:
        ui.ok(f"llama-cpp-python already installed ({want})")
        return

    py = _venv_py("llm")
    ui.info(f"Installing llama-cpp-python ({want}) ...")
    index = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag if want != 'cpu' else 'cpu'}"
    prebuilt = [str(py), "-m", "pip", "install", "--upgrade", "llama-cpp-python",
                "--extra-index-url", index, "--only-binary", "llama-cpp-python"]
    source = [str(py), "-m", "pip", "install", "--upgrade", "llama-cpp-python"]
    rc = ui.run_quiet(prebuilt, _install_log())
    if rc != 0:
        if want != "cpu":
            ui.warn("No prebuilt GPU wheel found, falling back to CPU")
            want = "cpu"
            rc = ui.run_quiet(prebuilt[:-4] + ["--extra-index-url",
                              "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                              "--only-binary", "llama-cpp-python"], _install_log())
        if rc != 0:
            ui.warn("No prebuilt wheel for this Python. Building from source; this can take 10-30 minutes.")
            ui.info(f"Progress: tail -f {_install_log()}")
            rc = ui.run_quiet(source, _install_log())
            if rc != 0:
                raise SystemExit(1)

    state["llama_cpp_build"] = want
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _remember_filename(filename: str) -> None:
    try:
        sel = json.loads(paths.SELECTION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    sel.setdefault("llm", {})["filename"] = filename
    paths.SELECTION_FILE.write_text(
        json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    settings.reload()


def download_embedder() -> None:
    py = _venv_py("embedder")
    target = paths.EMBEDDER_DIR / settings.EMBEDDER_KEY
    target.mkdir(parents=True, exist_ok=True)

    if (target / "model.safetensors").exists() or\
            (target / "pytorch_model.bin").exists():
        ui.ok(f"Embedder cached: {settings.EMBEDDER_KEY}")
        bus.publish("installer", "embedder",
                    {"step": "model_done", "cached": True})
        return

    ui.info(f"Downloading embedder: {settings.EMBEDDER_REPO}")
    bus.publish("installer", "embedder",
                {"step": "downloading_model", "repo": settings.EMBEDDER_REPO})
    script = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={settings.EMBEDDER_REPO!r},
    local_dir={str(target)!r},
    allow_patterns={settings.EMBEDDER_FILES!r},
    ignore_patterns={settings.EMBEDDER_IGNORE!r},
)
print("done")
"""
    rc = ui.run_quiet([str(py), "-c", script], _install_log())
    if rc != 0:
        raise RuntimeError("embedder download failed")
    ui.ok("Embedder ready")
    bus.publish("installer", "embedder",
                {"step": "model_done", "cached": False})


def download_llm(repo: str, filename: str | None) -> Path:
    py = _venv_py("llm")
    paths.LLM_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = _resolve_filename(py, repo)
        _remember_filename(filename)
    target = paths.LLM_DIR / filename
    if target.exists():
        ui.ok(f"LLM cached: {filename}")
        bus.publish("installer", "llm",
                    {"step": "model_done", "cached": True, "path": str(target)})
        return target

    ui.info(f"Downloading LLM: {repo}/{filename}")
    bus.publish("installer", "llm",
                {"step": "downloading_model", "repo": repo, "filename": filename})
    script = f"""
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id={repo!r}, filename={filename!r},
                local_dir={str(paths.LLM_DIR)!r})
print("done")
"""
    rc = ui.run_quiet([str(py), "-c", script], _install_log())
    if rc != 0:
        raise RuntimeError("llm download failed")
    ui.ok(f"LLM ready: {filename}")
    bus.publish("installer", "llm",
                {"step": "model_done", "cached": False, "path": str(target)})
    return target


def _resolve_filename(py: Path, repo: str) -> str:
    script = (
        "from huggingface_hub import list_repo_files\n"
        f"files = list_repo_files(repo_id={repo!r})\n"
        "ggufs = [f for f in files if f.endswith('.gguf') "
        "and 'q4_k_m' in f.lower()]\n"
        "if not ggufs:\n"
        "    ggufs = [f for f in files if f.endswith('.gguf')]\n"
        "print(ggufs[0] if ggufs else '')\n"
    )
    res = subprocess.run([str(py), "-c", script], capture_output=True, text=True)
    name = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    if not name:
        raise RuntimeError(f"No .gguf file found in {repo}")
    return name


def install_embedder_deps() -> None:
    py = _venv_py("embedder")
    pending: list[dict] = []
    for spec, extra in deps.EMBEDDER:
        if deps.base_name(spec) in deps.missing_by_name(py, [spec]):
            pending.append({
                "spec": spec, "extra": extra, "py": py,
                "lib": deps.base_name(spec),
                "component": "Embedder", "venv": "embedder",
            })
    if pending:
        deps.install_all(pending, _install_mode, _central_path, _install_log())


def install_llm_deps(device: str, cuda_tag: str | None) -> None:
    py = _venv_py("llm")
    pending: list[dict] = []
    for spec, extra in deps.LLM_BASE:
        if deps.base_name(spec) in deps.missing_by_name(py, [spec]):
            pending.append({
                "spec": spec, "extra": extra, "py": py,
                "lib": deps.base_name(spec),
                "component": "LLM", "venv": "llm",
            })
    if pending:
        deps.install_all(pending, _install_mode, _central_path, _install_log())
    _ensure_llama_build({
        "device": device, "cuda_tag": cuda_tag, "llm": {"enabled": True},
    })

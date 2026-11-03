from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from . import ui


SELECTION_FILE = paths.SELECTION_FILE


def _need_embedder(size_gb: float) -> float:
    return size_gb * 1.3 + 0.25


def _need_llm(size_gb: float, params_b: float) -> float:
    return size_gb * 1.1 + 0.3 + 0.08 * params_b


def _load_catalog() -> dict:
    try:
        return json.loads(paths.MODELS_CATALOG.read_text(encoding="utf-8"))
    except Exception:
        return {"embedders": [], "llms": []}


def _show_hardware(hw) -> None:
    ui.section("Hardware")
    lines = [
        f"CPU cores : {hw.cores}",
        f"RAM       : {hw.ram_total:.1f} GB total · {hw.ram_avail:.1f} free",
        f"Safe RAM  : {hw.safe_ram:.1f} GB",
    ]
    if hw.gpu_name:
        lines.append(f"GPU       : {hw.gpu_name}")
        lines.append(f"VRAM      : {hw.vram_total:.1f} GB total · {hw.vram_free:.1f} free")
        lines.append(f"Safe VRAM : {hw.safe_vram:.1f} GB")
        if hw.cuda_tag:
            lines.append(f"CUDA      : {hw.cuda_version} ({hw.cuda_tag})")
        else:
            lines.append(f"CUDA      : {hw.cuda_version or '?'} (need 12.1+)")
    else:
        lines.append("GPU       : not detected (CPU mode)")
    ui.box("YOUR HARDWARE", lines)


def _ask_device(hw, auto: bool) -> tuple[str, str | None]:
    if not hw.gpu_available:
        ui.section("Running device")
        ui.ok("CPU mode (no GPU available)")
        return "cpu", None
    if auto:
        ui.section("Running device")
        ui.ok(f"GPU mode ({hw.cuda_tag})")
        return "gpu", hw.cuda_tag

    c = ui.menu("Running device", ["Device", "Details"], [
        {"cols": ["GPU", "Faster - LLM runs on VRAM"], "state": "rec",
         "desc": "Use the NVIDIA GPU. Requires CUDA 12.1 or newer. Best performance."},
        {"cols": ["CPU", "Slower but safe"],
         "desc": "Use the CPU only. Works everywhere. Slower for big models."},
    ])
    return ("gpu", hw.cuda_tag) if c == 0 else ("cpu", None)


def _custom_embedder() -> dict | None:
    p_str = ui.ask_text("Folder of the embedding model (sentence-transformers format):")
    if not p_str:
        return None
    p = Path(p_str).expanduser()
    cfg = p / "config.json"
    if not p.is_dir() or not cfg.exists():
        ui.fail("config.json not found in that folder.")
        return None
    if not any((p / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))\
            and not list(p.glob("*.safetensors")):
        ui.fail("No weights file found.")
        return None
    try:
        dim = int(json.loads(cfg.read_text(encoding="utf-8"))["hidden_size"])
    except Exception as e:
        ui.fail(f"Cannot read hidden_size from config.json: {e}")
        return None
    e5 = ui.ask_text("Does it use e5 prefixes ('query: ' / 'passage: ')? [y/N]:")\
        .strip().lower() == "y"
    return {
        "key": f"custom-{p.name}",
        "repo": "",
        "dim": dim,
        "source": "local",
        "path": str(p),
        "passage_prefix": "passage: " if e5 else "",
        "query_prefix": "query: " if e5 else "",
        "max_seq_length": 512,
    }


def _ask_embedder(hw, auto: bool) -> dict:
    embedders = sorted(
        _load_catalog().get("embedders", []),
        key=lambda e: e["size_gb"],
    )
    if not embedders:
        raise RuntimeError("catalog/models.json has no embedders")

    fits = [_need_embedder(e["size_gb"]) <= hw.safe_ram for e in embedders]
    rec = max((i for i, f in enumerate(fits) if f), default=None)
    note = f"safe budget: {hw.safe_ram:.1f} GB RAM"

    if auto:
        e = embedders[rec if rec is not None else 0]
        ui.section("Embedding model", note)
        ui.ok(f"Embedder: {e['key']}")
        return _hub_embedder(e)

    items = []
    for i, e in enumerate(embedders):
        items.append({
            "cols": [e["key"], f"{e['size_gb']:.2f} GB",
                     f"~{_need_embedder(e['size_gb']):.1f} GB"],
            "state": "blocked" if not fits[i] else ("rec" if i == rec else "ok"),
        })
    extras = [{"cols": ["Enter a local model path"],
               "desc": "Use a model already on your disk. Nothing is downloaded."}]
    explain_lines = [("{range}", "Pick a model from the list, listed from smallest to "
                                 "largest. Smaller uses less RAM, larger matches better.")]

    while True:
        c = ui.menu("Embedding model", ["Model", "Size", "Needs"], items,
                    extras=extras, explain_lines=explain_lines, note=note)
        if c >= len(items):
            custom = _custom_embedder()
            if custom:
                ui.ok(f"Embedder: {custom['key']}")
                return custom
            continue
        ui.ok(f"Embedder: {embedders[c]['key']}")
        return _hub_embedder(embedders[c])


def _hub_embedder(e: dict) -> dict:
    return {
        "key": e["key"], "repo": e["repo"], "dim": e["dim"],
        "source": "hub", "path": None,
        "passage_prefix": e.get("passage_prefix", "passage: "),
        "query_prefix": e.get("query_prefix", "query: "),
        "max_seq_length": e.get("max_seq_length", 512),
    }


def _custom_llm() -> dict | None:
    p_str = ui.ask_text("Path of the .gguf file:")
    if not p_str:
        return None
    p = Path(p_str).expanduser()
    if not p.is_file() or p.suffix.lower() != ".gguf":
        ui.fail("Not a .gguf file.")
        return None
    return {
        "enabled": True, "name": p.stem, "repo": "", "filename": p.name,
        "source": "local", "path": str(p), "ctx": 4096, "n_gpu_layers": -1,
    }


def _ask_llm(hw, device: str, no_llm: bool, auto: bool) -> dict:
    if no_llm:
        ui.section("Language model")
        ui.ok("LLM disabled (--no-llm)")
        return {"enabled": False, "name": None, "repo": None, "filename": None,
                "source": "none", "path": None, "ctx": 0, "n_gpu_layers": 0}

    llms = sorted(
        _load_catalog().get("llms", []),
        key=lambda m: m["params_b"],
    )
    if not llms:
        raise RuntimeError("catalog/models.json has no LLMs")

    budget = hw.safe_vram if device == "gpu" else max(hw.safe_ram - 1.0, 1.0)
    fits = [_need_llm(m["size_gb"], m["params_b"]) <= budget for m in llms]
    rec = max((i for i, f in enumerate(fits) if f), default=None)
    if device == "cpu":
        small = [i for i, m in enumerate(llms) if m["params_b"] <= 4.0 and fits[i]]
        if small:
            rec = small[-1]
    label = "VRAM" if device == "gpu" else "RAM"
    note = f"safe budget: {budget:.1f} GB {label}"

    if auto:
        m = llms[rec if rec is not None else 0]
        ui.section("Language model", note)
        ui.ok(f"LLM: {m['name']}")
        return _hub_llm(m, device)

    items = []
    for i, m in enumerate(llms):
        item = {
            "cols": [m["name"], f"{m['params_b']:.1f}B", f"{m['size_gb']:.2f} GB",
                     f"~{_need_llm(m['size_gb'], m['params_b']):.1f} GB"],
            "state": "blocked" if not fits[i] else ("rec" if i == rec else "ok"),
        }
        if item["state"] == "ok" and device == "cpu" and m["params_b"] > 4.0:
            item["tag"] = "⚠ slow on CPU"
        items.append(item)
    extras = [{"cols": ["Enter a local model path"],
               "desc": "Use a .gguf file already on your disk."}]
    explain_lines = [("{range}", "Pick a model from the list, listed from smallest to "
                                 "largest. Smaller is faster, larger understands better.")]

    while True:
        c = ui.menu("Language model", ["Model", "Params", "Size", "Needs"], items,
                    extras=extras, explain_lines=explain_lines, note=note)
        if c >= len(items):
            custom = _custom_llm()
            if custom:
                ui.ok(f"LLM: {custom['name']}")
                return custom
            continue
        ui.ok(f"LLM: {llms[c]['name']}")
        return _hub_llm(llms[c], device)


def _hub_llm(m: dict, device: str) -> dict:
    return {
        "enabled": True, "name": m["name"], "repo": m["repo"],
        "filename": m.get("filename"),
        "source": "hub", "path": None, "ctx": m.get("ctx", 4096),
        "n_gpu_layers": -1 if device == "gpu" else 0,
    }


def _show_last_selection(sel: dict) -> None:
    device = (sel.get("device") or "auto").upper()
    emb = (sel.get("embedder") or {}).get("key", "?")
    llm_cfg = sel.get("llm") or {}
    if llm_cfg.get("enabled") is False:
        llm = "no LLM"
    else:
        llm = llm_cfg.get("name") or "?"
    ui.box("LAST SELECTION", [
        f"Device   : {device}",
        f"Embedder : {emb}",
        f"LLM      : {llm}",
    ])


def run(no_llm: bool = False, auto: bool = False) -> dict:
    if SELECTION_FILE.exists() and not auto:
        try:
            existing = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing and "embedder" in existing and "llm" in existing:
            ui.section("Previous selection")
            _show_last_selection(existing)
            print()
            c = ui.menu(None, ["Option"], [
                {"cols": ["Keep this selection and continue"], "state": "rec",
                 "desc": "Skip selection and use the previous models."},
                {"cols": ["Choose again"],
                 "desc": "Show the hardware box and choose models from scratch."},
            ])
            if c == 0:
                return existing

    from ..hardware import detect_hardware
    hw = detect_hardware()
    _show_hardware(hw)

    device, cuda_tag = _ask_device(hw, auto)
    embedder = _ask_embedder(hw, auto)
    llm = _ask_llm(hw, device, no_llm, auto)

    selection = {"device": device, "cuda_tag": cuda_tag,
                 "embedder": embedder, "llm": llm}
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    ui.ok("Selection saved")
    return selection

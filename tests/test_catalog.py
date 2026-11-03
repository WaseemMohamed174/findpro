import json

from src import paths
from src.cli import select


class HW:
    safe_ram = 2.6
    safe_vram = 3.3
    gpu_available = False


def test_catalog_fields():
    cat = json.loads(paths.MODELS_CATALOG.read_text(encoding="utf-8"))
    keys = {e["key"] for e in cat["embedders"]}
    names = {m["name"] for m in cat["llms"]}
    assert cat["defaults"]["embedder"] in keys and cat["defaults"]["llm"] in names
    for e in cat["embedders"]:
        assert {"key", "repo", "dim", "size_gb", "params", "files", "ignore"} <= set(e)
    for m in cat["llms"]:
        assert {"name", "repo", "params_b", "size_gb", "ctx"} <= set(m)


def test_auto_selection_cpu_and_gpu():
    e = select._ask_embedder(HW, True)
    assert e["source"] == "hub" and e["dim"] > 0
    cpu = select._ask_llm(HW, "cpu", False, True)
    assert cpu["enabled"] and cpu["n_gpu_layers"] == 0
    gpu = select._ask_llm(HW, "gpu", False, True)
    assert gpu["n_gpu_layers"] == -1 and gpu["params_b"] if "params_b" in gpu else gpu["name"]
    assert select._ask_llm(HW, "cpu", True, True)["enabled"] is False


def test_selection_feeds_settings(tmp_path, monkeypatch):
    from src import settings
    sel = {"device": "cpu", "cuda_tag": None,
           "embedder": select._ask_embedder(HW, True),
           "llm": select._ask_llm(HW, "gpu", False, True)}
    f = tmp_path / "sel.json"
    f.write_text(json.dumps(sel))
    monkeypatch.setattr(paths, "SELECTION_FILE", f)
    settings.reload()
    try:
        assert settings.EMBEDDER_KEY == sel["embedder"]["key"]
        assert settings.EMBEDDER_QUERY_PREFIX == "query: "
        assert settings.LLM_NAME == sel["llm"]["name"]
        assert settings.LLM_REPO == sel["llm"]["repo"]
        assert not settings.LLM_ENABLED
    finally:
        monkeypatch.undo()
        settings.reload()


def test_cpu_recommendation_is_never_blocked(capsys, monkeypatch):
    answers = []
    monkeypatch.setattr(select.ui, "ask", lambda n, default=1: (
        answers.append((n, default)) or default))
    chosen = select._ask_llm(HW, "cpu", False, False)
    n, default = answers[0]
    assert 1 <= default <= n
    out = select.ui.strip(capsys.readouterr().out)
    rec_line = next(l for l in out.splitlines() if "recommended" in l)
    assert "not available" not in rec_line
    assert f"LLM: {chosen['name']}" in out


def test_blocked_rows_are_not_numbered(capsys, monkeypatch):
    monkeypatch.setattr(select.ui, "ask", lambda n, default=1: default)
    select._ask_llm(HW, "cpu", False, False)
    lines = select.ui.strip(capsys.readouterr().out).splitlines()
    blocked = [l for l in lines if "not available" in l]
    assert blocked, "expected at least one blocked model"
    for l in blocked:
        assert "✕" in l and not any(ch.isdigit() for ch in l.split("│")[1][:3])
    numbered = [l for l in lines if l.startswith("│") and l.split("│")[1].strip()[:1].isdigit()]
    nums = [int(l.split("│")[1].split()[0]) for l in numbered]
    assert nums == list(range(1, len(nums) + 1))


def test_menu_lines_never_overflow(capsys, monkeypatch):
    monkeypatch.setattr(select.ui, "ask", lambda n, default=1: default)
    select._ask_embedder(HW, False)
    select._ask_llm(HW, "cpu", False, False)
    out = select.ui.strip(capsys.readouterr().out)
    limit = select.ui.width()
    assert max(select.ui._w(l) for l in out.splitlines()) <= limit


def test_explanation_comes_before_options(capsys, monkeypatch):
    monkeypatch.setattr(select.ui, "ask", lambda n, default=1: default)
    select._ask_embedder(HW, False)
    out = select.ui.strip(capsys.readouterr().out)
    assert out.index("WHAT EACH OPTION DOES") < out.index("Enter a local model path") \
        < out.index("Embedder: ")

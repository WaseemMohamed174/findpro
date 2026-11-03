import pytest

from src import settings
from src.db import Busy
from src.runtime import EventBus
from src.text import direction, digest, fuzzy, norm, parse_json, split_list
from tests.conftest import ROWS, add_workers


def test_norm_is_language_neutral():
    assert norm("  CaFé  ") == "cafe"
    assert norm("أحمد") == norm("احمد")
    assert norm("Ｈｅｌｌｏ") == "hello"
    assert norm(None) == ""


def test_split_and_fuzzy_and_digest():
    assert split_list("a, b،c;d\ne") == ["a", "b", "c", "d", "e"]
    assert fuzzy("plumbng", ["plumbing", "painting"], 0.8) == "plumbing"
    assert fuzzy("zzz", ["plumbing"], 0.8) is None
    assert digest("x") == digest("x") != digest("y")


def test_direction_and_json():
    assert direction("hello") == "ltr"
    assert direction("مرحبا بك") == "rtl"
    assert parse_json('noise {"a": 1} tail') == {"a": 1}
    assert parse_json("nope") is None


def test_event_bus_history_and_unsubscribe():
    bus = EventBus(max_history=2)
    for i in range(3):
        bus.publish("k", "s", {"i": i})
    q, hist = bus.subscribe()
    assert [e.payload["i"] for e in hist] == [1, 2]
    bus.publish("k", "s", 5)
    assert q.get_nowait().payload == {"value": 5}
    bus.unsubscribe(q)


def test_entities_only_with_workers(env):
    store, *_ = env
    store.entity_id("city", "Nowhere", create=True)
    assert store.entities("city") == []
    add_workers(store, ROWS)
    assert {c["name"] for c in store.entities("city")} == {"Cairo", "Giza"}
    assert store.entity_id("city", "cairo") == store.entity_id("city", "CAIRO")


def test_filter_count_sort_nulls_last(seeded):
    store, _, _, ids = seeded
    cairo = store.entity_id("city", "Cairo")
    assert store.count(cairo) == 2
    assert store.count(gender="female") == 2
    assert store.count(age_min=40) == 2
    assert store.count(age_max=29) == 1
    plumbing = store.entity_id("service", "Plumbing")
    best = store.filter_ids(service_id=plumbing, sort="rating")
    assert best[0] == ids[1] and best[-1] == ids[2]
    cheap = store.filter_ids(service_id=plumbing, sort="rate_asc")
    assert cheap[0] == ids[0] and cheap[-1] == ids[2]
    assert len(store.filter_ids(limit=2)) == 2


def test_worker_roundtrip_and_crud(seeded):
    store, _, _, ids = seeded
    w = store.worker(ids[0])
    assert w["skills"] == ["pipes"] and w["city_name"] == "Cairo"
    rec = {"name": "Alex M", "city_id": w["city_id"], "service_id": w["service_id"], "skills": ["x", "y"]}
    assert store.update_worker(ids[0], rec)
    assert store.worker(ids[0])["skills"] == ["x", "y"]
    assert store.delete_worker(ids[0]) and not store.delete_worker(ids[0])
    items, total = store.search_workers("sara")
    assert total == 1 and items[0]["name"] == "Sara Lee"


def test_vectors_search_and_hash(seeded):
    store, _, emb, ids = seeded
    store.save_vectors("fake", [(ids[0], emb.embed(["plumbing cairo"])[0], "h1"),
                                (ids[3], emb.embed(["painting giza"])[0], "h2")])
    q = emb.embed(["plumbing"])[0]
    hits = store.search_vectors(q, [ids[0], ids[3]], "fake", 1)
    assert hits[0][0] == ids[0]
    assert len(store.search_vectors(q, None, "fake", 5)) == 2
    assert store.vector_hash(ids[0], "fake") == "h1"
    store.delete_other_vectors("other")
    assert store.search_vectors(q, None, "fake", 5) == []
    assert store.search_vectors(q, [], "fake", 5) == []


def test_ctx_messages_jobs(env):
    store, *_ = env
    ctx = store.load_ctx("s1")
    assert ctx["last_shown_ids"] == [] and ctx["lang"] is None
    ctx["home_city"] = "Cairo"
    store.save_ctx("s1", ctx)
    assert store.load_ctx("s1")["home_city"] == "Cairo"
    store.add_message("s1", "user", "a")
    store.add_message("s1", "assistant", "b")
    assert [m["content"] for m in store.recent_messages("s1", 10)] == ["a", "b"]
    job = store.job_begin("upload", "x", 10)
    with pytest.raises(Busy):
        store.job_begin("upload", "y", 1)
    assert store.job_update(job["job_id"], done=5)["progress"] == 50
    assert store.job_finish(job["job_id"], {"ok": True})["status"] == "done"
    assert store.job_begin("reindex", "z", 1)["status"] == "running"
    store.recover_stale_jobs()
    assert store.jobs_recent()[0]["status"] == "failed"


def test_settings_llm_filename_override(tmp_path, monkeypatch):
    import json
    from src import paths
    cat = tmp_path / "models.json"
    cat.write_text(json.dumps({"embedders": [{"key": "e", "repo": "r", "dim": 8}],
                               "llms": [{"name": "m", "repo": "rr", "filename": None}],
                               "defaults": {"embedder": "e", "llm": "m"}}))
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"llm": {"name": "m", "filename": "m.gguf", "enabled": True}}))
    monkeypatch.setattr(paths, "MODELS_CATALOG", cat)
    monkeypatch.setattr(paths, "SELECTION_FILE", sel)
    settings.reload()
    try:
        assert settings.LLM_ENABLED and settings.LLM_PATH.name == "m.gguf"
        assert settings.EMBEDDER_KEY == "e"
    finally:
        monkeypatch.undo()
        settings.reload()

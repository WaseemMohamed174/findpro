from src import ingest, search, settings
from tests.conftest import ROWS, add_workers, spec_json


def spec(**kw):
    base = {"intent": "search", "city": None, "service": None, "gender": None, "age_min": None,
            "age_max": None, "sort": None, "targets": [], "raw": "plumber in cairo"}
    base.update(kw)
    return base


def test_blend_uses_rating_when_there_is_no_semantic_signal():
    assert search.blend([1, 2, 3], {}, {1: 3.0, 2: 4.9, 3: None}) == [2, 1, 3]


def test_blend_equal_similarity_lets_rating_decide():
    assert search.blend([1, 2], {1: 0.5, 2: 0.5}, {1: 4.0, 2: 4.8}) == [2, 1]


def test_blend_clear_semantic_match_beats_a_small_rating_gap():
    assert search.blend([1, 2], {1: 0.9, 2: 0.1}, {1: 4.5, 2: 4.8}) == [1, 2]


def test_blend_does_not_let_semantics_beat_a_big_rating_gap():
    assert search.blend([1, 2], {1: 0.55, 2: 0.50}, {1: 2.0, 2: 4.9}) == [2, 1]


def test_blend_unknown_rating_is_not_favoured():
    assert search.blend([1, 2], {}, {1: None, 2: 4.0}) == [2, 1]


def test_default_search_order_is_best_rated_first_when_embedder_is_down(seeded):
    store, llm, emb, ids = seeded
    emb.up = False
    out = search.run(spec(city="Cairo", service="Plumbing"))
    assert [w["name"] for w in out["workers"]] == ["Sara Lee", "Alex Morgan"]


def test_default_search_keeps_every_candidate_when_embedder_is_up(seeded):
    out = search.run(spec(city="Cairo", service="Plumbing"))
    assert {w["name"] for w in out["workers"]} == {"Alex Morgan", "Sara Lee"}
    assert out["route"] == "semantic"


def test_ratings_lookup(seeded):
    store, llm, emb, ids = seeded
    got = store.ratings(ids)
    assert got[ids[0]] == 4.5 and got[ids[2]] is None


def _embed_everyone(store, ids):
    for worker_id in ids:
        ingest.embed_worker(worker_id)
    assert store.search_vectors([1.0] * 64, None, "fake", 10), "vectors must exist for this test"


def test_name_search_does_not_guess_below_the_threshold(seeded, monkeypatch):
    store, llm, emb, ids = seeded
    _embed_everyone(store, ids)
    monkeypatch.setattr(settings, "NAME_SIM_MIN", 0.999)
    out = search.run(spec(intent="person", targets=["Zzzzqx Qwertyuiop"]))
    assert out["workers"] == [] and out["is_empty"] is True


def test_name_search_fallback_is_gated_only_by_the_threshold(seeded, monkeypatch):
    store, llm, emb, ids = seeded
    _embed_everyone(store, ids)
    monkeypatch.setattr(settings, "NAME_SIM_MIN", -1.0)
    out = search.run(spec(intent="person", targets=["Zzzzqx Qwertyuiop"]))
    assert len(out["workers"]) == 1 and out["approx"] is True


def test_like_wildcards_in_a_name_are_literal(seeded):
    store, *_ = seeded
    assert store.find_by_name("%") == []
    assert store.find_by_name("_") == []
    assert store.find_by_name("alex")


def test_settings_validate_selection_rules():
    ok = settings.validate_selection({"device": "cpu", "llm": {"enabled": False}})
    assert ok == {"device": "cpu", "llm": {"enabled": False}}
    for bad in ({"x": 1}, {"device": "tpu"}, {"llm": {"name": "nope"}},
                {"llm": {"name": "Qwen2.5-3B-Instruct", "n_gpu_layers": True}}, [], "x"):
        try:
            settings.validate_selection(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")

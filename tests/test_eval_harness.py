"""Tests for eval/run_eval.py and its dataset.

The dataset is checked against the demo workbook, and the whole pipeline around the model is
checked with an "oracle" LLM that always answers correctly: if the search, merge and routing
code is right, the evaluation must score 100 % with it. A real model will score lower; that gap
is what eval/run_eval.py measures.
"""
import json
import re
import sys
from pathlib import Path

import openpyxl
import pytest
from fastapi.testclient import TestClient

from src import api, ingest, paths
from src.text import norm
from tests.conftest import spec_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_eval  # noqa: E402


def demo_catalog():
    ws = openpyxl.load_workbook(paths.SEED_FILE, read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    head = rows[0]
    col = {name: head.index(name) for name in ("الاسم", "المدينة", "الخدمة الرئيسية")}
    return {
        "names": {norm(r[col["الاسم"]]) for r in rows[1:]},
        "cities": {norm(r[col["المدينة"]]) for r in rows[1:]},
        "services": {norm(r[col["الخدمة الرئيسية"]]) for r in rows[1:]},
    }


def test_dataset_is_valid_and_consistent_with_the_demo_data():
    cases = run_eval.load_cases()
    assert len(cases) >= 60
    cats = {c["category"] for c in cases}
    assert {"search", "filters", "count", "person", "comparison", "followup", "chat",
            "not_in_catalog"} <= cats
    cat = demo_catalog()
    for case in cases:
        exp = case["expect"]
        if exp.get("city"):
            assert norm(exp["city"]) in cat["cities"], case["id"]
        if exp.get("service"):
            assert norm(exp["service"]) in cat["services"], case["id"]
        if case["category"] == "person" and "is_empty" in exp:
            found = any(norm(w) in n for w in exp["targets_has"] for n in cat["names"])
            assert found == (not exp["is_empty"]), f"{case['id']}: name presence disagrees with data"


@pytest.mark.parametrize("bad, msg", [
    ({"id": "x", "category": "c", "turns": ["a"]}, "missing 'expect'"),
    ({"id": "x", "category": "c", "turns": [], "expect": {"intent": "chat"}}, "non-empty list"),
    ({"id": "x", "category": "c", "turns": ["a"], "expect": {"nope": 1}}, "unknown expect keys"),
])
def test_validate_case_rejects_bad_cases(bad, msg):
    with pytest.raises(ValueError, match=msg):
        run_eval.validate_case(bad)


def test_check_compares_the_right_things():
    resp = {"event": "search", "spec": {"intent": "search", "city": "المنصورة", "service": None,
                                        "gender": "female", "sort": None, "age_min": None,
                                        "age_max": 40, "keep": True, "targets": ["رجب يسري"],
                                        "user_name": "كريم", "home_city": "المعادى"},
            "result": {"city": "المنصورة", "service": "سباكة", "ignored_city": None, "is_empty": False}}
    ok = run_eval.check({
        "intent": "search", "event": "search", "city": "المنصورة", "service": "سباكة",
        "gender": "female", "sort": None, "age_min": None, "age_max": 40, "keep": True,
        "targets_has": ["رجب"], "user_name": "كريم", "home_city": "المعادي",
        "ignored_city": False, "is_empty": False}, resp)
    assert all(v[0] for v in ok.values()), ok
    bad = run_eval.check({"city": "طنطا", "gender": "male", "age_max": 30, "keep": False,
                          "targets_has": ["أمل"], "is_empty": True, "ignored_city": True}, resp)
    assert not any(v[0] for v in bad.values()), bad
    assert run_eval.check({"city": None}, {"spec": {}, "result": {}})["city"][0] is True


def test_summarize_counts_exact_cases_fields_and_errors():
    results = [
        {"id": "a", "category": "x", "checks": {"intent": (True, "s"), "city": (True, "c")}},
        {"id": "b", "category": "x", "checks": {"intent": (True, "s"), "city": (False, "z")}},
        {"id": "c", "category": "y", "error": "boom"},
    ]
    s = run_eval.summarize(results)
    assert (s["cases"], s["exact"], s["errors"]) == (3, 1, 1)
    assert s["by_field"]["city"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert s["by_category"]["x"]["accuracy"] == 0.5 and s["by_category"]["y"]["correct"] == 0


# ------------------------------------------------------------------ oracle pipeline run

TARGETS = {
    "person-01": ["رجب يسري"], "person-02": ["أمل صفي"], "person-03": ["دلال خالد"],
    "person-04": ["بدر الدين زكي"], "person-05": ["صفوان رائد"], "person-06": ["زيزو الفضائي"],
    "person-07": ["ماكسيموس الجلاد"], "comparison-03": ["رجب يسري", "أمل صفي"],
}
# What a perfect model writes for messages whose expectations are not a plain copy.
MANUAL = {
    "عايز طيار خاص": dict(intent="search", service="طيار"),
    "محتاج دكتور بيطري في طنطا": dict(intent="search", service="بيطري", city="طنطا"),
    "سباك في القاهرة": dict(intent="search", service="سباكة", city="القاهرة"),
    "كهربائي في الإسكندرية": dict(intent="search", service="كهرباء", city="الإسكندرية"),
    "محتاج شغالين": dict(intent="search"),
    "بس الستات": dict(intent="search", gender="female", keep=True),
    "الأرخص": dict(intent="search", sort="rate_asc", keep=True),
    "طب في الدقي؟": dict(intent="search", city="الدقي", keep=True),
    "وفي فيصل؟": dict(intent="search", city="فيصل", keep=True),
    "أحسن واحد": dict(intent="search", sort="rating", keep=True),
    "عايز كهربائي": dict(intent="search", service="كهرباء"),
    "اسمي محمد وساكن في طنطا": dict(intent="chat", user_name="محمد", home_city="طنطا"),
    "قارن بين أول اتنين": dict(intent="comparison"),
    "مين فيهم أحسن؟": dict(intent="comparison"),
    "عايز نجار في أكتوبر": dict(intent="search", service="نجارة", city="أكتوبر"),
    "عايز سباك في طنطا": dict(intent="search", service="سباكة", city="طنطا"),
    "كهربائي في المنصورة": dict(intent="search", service="كهرباء", city="المنصورة"),
    "سباك في الهرم": dict(intent="search", service="سباكة", city="الهرم"),
    "عايز مدرس خصوصي": dict(intent="search", service="تدريس خصوصي"),
    "عايز سباك في المنصورة": dict(intent="search", service="سباكة", city="المنصورة"),
    "عايز كهربائي في الدقي": dict(intent="search", service="كهرباء", city="الدقي"),
}


def oracle_specs(cases):
    specs = {}
    for case in cases:
        text = case["turns"][-1]
        if text in MANUAL:
            specs[text] = MANUAL[text]
            continue
        e = case["expect"]
        specs[text] = {k: e[k] for k in ("intent", "city", "service", "gender", "sort", "age_min",
                                          "age_max", "user_name", "home_city") if e.get(k) is not None}
        specs[text]["intent"] = e.get("intent", "search")
        specs[text]["targets"] = TARGETS.get(case["id"], [])
        if e.get("keep"):
            specs[text]["keep"] = True
    for case in cases:
        for text in case["turns"][:-1]:
            assert text in MANUAL, f"oracle needs a spec for the first turn {text!r}"
            specs[text] = MANUAL[text]
    return specs


def test_the_pipeline_scores_100_percent_with_a_perfect_model(env, monkeypatch):
    store, llm, emb = env
    records, report = ingest.parse_excel(paths.SEED_FILE.read_bytes())
    assert len(records) == 1000 and report["invalid_rows"] == 0
    job = store.job_begin("seed_demo", "t", len(records))
    ingest.run_import(job["job_id"], records, True)
    assert store.count_all() == 1000

    cases = run_eval.load_cases()
    specs = oracle_specs(cases)

    def perfect_chat(messages, json_mode=False, max_tokens=None, temperature=None):
        text = messages[-1]["content"].rsplit("User message: ", 1)[-1]
        return json.dumps(spec_json(**specs[text]))

    monkeypatch.setattr(llm, "chat", perfect_chat)
    client = TestClient(api.app, client=("127.0.0.1", 50000))

    def via_test_client(base, path, payload=None, token="", timeout=0):
        resp = client.post(path, json=payload) if payload is not None else client.get(path)
        assert resp.status_code == 200, (path, resp.text)
        return resp.json()

    monkeypatch.setattr(run_eval, "_call", via_test_client)
    results = [run_eval.run_case(c, "http://test", "") for c in cases]
    wrong = [(r["id"], r.get("error") or {k: g for k, (ok, g) in r["checks"].items() if not ok})
             for r in results if "error" in r or not all(ok for ok, _ in r["checks"].values())]
    assert not wrong, wrong
    assert run_eval.summarize(results)["accuracy"] == 1.0


def test_compare_reports_finds_the_cases_where_two_models_differ():
    import compare_reports

    def report(name, rights):
        results = [{"id": i, "category": "c", "checks": {"intent": {"ok": ok, "got": "x"}}}
                   for i, ok in rights.items()]
        summary = run_eval.summarize([
            {"id": i, "category": "c", "checks": {"intent": (ok, "x")}} for i, ok in rights.items()])
        return {"selection": {"llm": {"name": name}}, "summary": summary, "results": results}

    a = report("small", {"t1": True, "t2": False, "t3": True})
    b = report("big", {"t1": True, "t2": True, "t3": False})
    out = compare_reports.compare(a, b)
    assert out["only_a_right"] == ["t3"] and out["only_b_right"] == ["t2"]
    assert out["accuracy"] == (a["summary"]["accuracy"], b["summary"]["accuracy"])
    assert out["by_field"]["intent"] == (0.6667, 0.6667)
    assert compare_reports._name(a) == "small + ?"

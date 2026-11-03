import io
import json

from openpyxl import Workbook, load_workbook

from src import chat, ingest, nlu, search
from tests.conftest import spec_json


def _spec(**kw):
    s = nlu.parse(json.dumps(spec_json(**kw)), "raw")
    return s


def test_nlu_parse_validation():
    assert nlu.parse("garbage", "x") is None
    assert nlu.parse(json.dumps({"intent": "weird"}), "x") is None
    s = nlu.parse(json.dumps(spec_json(city="null", gender="robot", sort="x", age_min=50,
                                       age_max="30", targets=["A", 3, " "], lang=None)), "t")
    assert s["city"] is None and s["gender"] is None and s["sort"] is None
    assert (s["age_min"], s["age_max"]) == (30, 50)
    assert s["targets"] == ["A"] and s["raw"] == "t"


def test_nlu_prompt_contains_catalog(seeded):
    store, llm, *_ = seeded
    llm.nlu_reply = spec_json()
    nlu.understand("hi", store.load_ctx("s"), [], store.entities("city"),
                   store.entities("service"), ["Alex"])
    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert "Cairo | Giza" in system and "Painting | Plumbing" in system
    assert "workers shown last: Alex" in user


def test_nlu_examples_use_catalog_names_and_no_person_example(seeded):
    store, llm, *_ = seeded
    system = nlu._system(store.entities("city"), store.entities("service"))
    assert "{ex_" not in system
    assert '"service":"Painting"' in system and '"city":"Cairo"' in system
    assert '"intent":"person"' not in system


def test_nlu_none_when_llm_down(seeded):
    store, llm, *_ = seeded
    llm.up = False
    assert nlu.understand("hi", {}, [], [], [], []) is None


def test_resolve_exact_fuzzy_unknown_semantic(seeded):
    assert search.resolve("city", "cairo")[1] == "Cairo"
    assert search.resolve("city", "Cairoo")[1] == "Cairo"
    assert search.resolve("city", "Atlantis") == (None, None, "Atlantis")
    assert search.resolve("service", None) == (None, None, None)
    assert search.resolve("service", "Plumbin")[1] == "Plumbing"


def test_search_filters_and_sort(seeded):
    spec = _spec(city="Cairo", service="Plumbing", gender="female")
    r = search.run(spec)
    assert [w["name"] for w in r["workers"]] == ["Sara Lee"]
    r = search.run(_spec(service="Plumbing", sort="rating"))
    assert r["route"] == "sql" and r["workers"][0]["name"] == "Sara Lee"


def test_search_fallback_and_empty_and_unknown(seeded):
    r = search.run(_spec(city="Cairo", service="Painting"))
    assert r["is_fallback"] and r["fallback_cities"] == ["Giza"]
    r = search.run(_spec(city="Giza", service="Plumbing", age_min=99))
    assert r["is_empty"] and r["route"] == "empty"
    r = search.run(_spec(city="Atlantis", service="Plumbing"))
    assert r["intent"] == "search" and r["ignored_city"] == "Atlantis" and r["count"] == 3
    r = search.run(_spec(city="Cairo", service="Atlantis"))
    assert r["intent"] == "unknown_entity" and r["unknown_service"] == "Atlantis"


def test_count(seeded):
    r = search.run(_spec(intent="count", service="Plumbing"))
    assert r["intent"] == "count" and r["count"] == 3


def test_person_and_compare(seeded):
    store, _, _, ids = seeded
    r = search.run(_spec(intent="person", targets=["sara"]))
    assert r["workers"][0]["name"] == "Sara Lee" and not r["approx"]
    r = search.run(_spec(intent="person", targets=["Sarah Lee"]))
    assert r["workers"][0]["name"] == "Sara Lee" and r["approx"]
    r = search.run(_spec(intent="comparison", targets=["alex", "omar"]), ids)
    assert {w["name"] for w in r["workers"]} == {"Alex Morgan", "Omar Ali"}
    r = search.run(_spec(intent="comparison"), ids)
    assert len(r["workers"]) == 3
    r = search.run(_spec(intent="comparison"), [])
    assert r["is_empty"]


def run_chat(text, sid="s1"):
    return chat.handle(sid, text)


def test_chat_intent_and_language(seeded):
    _, llm, *_ = seeded
    llm.nlu_reply = spec_json(intent="chat", lang="ar")
    out = run_chat("مرحبا")
    assert out["intent"] == "chat" and out["workers"] == []
    assert out["lang"] == "ar" and out["dir"] == "rtl" and out["text"].startswith("LLM")


def test_chat_search_flow_and_context(seeded):
    store, llm, *_ = seeded
    llm.nlu_reply = spec_json(city="Cairo", service="Plumbing")
    out = run_chat("plumber in cairo")
    assert out["intent"] == "search" and len(out["workers"]) == 2
    facts = llm.calls[-1][0]["content"]
    assert "Facts:" in facts and "Alex Morgan" in facts and "name_norm" not in facts
    ctx = store.load_ctx("s1")
    assert ctx["last_filters"]["city"] == "Cairo" and len(ctx["last_shown_ids"]) == 2
    llm.nlu_reply = spec_json(gender="female", keep=True)
    out = run_chat("only women")
    assert [w["name"] for w in out["workers"]] == ["Sara Lee"]
    llm.nlu_reply = spec_json(intent="comparison")
    out = run_chat("compare them")
    assert out["intent"] == "comparison"


def test_chat_clarify_unknown_count_person(seeded):
    _, llm, *_ = seeded
    llm.nlu_reply = spec_json()
    assert run_chat("find someone")["intent"] == "clarify"
    llm.nlu_reply = spec_json(city="Atlantis")
    assert run_chat("anyone in atlantis")["intent"] == "clarify"  # bogus city ignored, nothing left
    llm.nlu_reply = spec_json(city="Atlantis", service="Plumbing")
    out = run_chat("plumber in atlantis")
    assert out["intent"] == "search" and len(out["workers"]) == 3
    assert '"ignored_city": "Atlantis"' in llm.calls[-1][0]["content"]
    llm.nlu_reply = spec_json(service="Atlantis")
    out = run_chat("atlantis service")
    assert out["intent"] == "unknown_entity"
    assert "available_services" in llm.calls[-1][0]["content"]
    llm.nlu_reply = spec_json(intent="count", service="Plumbing")
    assert run_chat("how many")["intent"] == "count"
    llm.nlu_reply = spec_json(intent="person", targets=["Mona"])
    assert run_chat("who is mona")["workers"][0]["name"] == "Mona Said"


def test_chat_home_city_default(seeded):
    store, llm, *_ = seeded
    llm.nlu_reply = spec_json(intent="chat", home_city="giza")
    run_chat("I live in giza")
    assert store.load_ctx("s1")["home_city"] == "Giza"
    llm.nlu_reply = spec_json(service="Plumbing")
    out = run_chat("plumber")
    assert {w["city_name"] for w in out["workers"]} == {"Giza"}


def test_no_canned_text_when_llm_down(seeded):
    _, llm, *_ = seeded
    llm.up = False
    out = run_chat("anything")
    assert out["intent"] == "error" and "not ready" in out["text"]
    assert not hasattr(chat, "template")


def test_nlu_failure_retries_then_asks_llm_to_rephrase(seeded):
    _, llm, *_ = seeded
    llm.json_replies = [{"intent": "bogus"}, {"intent": "bogus"}]
    out = run_chat("painting")
    assert out["intent"] == "unclear" and out["workers"] == []
    assert out["stages"][0]["label"] == "failed"
    assert len([c for c in llm.calls if c[0]["content"].startswith("You turn")]) == 2


def test_nlu_retry_recovers(seeded):
    _, llm, *_ = seeded
    llm.json_replies = [{"intent": "bogus"}, spec_json(service="Painting")]
    out = run_chat("painting")
    assert out["intent"] == "search" and [w["name"] for w in out["workers"]] == ["Mona Said"]


def test_nlu_history_has_only_user_messages(seeded):
    store, llm, *_ = seeded
    llm.nlu_reply = spec_json(intent="chat")
    run_chat("first")
    run_chat("second")
    nlu_calls = [c for c in llm.calls if c[0]["content"].startswith("You turn")]
    assert "assistant:" not in nlu_calls[-1][1]["content"]


def test_explain_returns_raw_nlu(seeded):
    _, llm, *_ = seeded
    llm.nlu_reply = spec_json(service="Painting")
    out = chat.explain("painters")
    assert out["spec"]["service"] == "Painting"
    assert out["nlu"]["attempts"][0]["valid"] is True and "raw" in out["nlu"]["attempts"][0]
    llm.up = False
    assert "error" in chat.explain("x")


def test_chat_stream_events(seeded):
    _, llm, *_ = seeded
    llm.nlu_reply = spec_json(intent="chat")
    kinds = [e["type"] for e in chat.handle_stream("s9", "hello")]
    assert kinds[0] == "stage" and "token" in kinds and kinds[-1] == "done"


def test_chat_error_path(seeded, monkeypatch):
    monkeypatch.setattr(chat, "_plan", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    out = run_chat("x")
    assert out["intent"] == "error" and "boom" in out["text"]


def make_xlsx(rows):
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_template_roundtrip():
    records, report = ingest.parse_excel(ingest.template_bytes())
    assert len(records) == 2 and records[0]["gender"] == "male" and records[0]["skills"][0] == "doors"
    assert report["invalid_rows"] == 0


def test_parse_excel_llm_mapping_foreign_headers(env):
    store, llm, _ = env
    header = ["الاسم", "المدينة", "الخدمة", "النوع", "التقييم"]
    data = make_xlsx([header, ["أحمد", "القاهرة", "سباكة", "ذكر", 4.5], ["x", "a", "b", "ذكر", 1],
                      ["سارة", "الجيزة", "دهان", "أنثى", 9]])
    llm.json_replies = [{"name": 0, "city": 1, "service": 2, "gender": 3, "rating": 4},
                        {"ذكر": "male", "أنثى": "female"}]
    records, report = ingest.parse_excel(data)
    assert len(records) == 2 and report["invalid_rows"] == 1
    assert records[0]["gender"] == "male" and records[1]["gender"] == "female"
    assert records[1]["rating"] == 5.0


def test_parse_excel_errors(env):
    store, llm, _ = env
    llm.up = False
    try:
        ingest.parse_excel(b"not excel")
        assert False
    except ValueError as e:
        assert "valid" in str(e)
    try:
        ingest.parse_excel(make_xlsx([["foo", "bar"], [1, 2]]))
        assert False
    except ValueError as e:
        assert "Missing columns" in str(e)


def test_import_dedupe_and_embeddings(env):
    store, llm, emb = env
    records, _ = ingest.parse_excel(ingest.template_bytes())
    job = store.job_begin("upload", "t", 2)
    ingest.run_import(job["job_id"], records, True)
    assert store.job(job["job_id"])["status"] == "done"
    assert store.count_all() == 2
    assert len(store.search_vectors(emb.embed(["x"])[0], None, "fake", 10)) == 2
    job2 = store.job_begin("upload", "t", 2)
    ingest.run_import(job2["job_id"], records, True)
    assert json.loads(store.job(job2["job_id"])["result_json"])["duplicates"] == 2
    assert store.count_all() == 2


def test_import_warns_when_embedder_down(env):
    store, llm, emb = env
    emb.up = False
    records, _ = ingest.parse_excel(ingest.template_bytes())
    job = store.job_begin("upload", "t", 2)
    ingest.run_import(job["job_id"], records, True)
    assert "embedder" in json.loads(store.job(job["job_id"])["result_json"])["warning"]


def test_reindex_and_embed_worker(seeded):
    store, llm, emb, ids = seeded
    job = store.job_begin("reindex", "r", 1)
    ingest.run_reindex(job["job_id"])
    assert store.job(job["job_id"])["status"] == "done"
    assert store.vector_hash(ids[0], "fake")
    store.entity_id("city", "Z", create=True)
    before = store.vector_hash(ids[1], "fake")
    ingest.embed_worker(ids[1])
    assert store.vector_hash(ids[1], "fake") == before


def test_seed_missing_file_fails_job(env, monkeypatch):
    from src import paths
    store, *_ = env
    monkeypatch.setattr(paths, "SEED_FILE", paths.BASE_DIR / "nope.xlsx")
    job = store.job_begin("seed_demo", "s", 1)
    ingest.run_seed(job["job_id"])
    assert store.job(job["job_id"])["status"] == "failed"


def test_real_seed_file_with_fake_llm_mapping(env):
    from src import paths
    store, llm, _ = env
    if not paths.SEED_FILE.exists():
        return
    llm.json_replies = [{"name": 0, "age": 1, "gender": 2, "nationality": 3, "city": 4, "service": 5,
                         "experience_years": 6, "rating": 7, "hourly_rate": 8, "phone": 9,
                         "completed_projects": 10, "skills": 11},
                        {"ذكر": "male", "أنثى": "female"}]
    records, _ = ingest.parse_excel(paths.SEED_FILE.read_bytes())
    assert len(records) > 50 and records[0]["skills"]


def test_import_commits_before_embedding_and_survives_concurrent_writer(env):
    import threading
    from src import ingest, db
    store, llm, emb = env
    seen = {}
    real = emb.embed

    def embed_probe(texts, kind="passage"):
        # another connection must be able to write while we "wait on the embedder"
        def other():
            c = db.sqlite3.connect(str(db.paths.DB_PATH), timeout=0.2)
            try:
                c.execute("INSERT INTO chat_sessions(session_id, context_json, created_at, last_active) "
                          "VALUES(?,'{}','t','t')", (f"probe{len(seen)}-{id(c)}",))
                c.commit()
                seen[f"ok{len(seen)}"] = True
            except Exception as e:
                seen["err"] = str(e)
            finally:
                c.close()
        t = threading.Thread(target=other)
        t.start(); t.join()
        return real(texts, kind)

    emb.embed = embed_probe
    records = [{"name": f"Worker {i}", "city": "Cairo", "service": "Plumbing", "skills": []}
               for i in range(120)]
    added, dup, warn = ingest._import(job_id=store.job_begin("t", "t", 1)["job_id"],
                                      records=records, dedupe=True)
    assert added == 120 and warn is None
    assert "err" not in seen and len(seen) >= 3

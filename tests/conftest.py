import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import chat, db, ingest, nlu, paths, search, settings


class FakeEmbedder:
    def __init__(self):
        self.up = True

    def ready(self):
        return self.up

    def embed(self, texts, kind="passage"):
        out = []
        for t in texts:
            v = [0.0] * 64
            s = str(t).lower()
            for i in range(len(s) - 2):
                v[hash(s[i:i + 3]) % 64] += 1.0
            out.append(v)
        return out


class FakeLLM:
    def __init__(self):
        self.up = True
        self.nlu_reply = {}
        self.json_replies = []
        self.stream_text = "LLM reply"
        self.calls = []

    def ready(self):
        return self.up

    def chat(self, messages, json_mode=False, max_tokens=None, temperature=None):
        self.calls.append(messages)
        if self.json_replies:
            return json.dumps(self.json_replies.pop(0))
        return json.dumps(self.nlu_reply)

    def stream(self, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        for part in self.stream_text.split(" "):
            yield part + " "


def spec_json(**kw):
    base = {"intent": "search", "city": None, "service": None, "gender": None,
            "age_min": None, "age_max": None, "sort": None, "targets": [],
            "user_name": None, "home_city": None, "keep": False, "lang": "en"}
    base.update(kw)
    return base


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DB_PATH", tmp_path / "t.db")
    db._local.__dict__.pop("conn", None)
    db.store.init_schema()
    llm, emb = FakeLLM(), FakeEmbedder()
    for mod in (chat, nlu, ingest):
        monkeypatch.setattr(mod, "llm", llm, raising=False)
    for mod in (chat, search, ingest):
        monkeypatch.setattr(mod, "embedder", emb, raising=False)
    monkeypatch.setattr(settings, "EMBEDDER_KEY", "fake")
    yield db.store, llm, emb
    db._local.__dict__.pop("conn", None)


def add_workers(store, rows):
    ids = []
    for r in rows:
        cid = store.entity_id("city", r["city"], create=True)
        sid = store.entity_id("service", r["service"], create=True)
        rec = {k: v for k, v in r.items() if k not in ("city", "service")}
        ids.append(store.insert_worker({**rec, "city_id": cid, "service_id": sid}))
    return ids


ROWS = [
    {"name": "Alex Morgan", "gender": "male", "age": 35, "city": "Cairo", "service": "Plumbing", "rating": 4.5, "hourly_rate": 80, "experience_years": 10, "skills": ["pipes"]},
    {"name": "Sara Lee", "gender": "female", "age": 29, "city": "Cairo", "service": "Plumbing", "rating": 4.9, "hourly_rate": 120, "experience_years": 5},
    {"name": "Omar Ali", "gender": "male", "age": 50, "city": "Giza", "service": "Plumbing", "rating": None, "hourly_rate": None},
    {"name": "Mona Said", "gender": "female", "age": 41, "city": "Giza", "service": "Painting", "rating": 4.0, "hourly_rate": 60},
]


@pytest.fixture
def seeded(env):
    store, llm, emb = env
    ids = add_workers(store, ROWS)
    return store, llm, emb, ids

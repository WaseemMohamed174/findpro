import json
import sqlite3
import threading

import numpy as np

from . import paths
from .text import norm, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS cities (
    city_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    name_norm TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    name_norm  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL,
    name_norm          TEXT NOT NULL,
    age                INTEGER,
    gender             TEXT,
    nationality        TEXT,
    city_id            INTEGER REFERENCES cities(city_id) ON DELETE SET NULL,
    service_id         INTEGER REFERENCES services(service_id) ON DELETE SET NULL,
    experience_years   INTEGER,
    rating             REAL,
    hourly_rate        INTEGER,
    phone              TEXT,
    completed_projects INTEGER,
    skills             TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workers_city    ON workers(city_id);
CREATE INDEX IF NOT EXISTS idx_workers_service ON workers(service_id);
CREATE INDEX IF NOT EXISTS idx_workers_name    ON workers(name_norm);

CREATE TABLE IF NOT EXISTS worker_vectors (
    worker_id INTEGER PRIMARY KEY REFERENCES workers(worker_id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    text_hash TEXT
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id   TEXT PRIMARY KEY,
    context_json TEXT,
    created_at   TEXT NOT NULL,
    last_active  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    intent     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    label       TEXT NOT NULL,
    total       INTEGER DEFAULT 0,
    done        INTEGER DEFAULT 0,
    result_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

ENTITIES = {"city": ("cities", "city_id"), "service": ("services", "service_id")}

WORKER_COLS = (
    "age", "gender", "nationality", "city_id", "service_id",
    "experience_years", "rating", "hourly_rate", "phone", "completed_projects",
)

SORTS = {
    "rating": ("w.rating", "DESC"),
    "rate_asc": ("w.hourly_rate", "ASC"),
    "rate_desc": ("w.hourly_rate", "DESC"),
    "experience": ("w.experience_years", "DESC"),
}

WORKER_SELECT = (
    "SELECT w.*, c.name AS city_name, s.name AS service_name FROM workers w "
    "LEFT JOIN cities c ON w.city_id = c.city_id "
    "LEFT JOIN services s ON w.service_id = s.service_id "
)

CHUNK = 500

_local = threading.local()
_job_lock = threading.Lock()


class Busy(Exception):
    pass


def conn():
    c = getattr(_local, "conn", None)
    if c is None:
        paths.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(paths.DB_PATH), timeout=30.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


def _chunks(items):
    for i in range(0, len(items), CHUNK):
        yield items[i:i + CHUNK]


def _where(city_id, service_id, gender, age_min, age_max):
    where, params = [], []
    if city_id is not None:
        where.append("w.city_id = ?")
        params.append(city_id)
    if service_id is not None:
        where.append("w.service_id = ?")
        params.append(service_id)
    if gender:
        where.append("w.gender = ?")
        params.append(gender)
    if age_min is not None:
        where.append("w.age >= ?")
        params.append(age_min)
    if age_max is not None:
        where.append("w.age <= ?")
        params.append(age_max)
    return ("WHERE " + " AND ".join(where)) if where else "", params


def _like_escape(text):
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _worker(row):
    d = dict(row)
    try:
        d["skills"] = json.loads(d.get("skills") or "[]")
    except ValueError:
        d["skills"] = []
    return d


def _job(row):
    d = dict(row)
    d["result_json"] = d.get("result_json") or ""
    d["error"] = d.get("error") or ""
    d["finished_at"] = d.get("finished_at") or ""
    total = d.get("total") or 0
    d["progress"] = min(100, int(100 * (d.get("done") or 0) / total)) if total > 0 else 0
    return d


def default_ctx():
    return {
        "user_name": None,
        "home_city": None,
        "lang": None,
        "last_filters": {},
        "last_shown_ids": [],
    }


class Store:
    def init_schema(self):
        c = conn()
        c.executescript(SCHEMA)
        c.commit()

    def entity_id(self, kind, name, create=False):
        table, id_col = ENTITIES[kind]
        n = norm(name)
        if not n:
            return None
        c = conn()
        row = c.execute(f"SELECT {id_col} FROM {table} WHERE name_norm=?", (n,)).fetchone()
        if row:
            return row[0]
        if not create:
            return None
        cur = c.execute(
            f"INSERT INTO {table}(name, name_norm) VALUES(?,?)", (str(name).strip(), n)
        )
        c.commit()
        return cur.lastrowid

    def entities(self, kind):
        table, id_col = ENTITIES[kind]
        rows = conn().execute(
            f"SELECT t.{id_col} AS id, t.name FROM {table} t "
            f"WHERE EXISTS (SELECT 1 FROM workers w WHERE w.{id_col} = t.{id_col}) "
            f"ORDER BY t.name"
        ).fetchall()
        return [{"id": r["id"], "name": r["name"]} for r in rows]

    def workers(self, ids):
        ids = list(ids)
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = conn().execute(WORKER_SELECT + f"WHERE w.worker_id IN ({ph})", ids).fetchall()
        by_id = {r["worker_id"]: _worker(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def ratings(self, ids):
        out = {}
        for part in _chunks(list(ids)):
            ph = ",".join("?" * len(part))
            rows = conn().execute(
                f"SELECT worker_id, rating FROM workers WHERE worker_id IN ({ph})", part
            ).fetchall()
            out.update({r["worker_id"]: r["rating"] for r in rows})
        return out

    def worker(self, worker_id):
        found = self.workers([worker_id])
        return found[0] if found else None

    def find_by_name(self, name_norm, limit=20):
        if not name_norm:
            return []
        rows = conn().execute(
            "SELECT worker_id FROM workers WHERE name_norm LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{_like_escape(name_norm)}%", limit),
        ).fetchall()
        return [r["worker_id"] for r in rows]

    def all_names(self):
        rows = conn().execute("SELECT worker_id, name_norm FROM workers").fetchall()
        return {r["worker_id"]: r["name_norm"] for r in rows}

    def worker_keys(self):
        rows = conn().execute("SELECT name_norm, city_id, phone FROM workers").fetchall()
        return {(r["name_norm"], r["city_id"], r["phone"]) for r in rows}

    def filter_ids(self, city_id=None, service_id=None, gender=None,
                   age_min=None, age_max=None, sort=None, limit=None):
        where, params = _where(city_id, service_id, gender, age_min, age_max)
        order = ""
        if sort in SORTS:
            col, direction = SORTS[sort]
            order = f"ORDER BY {col} IS NULL, {col} {direction}"
        sql = f"SELECT w.worker_id FROM workers w {where} {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [r["worker_id"] for r in conn().execute(sql, params).fetchall()]

    def count(self, city_id=None, service_id=None, gender=None, age_min=None, age_max=None):
        where, params = _where(city_id, service_id, gender, age_min, age_max)
        row = conn().execute(f"SELECT COUNT(*) AS n FROM workers w {where}", params).fetchone()
        return int(row["n"])

    def count_all(self):
        return self.count()

    def commit(self):
        conn().commit()

    def insert_worker(self, rec, commit=True):
        c = conn()
        cols = ", ".join(WORKER_COLS)
        marks = ",".join("?" * len(WORKER_COLS))
        cur = c.execute(
            f"INSERT INTO workers(name, name_norm, {cols}, skills, created_at) "
            f"VALUES(?,?,{marks},?,?)",
            (rec["name"], norm(rec["name"]), *[rec.get(k) for k in WORKER_COLS],
             json.dumps(rec.get("skills") or [], ensure_ascii=False), now_iso()),
        )
        if commit:
            c.commit()
        return cur.lastrowid

    def update_worker(self, worker_id, rec):
        c = conn()
        sets = ", ".join(f"{k}=?" for k in WORKER_COLS)
        cur = c.execute(
            f"UPDATE workers SET name=?, name_norm=?, {sets}, skills=? WHERE worker_id=?",
            (rec["name"], norm(rec["name"]), *[rec.get(k) for k in WORKER_COLS],
             json.dumps(rec.get("skills") or [], ensure_ascii=False), worker_id),
        )
        c.commit()
        return cur.rowcount > 0

    def delete_worker(self, worker_id):
        c = conn()
        cur = c.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))
        c.commit()
        return cur.rowcount > 0

    def clear_workers(self):
        c = conn()
        cur = c.execute("DELETE FROM workers")
        c.commit()
        return cur.rowcount

    def search_workers(self, q="", city_id=None, service_id=None, limit=100, offset=0):
        where, params = [], []
        if q:
            where.append("(w.name_norm LIKE ? OR w.phone LIKE ?)")
            params += [f"%{norm(q)}%", f"%{q}%"]
        if city_id is not None:
            where.append("w.city_id=?")
            params.append(city_id)
        if service_id is not None:
            where.append("w.service_id=?")
            params.append(service_id)
        cond = ("WHERE " + " AND ".join(where)) if where else ""
        c = conn()
        total = c.execute(f"SELECT COUNT(*) AS n FROM workers w {cond}", params).fetchone()["n"]
        ids = [r["worker_id"] for r in c.execute(
            f"SELECT w.worker_id FROM workers w {cond} ORDER BY w.worker_id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()]
        by_id = {w["worker_id"]: w for w in self.workers(ids)}
        return [by_id[i] for i in ids if i in by_id], total

    def meta(self):
        c = conn()
        out = {}
        for key, (table, id_col) in (("cities", ENTITIES["city"]), ("services", ENTITIES["service"])):
            rows = c.execute(
                f"SELECT t.{id_col} AS id, t.name, COUNT(w.worker_id) AS workers "
                f"FROM {table} t LEFT JOIN workers w ON w.{id_col} = t.{id_col} "
                f"GROUP BY t.{id_col} ORDER BY t.name"
            ).fetchall()
            out[key] = [
                {"id": r["id"], "name": r["name"], "aliases": [], "workers": r["workers"]}
                for r in rows
            ]
        out["workers_count"] = self.count_all()
        return out

    def save_vectors(self, model_key, items):
        c = conn()
        for worker_id, vec, text_hash in items:
            arr = np.asarray(vec, dtype=np.float32)
            c.execute(
                "INSERT OR REPLACE INTO worker_vectors(worker_id, model_key, dim, vec, text_hash) "
                "VALUES(?,?,?,?,?)",
                (worker_id, model_key, int(arr.shape[0]), arr.tobytes(), text_hash),
            )
        c.commit()

    def search_vectors(self, query_vec, ids, model_key, k):
        c = conn()
        rows = []
        if ids is None:
            rows = c.execute(
                "SELECT worker_id, vec FROM worker_vectors WHERE model_key=?", (model_key,)
            ).fetchall()
        else:
            for part in _chunks(list(ids)):
                ph = ",".join("?" * len(part))
                rows += c.execute(
                    f"SELECT worker_id, vec FROM worker_vectors "
                    f"WHERE model_key=? AND worker_id IN ({ph})",
                    [model_key, *part],
                ).fetchall()
        if not rows:
            return []
        found = [r["worker_id"] for r in rows]
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = (mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)) @ q
        top = np.argsort(-sims)[:k]
        return [(found[i], float(sims[i])) for i in top]

    def vector_hash(self, worker_id, model_key):
        row = conn().execute(
            "SELECT text_hash FROM worker_vectors WHERE worker_id=? AND model_key=?",
            (worker_id, model_key),
        ).fetchone()
        return row["text_hash"] if row else None

    def delete_other_vectors(self, model_key):
        c = conn()
        c.execute("DELETE FROM worker_vectors WHERE model_key != ?", (model_key,))
        c.commit()

    def load_ctx(self, session_id):
        c = conn()
        ts = now_iso()
        c.execute(
            "INSERT INTO chat_sessions(session_id, context_json, created_at, last_active) "
            "VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET last_active=excluded.last_active",
            (session_id, "{}", ts, ts),
        )
        c.commit()
        row = c.execute(
            "SELECT context_json FROM chat_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        try:
            data = json.loads(row["context_json"] or "{}")
        except ValueError:
            data = {}
        return {**default_ctx(), **data}

    def save_ctx(self, session_id, ctx):
        c = conn()
        c.execute(
            "UPDATE chat_sessions SET context_json=?, last_active=? WHERE session_id=?",
            (json.dumps(ctx, ensure_ascii=False), now_iso(), session_id),
        )
        c.commit()

    def add_message(self, session_id, role, content, intent=None):
        c = conn()
        cur = c.execute(
            "INSERT INTO chat_messages(session_id, role, content, intent, created_at) "
            "VALUES(?,?,?,?,?)",
            (session_id, role, content, intent, now_iso()),
        )
        c.commit()
        return cur.lastrowid

    def recent_messages(self, session_id, limit):
        rows = conn().execute(
            "SELECT role, content FROM chat_messages WHERE session_id=? "
            "ORDER BY message_id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def history(self, session_id, limit=100):
        rows = conn().execute(
            "SELECT role, content, intent FROM chat_messages WHERE session_id=? "
            "ORDER BY message_id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "intent": r["intent"]}
                for r in reversed(rows)]

    def job_begin(self, kind, label, total=0):
        with _job_lock:
            c = conn()
            if c.execute(
                "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1"
            ).fetchone():
                raise Busy("A job is already running")
            cur = c.execute(
                "INSERT INTO jobs(kind, status, label, total, done, created_at) VALUES(?,?,?,?,?,?)",
                (kind, "running", label, total, 0, now_iso()),
            )
            c.commit()
        return self.job(cur.lastrowid)

    def job_update(self, job_id, done=None, total=None, label=None):
        c = conn()
        if done is not None:
            c.execute("UPDATE jobs SET done=? WHERE job_id=?", (done, job_id))
        if total is not None:
            c.execute("UPDATE jobs SET total=? WHERE job_id=?", (total, job_id))
        if label:
            c.execute("UPDATE jobs SET label=? WHERE job_id=?", (label, job_id))
        c.commit()
        return self.job(job_id)

    def job_finish(self, job_id, result):
        c = conn()
        c.execute(
            "UPDATE jobs SET status='done', result_json=?, finished_at=? WHERE job_id=?",
            (json.dumps(result, ensure_ascii=False), now_iso(), job_id),
        )
        c.commit()
        return self.job(job_id)

    def job_fail(self, job_id, error):
        c = conn()
        c.execute(
            "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE job_id=?",
            (error, now_iso(), job_id),
        )
        c.commit()
        return self.job(job_id)

    def job(self, job_id):
        row = conn().execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _job(row) if row else None

    def jobs_recent(self, limit=50):
        rows = conn().execute(
            "SELECT * FROM jobs ORDER BY job_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_job(r) for r in rows]

    def recover_stale_jobs(self):
        c = conn()
        c.execute(
            "UPDATE jobs SET status='failed', error='interrupted by restart' "
            "WHERE status IN ('queued','running')"
        )
        c.commit()


store = Store()

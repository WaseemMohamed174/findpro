import io
import re
import threading

from . import paths, settings
from .clients import LLMError, embedder, llm
from .db import store
from .runtime import bus
from .text import digest, norm, parse_json, split_list

FIELDS = (
    "name", "age", "gender", "nationality", "city", "service",
    "experience_years", "rating", "hourly_rate", "phone",
    "completed_projects", "skills",
)
REQUIRED = ("name", "city", "service")
GENDERS = ("male", "female")
BATCH = 32

FIELD_HELP = (
    "name, age, gender, nationality, city, service (the main profession or service offered), "
    "experience_years, rating, hourly_rate, phone, completed_projects, "
    "skills (a delimited list of skills)"
)


# Common header names (Arabic and English). They are tried first so that ordinary files, and
# the demo workbook, import instantly and without a model; the LLM only handles what is left.
HEADER_WORDS = {
    "name": {"name", "full_name", "الاسم", "اسم", "اسم_العامل", "الاسم_بالكامل"},
    "age": {"age", "العمر", "السن"},
    "gender": {"gender", "sex", "النوع", "الجنس"},
    "nationality": {"nationality", "الجنسية"},
    "city": {"city", "town", "المدينة", "المدينه", "مدينة", "المنطقة"},
    "service": {"service", "profession", "job", "الخدمة", "الخدمة_الرئيسية", "المهنة",
                "الحرفة", "التخصص"},
    "experience_years": {"experience_years", "experience", "years_of_experience",
                         "سنوات_الخبرة", "الخبرة"},
    "rating": {"rating", "التقييم"},
    "hourly_rate": {"hourly_rate", "price", "السعر", "سعر_الساعة"},
    "phone": {"phone", "mobile", "tel", "رقم_الهاتف", "الهاتف", "رقم_التليفون", "التليفون",
              "الموبايل", "رقم_الموبايل"},
    "completed_projects": {"completed_projects", "projects", "المشاريع_المكتملة", "المشاريع"},
    "skills": {"skills", "المهارات"},
}
GENDER_WORDS = {
    "male": {"male", "m", "man", "ذكر", "رجل", "راجل", "ولد"},
    "female": {"female", "f", "woman", "أنثى", "انثى", "امرأة", "امراة", "ست", "بنت", "سيدة"},
}
# norm() folds hamza forms (أ/إ -> ا, ئ -> ي), so compare against normalized words.
HEADER_WORDS = {f: {norm(w).replace(" ", "_") for w in ws} for f, ws in HEADER_WORDS.items()}
GENDER_WORDS = {g: {norm(w) for w in ws} for g, ws in GENDER_WORDS.items()}
_HEADER_NOISE = re.compile(r"\(.*?\)|/.*")


def _header_key(header):
    """'السعر/الساعة (جنيه)' -> 'السعر', 'التقييم (من 5)' -> 'التقييم', 'Full Name' -> 'full_name'."""
    return _HEADER_NOISE.sub("", norm(header)).strip().replace(" ", "_")


def _llm_json(prompt, max_tokens):
    if not llm.ready():
        return None
    try:
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True, max_tokens=max_tokens, temperature=0.0,
        )
    except LLMError:
        return None
    return parse_json(raw)


def _map_columns(header, rows):
    mapping = {}
    for i, h in enumerate(header):
        key = norm(h).replace(" ", "_")
        if key in FIELDS and key not in mapping:
            mapping[key] = i
    used = set(mapping.values())
    for i, h in enumerate(header):
        key = _header_key(h)
        for field, words in HEADER_WORDS.items():
            if field not in mapping and i not in used and key in words:
                mapping[field] = i
                used.add(i)
    if all(f in mapping for f in REQUIRED):
        return mapping
    cols = "\n".join(f"{i}: {h}" for i, h in enumerate(header))
    sample = "\n".join(str(list(r)) for r in rows[:3])
    data = _llm_json(
        f"A spreadsheet describes workers. Map its columns to these fields: {FIELD_HELP}.\n"
        f"Columns (index: header):\n{cols}\n\nFirst rows:\n{sample}\n\n"
        "Return ONLY a JSON object mapping each field name to the column index, "
        "or null when no column fits.",
        400,
    )
    if isinstance(data, dict):
        used = set(mapping.values())
        for field in FIELDS:
            idx = data.get(field)
            if (field not in mapping and isinstance(idx, int) and not isinstance(idx, bool)
                    and 0 <= idx < len(header) and idx not in used):
                mapping[field] = idx
                used.add(idx)
    return mapping


def _gender_map(values):
    mapping = {}
    for v in values:
        word = norm(v)
        for gender, words in GENDER_WORDS.items():
            if word in words:
                mapping[v] = gender
    rest = [v for v in values if v not in mapping]
    if rest:
        data = _llm_json(
            "Classify each value as the gender it denotes: \"male\", \"female\" or null.\n"
            f"Values: {rest}\nReturn ONLY a JSON object mapping each value to its class.",
            300,
        )
        if isinstance(data, dict):
            for v in rest:
                if data.get(v) in GENDERS:
                    mapping[v] = data[v]
    return mapping


def _text(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _int(v):
    try:
        return int(float(str(v).strip())) if _text(v) else None
    except ValueError:
        return None


def _rating(v):
    try:
        return max(0.0, min(5.0, round(float(str(v).strip()), 1))) if _text(v) else None
    except ValueError:
        return None


def _phone(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().replace(" ", "") or None


def parse_excel(content):
    from openpyxl import load_workbook

    try:
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception:
        raise ValueError("The file is not a valid .xlsx workbook")
    rows = [r for r in wb.active.iter_rows(values_only=True)]
    wb.close()
    if not rows:
        raise ValueError("The file is empty")
    header, body = rows[0], [r for r in rows[1:] if any(_text(c) for c in r)]
    mapping = _map_columns(header, body)
    missing = [f for f in REQUIRED if f not in mapping]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    def cell(row, field):
        i = mapping.get(field)
        return row[i] if i is not None and i < len(row) else None

    genders = {}
    if "gender" in mapping:
        distinct = sorted({_text(cell(r, "gender")) for r in body if _text(cell(r, "gender"))})
        genders = _gender_map(distinct[:20])

    records, invalid = [], 0
    for row in body:
        name, city, service = _text(cell(row, "name")), _text(cell(row, "city")), _text(cell(row, "service"))
        if not name or len(name) < 2 or not city or not service:
            invalid += 1
            continue
        records.append({
            "name": name,
            "city": city,
            "service": service,
            "age": _int(cell(row, "age")),
            "gender": genders.get(_text(cell(row, "gender"))),
            "nationality": _text(cell(row, "nationality")),
            "experience_years": _int(cell(row, "experience_years")),
            "rating": _rating(cell(row, "rating")),
            "hourly_rate": _int(cell(row, "hourly_rate")),
            "phone": _phone(cell(row, "phone")),
            "completed_projects": _int(cell(row, "completed_projects")),
            "skills": split_list(_text(cell(row, "skills")) or ""),
        })
    if not records:
        raise ValueError("No valid rows found")
    return records, {"invalid_rows": invalid}


def template_bytes():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "workers"
    ws.append(list(FIELDS))
    ws.append(["Alex Morgan", 35, "male", "Canadian", "Toronto", "Carpentry", 12, 4.7, 80,
               "5551230001", 140, "doors, furniture, flooring"])
    ws.append(["Sara Lee", 29, "female", "Korean", "Seoul", "Painting", 6, 4.8, 75,
               "5551230002", 60, "interior walls, murals"])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1D6B5C")
        c.alignment = Alignment(horizontal="center")
    for row in ws.iter_rows(min_row=2, min_col=10, max_col=10):
        for c in row:
            c.number_format = "@"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def worker_text(w):
    parts = [w.get("name"), w.get("city_name"), w.get("service_name")]
    if w.get("skills"):
        parts.append(", ".join(w["skills"]))
    return " · ".join(p for p in parts if p)


def _tick(job_id, done=None, total=None):
    job = store.job_update(job_id, done=done, total=total)
    bus.publish("job_progress", f"job:{job_id}", job)


def _finish(job_id, result):
    bus.publish("job_progress", f"job:{job_id}", store.job_finish(job_id, result))


def _fail(job_id, error):
    bus.publish("job_progress", f"job:{job_id}", store.job_fail(job_id, error))


def _embed_ids(ids, job_id=None, offset=0):
    for start in range(0, len(ids), BATCH):
        chunk = store.workers(ids[start:start + BATCH])
        texts = [worker_text(w) for w in chunk]
        vectors = embedder.embed(texts, kind="passage")
        store.save_vectors(
            settings.EMBEDDER_KEY,
            [(w["worker_id"], v, digest(t)) for w, v, t in zip(chunk, vectors, texts)],
        )
        if job_id:
            _tick(job_id, offset + start + len(chunk))


def _import(job_id, records, dedupe):
    _tick(job_id, 0, len(records) * 2)
    city_ids, service_ids = {}, {}
    for r in records:
        if r["city"] not in city_ids:
            city_ids[r["city"]] = store.entity_id("city", r["city"], create=True)
        if r["service"] not in service_ids:
            service_ids[r["service"]] = store.entity_id("service", r["service"], create=True)

    existing = store.worker_keys() if dedupe else set()
    added, duplicates = [], 0
    for i, r in enumerate(records, 1):
        cid, sid = city_ids[r["city"]], service_ids[r["service"]]
        if dedupe:
            key = (norm(r["name"]), cid, r.get("phone"))
            if key in existing:
                duplicates += 1
                continue
            existing.add(key)
        added.append(store.insert_worker({**r, "city_id": cid, "service_id": sid}, commit=False))
        if i % 50 == 0:
            _tick(job_id, i)  # job_update commits the pending batch too
    store.commit()  # never hold the write lock while waiting on the embedder

    warning = None
    if added:
        if embedder.ready():
            try:
                _embed_ids(added, job_id, offset=len(records))
            except LLMError as e:
                warning = f"Workers were added but embedding failed ({e}). Run a reindex."
        else:
            warning = "Workers were added but the embedder is not ready. Run a reindex later."
    return len(added), duplicates, warning


def run_import(job_id, records, dedupe=True):
    try:
        added, duplicates, warning = _import(job_id, records, dedupe)
        _finish(job_id, {"ok": True, "added": added, "duplicates": duplicates, "warning": warning})
    except Exception as e:
        _fail(job_id, str(e))


def run_seed(job_id):
    try:
        if not paths.SEED_FILE.exists():
            raise FileNotFoundError(f"seed file not found: {paths.SEED_FILE}")
        records, _ = parse_excel(paths.SEED_FILE.read_bytes())
        added, duplicates, warning = _import(job_id, records, dedupe=True)
        _finish(job_id, {"ok": True, "added": added, "duplicates": duplicates, "warning": warning})
    except Exception as e:
        _fail(job_id, str(e))


def run_reindex(job_id):
    try:
        ids = store.filter_ids()
        _tick(job_id, 0, len(ids))
        if ids:
            _embed_ids(ids, job_id)
            store.delete_other_vectors(settings.EMBEDDER_KEY)
        _finish(job_id, {"ok": True, "reindexed": len(ids)})
    except Exception as e:
        _fail(job_id, str(e))


def embed_worker(worker_id):
    w = store.worker(worker_id)
    if not w or not embedder.ready():
        return
    text = worker_text(w)
    if store.vector_hash(worker_id, settings.EMBEDDER_KEY) == digest(text):
        return
    vector = embedder.embed([text], kind="passage")[0]
    store.save_vectors(settings.EMBEDDER_KEY, [(worker_id, vector, digest(text))])


def embed_worker_later(worker_id):
    threading.Thread(
        target=embed_worker, args=(worker_id,), daemon=True, name=f"embed-{worker_id}"
    ).start()

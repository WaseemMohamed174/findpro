import numpy as np

from . import settings
from .clients import embedder
from .db import store
from .text import fuzzy, name_score, norm

_service_cache = {}


def _unit(matrix):
    return matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-9)


def _semantic_service(name, items):
    if not items or not embedder.ready():
        return None
    try:
        key = tuple(i["name"] for i in items)
        cached = _service_cache.get("service")
        if cached is None or cached[0] != key:
            matrix = _unit(np.asarray(embedder.embed(list(key), kind="passage"), dtype=np.float32))
            cached = _service_cache["service"] = (key, matrix)
        query = _unit(np.asarray(embedder.embed([name], kind="query"), dtype=np.float32))[0]
        sims = cached[1] @ query
        best = int(np.argmax(sims))
        return items[best] if float(sims[best]) >= settings.SERVICE_SIM_MIN else None
    except Exception:
        return None


def resolve(kind, name):
    if not name:
        return None, None, None
    items = store.entities(kind)
    by_norm = {norm(i["name"]): i for i in items}
    target = norm(name)
    hit = by_norm.get(target)
    if hit is None:
        close = fuzzy(target, list(by_norm), settings.FUZZY_CUTOFF)
        hit = by_norm.get(close) if close else None
    if hit is None and kind == "service":
        hit = _semantic_service(name, items)
    if hit:
        return hit["id"], hit["name"], None
    return None, None, name


def _result(intent, **extra):
    base = {
        "intent": intent,
        "workers": [],
        "count": 0,
        "city": None,
        "service": None,
        "ignored_city": None,
        "unknown_service": None,
        "is_empty": False,
        "is_fallback": False,
        "fallback_cities": [],
        "approx": False,
        "route": intent,
    }
    base.update(extra)
    return base


def blend(ids, sims, ratings):
    """Order candidate ids by a mix of semantic match and rating (best first).

    All candidates already satisfy the SQL filters (city, service, gender, age), so the
    semantic part only has to separate them by what the user said beyond those filters
    (a skill, a kind of job). Rating keeps the order sensible when the message says nothing
    extra, and a worker with no rating is treated as average instead of being favoured.
    `sims` maps worker id to cosine similarity and may be empty (embedder down).
    """
    weight = settings.RANK_SIM_WEIGHT if sims else 0.0
    lo = min(sims.values()) if sims else 0.0
    span = (max(sims.values()) - lo) if sims else 0.0

    def score(worker_id):
        rating = ratings.get(worker_id)
        rating = settings.RATING_DEFAULT if rating is None else rating
        sim = (sims.get(worker_id, lo) - lo) / span if span > 1e-6 else 0.5
        return weight * sim + (1 - weight) * (rating / 5.0)

    return sorted(ids, key=lambda i: (-score(i), i))


def _rank(ids, spec):
    sims = {}
    query = (spec.get("raw") or "").strip()
    if len(ids) > 1 and query and embedder.ready():
        try:
            vec = embedder.embed([query], kind="query")[0]
            sims = dict(store.search_vectors(vec, ids, settings.EMBEDDER_KEY, len(ids)))
        except Exception:
            sims = {}
    ordered = blend(ids, sims, store.ratings(ids))
    return store.workers(ordered[:settings.TOP_K])


def _find_named(targets):
    names = None
    found, approx = [], False
    for target in targets:
        t = norm(target)
        ids = store.find_by_name(t)
        if not ids:
            if names is None:
                names = store.all_names()
            by_name = {}
            for wid, n in names.items():
                by_name.setdefault(n, []).append(wid)
            scored = sorted(
                ((name_score(t, n, settings.FUZZY_NAME_CUTOFF, settings.FUZZY_NAME_JOINED_CUTOFF), n)
                 for n in by_name),
                reverse=True,
            )
            close = []
            for score, n in scored[:3]:
                if score > 0:
                    close += by_name[n]
            ids = close
            approx = approx or bool(close)
        if not ids and names and embedder.ready():
            try:
                vec = embedder.embed([target], kind="query")[0]
                hits = store.search_vectors(vec, None, settings.EMBEDDER_KEY, 1)
                ids = [i for i, sim in hits if sim >= settings.NAME_SIM_MIN]
                approx = approx or bool(ids)
            except Exception:
                pass
        found += ids
    unique = list(dict.fromkeys(found))
    return store.workers(unique[:settings.TOP_K]), approx


def _person(spec):
    workers, approx = _find_named(spec["targets"])
    return _result("person", workers=workers, count=len(workers),
                   is_empty=not workers, approx=approx)


def _compare(spec, last_ids):
    workers = store.workers(last_ids) if last_ids else []
    if workers and spec["targets"]:
        by_name = {norm(w["name"]): w for w in workers}
        picked = []
        for target in spec["targets"]:
            close = fuzzy(norm(target), list(by_name), settings.FUZZY_COMPARE_CUTOFF)
            if close:
                picked.append(by_name[close])
        workers = picked or workers
    elif not workers and spec["targets"]:
        workers, _ = _find_named(spec["targets"])
    workers = workers[:settings.COMPARE_MAX]
    return _result("comparison", workers=workers, count=len(workers),
                   is_empty=not workers, route="compare")


def _search(spec, city_id, service_id, city, service, extra):
    filters = {"gender": spec["gender"], "age_min": spec["age_min"], "age_max": spec["age_max"]}
    base = {"city": city, "service": service, **extra}
    if spec["sort"]:
        ids = store.filter_ids(city_id, service_id, sort=spec["sort"],
                               limit=settings.TOP_K, **filters)
        workers = store.workers(ids)
        return _result("search", workers=workers, count=len(workers),
                       is_empty=not workers, route="sql", **base)
    ids = store.filter_ids(city_id, service_id, **filters)
    if ids:
        workers = _rank(ids, spec)
        return _result("search", workers=workers, count=len(workers),
                       is_empty=not workers, route="semantic", **base)
    if city_id is not None:
        alt = store.filter_ids(None, service_id, **filters)
        if alt:
            workers = _rank(alt, spec)
            cities = list(dict.fromkeys(w["city_name"] for w in workers if w.get("city_name")))
            return _result("search", workers=workers, count=len(workers),
                           is_fallback=True, fallback_cities=cities,
                           route="fallback", **base)
    return _result("search", is_empty=True, route="empty", **base)


def run(spec, last_ids=None):
    intent = spec["intent"]
    if intent == "person":
        return _person(spec)
    if intent == "comparison":
        return _compare(spec, last_ids or [])
    city_id, city, ignored_city = resolve("city", spec["city"])
    service_id, service, unknown_service = resolve("service", spec["service"])
    extra = {"ignored_city": ignored_city}
    if unknown_service:
        return _result("unknown_entity", unknown_service=unknown_service, city=city,
                       is_empty=True, route="unknown_entity", **extra)
    if intent == "count":
        n = store.count(city_id, service_id, spec["gender"], spec["age_min"], spec["age_max"])
        return _result("count", count=n, city=city, service=service,
                       is_empty=n == 0, route="count", **extra)
    return _search(spec, city_id, service_id, city, service, extra)

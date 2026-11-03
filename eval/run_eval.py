#!/usr/bin/env python3
"""Measure how well the language model turns messages into structured requests.

It talks to a RUNNING FindPro server (python run.py), so it measures the model you actually
selected. For every case in nlu_cases.jsonl it sends the message(s) and compares what the
server understood (POST /api/explain) with what the case expects.

    python eval/run_eval.py                      # all cases
    python eval/run_eval.py --skip-followups     # faster: single-message cases only
    python eval/run_eval.py --min-accuracy 0.8   # exit code 1 below 80 % (for CI or scripts)

Only the standard library is needed. See eval/README.md.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.text import norm  # noqa: E402  (stdlib-only helper shared with the app)

CASES_FILE = HERE / "nlu_cases.jsonl"
REPORT_FILE = HERE / "last_report.json"

# Every key a case may put in "expect", and where its value is read from.
FIELDS = (
    "intent", "event", "city", "service", "gender", "sort", "age_min", "age_max", "keep",
    "targets_has", "user_name", "home_city", "ignored_city", "is_empty",
)


def load_cases(path=CASES_FILE):
    cases = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith("#"):
                case = json.loads(line)
                validate_case(case, n)
                cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def validate_case(case, line=0):
    where = f"case {case.get('id', '?')} (line {line})"
    for key in ("id", "category", "turns", "expect"):
        if key not in case:
            raise ValueError(f"{where}: missing '{key}'")
    turns = case["turns"]
    if not (isinstance(turns, list) and turns and all(isinstance(t, str) and t.strip() for t in turns)):
        raise ValueError(f"{where}: 'turns' must be a non-empty list of messages")
    expect = case["expect"]
    if not isinstance(expect, dict) or not expect:
        raise ValueError(f"{where}: 'expect' must be a non-empty object")
    unknown = set(expect) - set(FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown expect keys {sorted(unknown)}")
    return case


_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def fold(text):
    """Lower-case and ignore the usual Arabic spelling variants (أ/إ/ا, ى/ي, ة/ه): the
    evaluation judges what the model understood, not how it spelled a city."""
    return norm(text).translate(_FOLD)


def _same(got, want):
    return fold(got) == fold(want) if want else not got


def check(expect, resp):
    """Compare one /api/explain response with the expectations. Returns {field: (ok, got)}."""
    spec = resp.get("spec") or {}
    result = resp.get("result") or {}
    out = {}
    for key, want in expect.items():
        if key == "event":
            got = resp.get("event")
            ok = got == want
        elif key in ("city", "service"):
            got = result.get(key) or spec.get(key)
            ok = _same(got, want)
        elif key in ("user_name", "home_city"):
            got = spec.get(key)
            ok = _same(got, want)
        elif key == "keep":
            got = bool(spec.get("keep"))
            ok = got == want
        elif key == "ignored_city":
            got = bool(result.get("ignored_city"))
            ok = got == want
        elif key == "is_empty":
            got = bool(result.get("is_empty"))
            ok = got == want
        elif key == "targets_has":
            got = spec.get("targets") or []
            ok = all(any(fold(w) in fold(t) for t in got) for w in want)
        else:  # intent, gender, sort, age_min, age_max
            got = spec.get(key)
            ok = got == want
        out[key] = (ok, got)
    return out


def summarize(results):
    """results: list of {id, category, checks|error}. Returns accuracy numbers."""
    fields = defaultdict(lambda: [0, 0])
    cats = defaultdict(lambda: [0, 0])
    exact = 0
    errors = 0
    for r in results:
        checks = r.get("checks")
        ok_case = bool(checks) and all(ok for ok, _ in checks.values())
        if not checks:
            errors += 1
        for key, (ok, _) in (checks or {}).items():
            fields[key][0] += int(ok)
            fields[key][1] += 1
        cats[r["category"]][0] += int(ok_case)
        cats[r["category"]][1] += 1
        exact += int(ok_case)
    total = len(results)
    return {
        "cases": total,
        "exact": exact,
        "accuracy": round(exact / total, 4) if total else 0.0,
        "errors": errors,
        "by_field": {k: {"correct": c, "total": t, "accuracy": round(c / t, 4)}
                     for k, (c, t) in sorted(fields.items())},
        "by_category": {k: {"correct": c, "total": t, "accuracy": round(c / t, 4)}
                        for k, (c, t) in sorted(cats.items())},
    }


def _call(base, path, payload=None, token="", timeout=240):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Admin-Token"] = token
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} on {path}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {base}: {e.reason}") from None


def run_case(case, base, token):
    sid = "eval-" + uuid.uuid4().hex[:16]
    for text in case["turns"][:-1]:
        _call(base, "/api/chat", {"session_id": sid, "text": text}, token)
    resp = _call(base, "/api/explain", {"session_id": sid, "text": case["turns"][-1]}, token)
    if resp.get("error"):
        return {"id": case["id"], "category": case["category"], "error": resp["error"]}
    return {"id": case["id"], "category": case["category"], "text": case["turns"][-1],
            "checks": check(case["expect"], resp)}


def _bar(acc, width=20):
    filled = round(acc * width)
    return "#" * filled + "." * (width - filled)


def print_report(summary, results):
    print(f"\nCases: {summary['cases']}   exactly right: {summary['exact']}   "
          f"accuracy: {summary['accuracy']:.0%}   errors: {summary['errors']}\n")
    print("By category")
    for k, v in summary["by_category"].items():
        print(f"  {k:<16} {v['correct']:>3}/{v['total']:<3} {_bar(v['accuracy'])} {v['accuracy']:.0%}")
    print("\nBy field")
    for k, v in summary["by_field"].items():
        print(f"  {k:<16} {v['correct']:>3}/{v['total']:<3} {_bar(v['accuracy'])} {v['accuracy']:.0%}")
    bad = [r for r in results if "checks" not in r or not all(ok for ok, _ in r["checks"].values())]
    if bad:
        print(f"\nFailures ({len(bad)})")
    for r in bad:
        if "error" in r:
            print(f"  {r['id']}: ERROR {r['error']}")
            continue
        wrong = {k: got for k, (ok, got) in r["checks"].items() if not ok}
        print(f"  {r['id']}: {r['text']!r}")
        print(f"      got {wrong}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('FINDPRO_PORT', '8000')}")
    ap.add_argument("--token", default=os.environ.get("FINDPRO_ADMIN_TOKEN", ""))
    ap.add_argument("--cases", default=str(CASES_FILE))
    ap.add_argument("--category", action="append", help="only this category (repeatable)")
    ap.add_argument("--skip-followups", action="store_true", help="skip cases with several turns")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    ap.add_argument("--report", default=str(REPORT_FILE))
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if args.skip_followups:
        cases = [c for c in cases if len(c["turns"]) == 1]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print("no cases selected")
        return 2

    try:
        health = _call(args.url, "/health")
    except RuntimeError as e:
        print(e)
        return 2
    if not health.get("workers_count"):
        print("The database is empty. Load data first (Demo data button or POST /api/seed_demo).")
        return 2
    selection = health.get("selection", {})
    print(f"Server: {args.url}   model: {selection.get('llm', {}).get('name')}   "
          f"embedder: {selection.get('embedder', {}).get('key')}   workers: {health['workers_count']}")

    results, t0 = [], time.time()
    for i, case in enumerate(cases, 1):
        try:
            res = run_case(case, args.url, args.token)
        except RuntimeError as e:
            res = {"id": case["id"], "category": case["category"], "error": str(e)}
        results.append(res)
        mark = "ERR" if "error" in res else ("ok " if all(ok for ok, _ in res["checks"].values()) else "BAD")
        print(f"[{i:>3}/{len(cases)}] {mark} {case['id']}", flush=True)

    summary = summarize(results)
    summary["seconds"] = round(time.time() - t0, 1)
    print_report(summary, results)

    report = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection": selection,
        "summary": summary,
        "results": [
            {**{k: v for k, v in r.items() if k != "checks"},
             **({"checks": {k: {"ok": ok, "got": got} for k, (ok, got) in r["checks"].items()}}
                if "checks" in r else {})}
            for r in results
        ],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.report}")
    return 0 if summary["accuracy"] >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())

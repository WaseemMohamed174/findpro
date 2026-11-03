#!/usr/bin/env python3
"""Compare two reports written by run_eval.py (for example two different language models).

    python eval/compare_reports.py qwen3b.json qwen7b.json

Shows the model behind each report, the accuracy per category and per field side by side, and
the cases that one model got right and the other got wrong.
"""
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _right(result):
    checks = result.get("checks")
    return bool(checks) and all(c["ok"] for c in checks.values())


def compare(a, b):
    """Return the numbers as data so they can be tested; main() only prints them."""
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    shared = sorted(set(ra) & set(rb))
    only_a = [i for i in shared if _right(ra[i]) and not _right(rb[i])]
    only_b = [i for i in shared if _right(rb[i]) and not _right(ra[i])]
    return {
        "shared_cases": len(shared),
        "only_a_right": only_a,
        "only_b_right": only_b,
        "accuracy": (a["summary"]["accuracy"], b["summary"]["accuracy"]),
        "by_category": _side_by_side(a["summary"]["by_category"], b["summary"]["by_category"]),
        "by_field": _side_by_side(a["summary"]["by_field"], b["summary"]["by_field"]),
    }


def _side_by_side(x, y):
    return {k: (x.get(k, {}).get("accuracy"), y.get(k, {}).get("accuracy"))
            for k in sorted(set(x) | set(y))}


def _name(report):
    sel = report.get("selection", {})
    return f"{sel.get('llm', {}).get('name', '?')} + {sel.get('embedder', {}).get('key', '?')}"


def _pct(v):
    return "  n/a" if v is None else f"{v:>4.0%}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = load(argv[0]), load(argv[1])
    out = compare(a, b)
    print(f"A = {_name(a)}   ({argv[0]})\nB = {_name(b)}   ({argv[1]})")
    print(f"\nExact-match accuracy over {out['shared_cases']} shared cases:  "
          f"A {_pct(out['accuracy'][0])}   B {_pct(out['accuracy'][1])}")
    for title, table in (("Category", out["by_category"]), ("Field", out["by_field"])):
        print(f"\n{title:<16} {'A':>5} {'B':>5}")
        for key, (x, y) in table.items():
            print(f"{key:<16} {_pct(x)} {_pct(y)}")
    print(f"\nOnly A got right ({len(out['only_a_right'])}): {', '.join(out['only_a_right']) or '-'}")
    print(f"Only B got right ({len(out['only_b_right'])}): {', '.join(out['only_b_right']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

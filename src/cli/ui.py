from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
import unicodedata

if os.name == "nt":
    os.system("")

_COLOR = bool(sys.stdout.isatty()) and not os.environ.get("NO_COLOR")
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t):   return _c("1", t)
def dim(t):    return _c("2", t)
def gray(t):   return _c("90", t)
def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def strip(s):  return _ANSI.sub("", s)


# ---------------------------------------------------------------- measuring

def _w(s: str) -> int:
    """Display width (ANSI codes ignored, wide chars count 2, marks 0)."""
    n = 0
    for ch in strip(s):
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def _pad(s: str, w: int) -> str:
    return s + " " * max(w - _w(s), 0)


def _trunc(s: str, w: int) -> str:
    if _w(s) <= w:
        return s
    while s and _w(s) > w - 1:
        s = s[:-1]
    return s + "\u2026"


def width() -> int:
    cols = shutil.get_terminal_size((80, 24)).columns
    return max(78, min(cols - 1, 100))


# ------------------------------------------------------------ one-line msgs

def info(msg: str) -> None: print(f"  {cyan('•')} {msg}", flush=True)
def ok(msg: str) -> None:   print(green(f"  ✓ {msg}"), flush=True)
def warn(msg: str) -> None: print(yellow(f"  ⚠ {msg}"), flush=True)
def fail(msg: str) -> None: print(red(f"  ✗ {msg}"), flush=True)


# ------------------------------------------------------- steps and sections

STEPS = ("Libraries", "Models", "Download", "Launch")


def step(n: int, title: str) -> None:
    """Big banner for a major stage + a breadcrumb showing where we are."""
    W = width()
    crumbs = []
    for i, name in enumerate(STEPS, 1):
        if i < n:
            crumbs.append(green(f"✓ {name}"))
        elif i == n:
            crumbs.append(bold(cyan(f"● {name}")))
        else:
            crumbs.append(gray(f"○ {name}"))
    print()
    print(cyan("━" * W))
    print(" " + bold(cyan(f"STEP {n}/{len(STEPS)}")) + gray("  ·  ") + bold(title.upper()))
    print(" " + gray("  ›  ").join(crumbs))
    print(cyan("━" * W))


def section(title: str, note: str = "") -> None:
    """Thin full-width rule with a title: separates sub-sections of a step."""
    W = width()
    right = f" {note} ──" if note else ""
    fill = max(W - _w(f"── {title} ") - _w(right), 2)
    line = cyan("── ") + bold(title) + " " + cyan("─" * fill)
    if note:
        line += " " + gray(note) + " " + cyan("──")
    print()
    print(line)


# ------------------------------------------------------------------- boxes

def box(title: str, lines: list[str], width_: int | None = None) -> None:
    W = width_ or width()
    inner = W - 4
    flat: list[str] = []
    for ln in lines:
        if strip(ln) == ln and _w(ln) > inner:
            flat.extend(textwrap.wrap(ln, inner) or [""])
        else:
            flat.append(ln)

    print(cyan("╔" + "═" * (W - 2) + "╗"))
    t = f" {title} "
    print(cyan("║") + bold(t) + " " * max(W - 2 - _w(t), 0) + cyan("║"))
    print(cyan("╟" + "─" * (W - 2) + "╢"))
    for ln in flat:
        print(cyan("║") + " " + ln + " " * max(inner - _w(ln), 0) + " " + cyan("║"))
    print(cyan("╚" + "═" * (W - 2) + "╝"))


def explain(options: list[tuple[str, str]], title: str = "WHAT EACH OPTION DOES") -> None:
    """Explanation box. Long text wraps INSIDE the box with a hanging indent,
    and all the descriptions line up in one column."""
    inner = width() - 4
    prefixes = [f"{k}." if k[:1].isdigit() else k for k, _ in options]
    pw = max((_w(p) for p in prefixes), default=0)
    indent = pw + 1
    lines: list[str] = []
    for prefix, (_, text) in zip(prefixes, options):
        wrapped = textwrap.wrap(text, max(inner - indent, 20)) or [""]
        lines.append(bold(_pad(prefix, pw)) + " " + wrapped[0])
        lines.extend(" " * indent + w for w in wrapped[1:])
    box(title, lines)


def table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    if rows:
        w = [max(_w(h), *(_w(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    else:
        w = [_w(h) for h in headers]

    def rule(left, mid, right):
        return cyan(left + mid.join("─" * (x + 2) for x in w) + right)

    if title:
        print(bold(f" {title}"))
    print(rule("┌", "┬", "┐"))
    print(cyan("│")
          + cyan("│").join(f" {bold(_pad(h, w[i]))} " for i, h in enumerate(headers))
          + cyan("│"))
    print(rule("├", "┼", "┤"))
    for r in rows:
        cells = [_pad(str(c), w[i]) for i, c in enumerate(r)]
        print(cyan("│") + cyan("│").join(f" {c} " for c in cells) + cyan("│"))
    print(rule("└", "┴", "┘"))


# -------------------------------------------------------------------- menus

def ask(n: int, default: int = 1) -> int:
    if not sys.stdin.isatty() or os.environ.get("FINDPRO_AUTO") == "1":
        print(dim(f"  (non-interactive → using default {default})"), flush=True)
        return default

    rng = "1" if n == 1 else f"1-{n}"
    while True:
        try:
            raw = input(
                f"  Select {cyan(f'[{rng}]')} {dim(f'(Enter = {default})')}: "
            ).strip()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw)
        print(red(f"  Enter a number from {rng}."), flush=True)


def ask_text(prompt: str) -> str:
    try:
        return input(f"  {prompt}\n  > ").strip().strip('"').strip("'")
    except EOFError:
        return ""


def _options_box(headers: list[str], rows: list[dict], numbers: dict[int, int]) -> None:
    W = width()
    inner = W - 4
    gap = 2
    n_data = len(headers)
    has_status = any(r.get("state") in ("rec", "blocked") or r.get("tag") for r in rows)

    head = ["#"] + list(headers) + ([""] if has_status else [])
    body: list[tuple[list[str], str]] = []
    for i, r in enumerate(rows):
        state = r.get("state", "ok")
        num = str(numbers[i]) if i in numbers else "✕"
        data = [str(x) for x in r["cols"]]
        data += [""] * (n_data - len(data))
        if state == "rec":
            status = "★ recommended"
        elif state == "blocked":
            status = "not available"
        else:
            status = r.get("tag", "")
        body.append(([num] + data + ([status] if has_status else []), state))

    nc = len(head)
    widths = [max(_w(head[c]), *(_w(cells[c]) for cells, _ in body)) for c in range(nc)]
    over = sum(widths) + gap * (nc - 1) - inner
    if over > 0 and nc > 1:
        widths[1] = max(widths[1] - over, 8)
    used = sum(widths) + gap * (nc - 1)

    def line(cells: list[str], state: str, header: bool = False) -> str:
        parts = []
        for c, (txt, w) in enumerate(zip(cells, widths)):
            txt = _pad(_trunc(txt, w), w)
            last_status = has_status and c == nc - 1
            if header:
                txt = bold(txt)
            elif state == "blocked":
                txt = gray(txt)
            elif state == "rec" and c == 0:
                txt = bold(green(txt))
            elif state == "rec" and last_status:
                txt = green(txt)
            elif last_status and txt.strip():
                txt = yellow(txt)
            parts.append(txt)
        return (cyan("│") + " " + (" " * gap).join(parts)
                + " " * max(inner - used, 0) + " " + cyan("│"))

    print(cyan("┌" + "─" * (W - 2) + "┐"))
    print(line(head, "ok", header=True))
    print(cyan("├" + "─" * (W - 2) + "┤"))
    for cells, state in body:
        print(line(cells, state))
    print(cyan("└" + "─" * (W - 2) + "┘"))


def menu(title: str | None, headers: list[str], items: list[dict], *,
         extras: list[dict] | tuple = (), explain_lines: list[tuple[str, str]] | None = None,
         note: str = "") -> int:
    """One layout for every selection screen:

        ── Title ────────────── note ──
        ╔ WHAT EACH OPTION DOES ╗      <- explanation first
        ┌ numbered options table ┐     <- options last, right above the prompt
        Select [1-N] (Enter = d):

    items / extras are dicts: cols (list[str]), state ("ok" | "rec" | "blocked"),
    desc (explanation text), tag (small status text for "ok" rows).
    Rows in state "blocked" are shown gray, with no number, and cannot be chosen.
    Returns the 0-based index into items + extras.
    """
    rows = list(items) + list(extras)
    n_items = len(items)

    numbers: dict[int, int] = {}
    for i, r in enumerate(rows):
        if r.get("state", "ok") != "blocked":
            numbers[i] = len(numbers) + 1
    if not numbers:
        raise RuntimeError("no selectable options")

    rec = [numbers[i] for i, r in enumerate(rows)
           if r.get("state") == "rec" and i in numbers]
    default = rec[0] if rec else 1

    exp: list[tuple[str, str]] = []
    if explain_lines is None:
        for i, r in enumerate(rows):
            if i in numbers and r.get("desc"):
                exp.append((str(numbers[i]), r["desc"]))
    else:
        nums = [numbers[i] for i in range(n_items) if i in numbers]
        if nums:
            rng = str(nums[0]) if len(nums) == 1 else f"{nums[0]}-{nums[-1]}"
            exp.extend((k.format(range=rng), t) for k, t in explain_lines)
        for i in range(n_items, len(rows)):
            if i in numbers and rows[i].get("desc"):
                exp.append((str(numbers[i]), rows[i]["desc"]))
    if any(r.get("state") == "blocked" for r in rows):
        exp.append(("✕", "Gray rows need more memory than is safe on this "
                         "machine, so they cannot be selected."))

    if title:
        section(title, note)
    if exp:
        explain(exp)
        print()
    _options_box(headers, rows, numbers)

    chosen = ask(len(numbers), default)
    return next(i for i, num in numbers.items() if num == chosen)


def run_quiet(cmd: list[str], log_path, on_line=None) -> int:
    from pathlib import Path
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            if on_line:
                on_line(line)
        return p.wait()


class InstallProgress:

    W = 30

    def __init__(self, total: int):
        self.total = total
        self.i = 0
        self.name = ""
        self._drawn = False

    def start(self, name: str) -> None:
        self.i += 1
        self.name = name
        self.update(0.0)

    def update(self, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        if not _COLOR:
            if frac in (0.0, 1.0):
                print(f"({self.i}/{self.total}) {self.name} "
                      f"{int(frac * 100):3d}%", flush=True)
            return
        f = int(self.W * frac)
        l1 = f"({self.i}/{self.total}) {bold(self.name)}"
        l2 = (f"progress: [{green('█' * f)}{dim('░' * (self.W - f))}] "
              f"{int(frac * 100):3d}%")
        if self._drawn:
            sys.stdout.write("\033[2F")
        sys.stdout.write("\033[2K" + l1 + "\n\033[2K" + l2 + "\n")
        sys.stdout.flush()
        self._drawn = True

    def done(self) -> None:
        self.update(1.0)

    def close(self) -> None:
        self._drawn = False
        ok(f"{self.total}/{self.total} libraries installed")


def pip_progress_hook(prog: InstallProgress):
    state = {"f": 0.0}
    steps = (
        ("Collecting", 0.15),
        ("Downloading", 0.45),
        ("Building", 0.60),
        ("Installing collected", 0.85),
        ("Successfully installed", 1.0),
    )

    def hook(line: str) -> None:
        for key, val in steps:
            if line.startswith(key) and val > state["f"]:
                state["f"] = val
                prog.update(val)
    return hook

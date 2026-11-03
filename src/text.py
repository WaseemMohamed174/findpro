import hashlib
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher, get_close_matches

_WS = re.compile(r"\s+")
_SEP = re.compile(r"[,;|\n\u060c\u061b\u3001\uff0c]")


def norm(text):
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WS.sub(" ", stripped.casefold()).strip()


def split_list(value):
    return [x.strip() for x in _SEP.split(value or "") if x.strip()]


def fuzzy(word, candidates, cutoff):
    if not word or not candidates:
        return None
    hits = get_close_matches(word, candidates, n=1, cutoff=cutoff)
    return hits[0] if hits else None


def clean_field(value, limit=120):
    """Make a stored text value safe to paste into a model prompt as data.

    Drops control characters and line breaks (so a cell cannot fake a new instruction block)
    and clips the length.
    """
    if value is None:
        return value
    flat = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in str(value))
    return _WS.sub(" ", flat).strip()[:limit]


_SPELLING = str.maketrans({"ى": "ي", "ة": "ه"})


def name_score(typed, stored, word_cutoff=0.70, joined_cutoff=0.82):
    """How well a typed person name matches a stored one (both already norm()-ed): 0 or a score.

    Every typed word must be close to some word of the stored name (so a stranger's name does
    not match just because it shares a few letters with a long full name), or the two names must
    be nearly identical once spaces are ignored ("عبد الكريم" vs "عبدالكريم"). ى/ي and ة/ه are
    treated as the same letter because they are the most common spelling slips.
    """
    t, n = typed.translate(_SPELLING), stored.translate(_SPELLING)
    if not t or not n:
        return 0.0
    parts = n.split()
    words = [max(SequenceMatcher(None, w, p).ratio() for p in parts) for w in t.split()]
    best = min(words) if words and min(words) >= word_cutoff else 0.0
    joined = SequenceMatcher(None, t.replace(" ", ""), n.replace(" ", "")).ratio()
    return max(best, joined if joined >= joined_cutoff else 0.0)


def direction(text):
    rtl = sum(1 for ch in text if unicodedata.bidirectional(ch) in ("R", "AL"))
    ltr = sum(1 for ch in text if unicodedata.bidirectional(ch) == "L")
    return "rtl" if rtl > ltr else "ltr"


def parse_json(raw):
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None
    return None


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

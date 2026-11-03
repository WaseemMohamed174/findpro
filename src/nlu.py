from . import log, settings
from .clients import LLMError, llm
from .text import parse_json

logger = log.get("nlu")

INTENTS = ("search", "count", "person", "comparison", "chat")
GENDERS = ("male", "female")
SORTS = ("rating", "rate_asc", "rate_desc", "experience")

PROMPT = """You turn a user's message into a JSON request for a service that finds workers.
The user may write in any language or dialect. Understand the message as written; never ask for a different language.

Return ONLY one JSON object with these keys:
- "intent": one of
  "search"     the user wants workers listed or recommended
  "count"      the user asks how many workers match
  "person"     the user asks about a specific named worker
  "comparison" the user wants workers compared (the ones shown earlier or the ones named)
  "chat"       greetings, thanks, goodbyes, introductions, complaints, small talk, or anything that does not ask to find, count or compare workers
- "city": a value copied exactly from AVAILABLE CITIES, or null. Only a real place name belongs here: never put a person's name, a profession or a descriptive word in "city". If the user names a place that is not in the list, write the place as the user wrote it.
- "service": a value copied character for character from AVAILABLE SERVICES, or null. Never translate it and never write it in another language: use the exact listed spelling. Map professions, trades, tools and problems to the closest listed service, in any language. If nothing in the list fits, write the user's own wording.
- "gender": "male", "female" or null
- "age_min", "age_max": integers or null
- "sort": "rating" (best rated), "rate_asc" (cheapest), "rate_desc" (most expensive), "experience" (most experienced), or null
- "targets": person names that appear in the user's message (for "person" and "comparison"). Copy them exactly as the user wrote them. If the user wrote no person name, "targets" is [] and the intent is never "person". Never invent a name.
- "user_name": the user's own name if they state it, otherwise null
- "home_city": the city where the user says they live, otherwise null
- "keep": true when the message refines the previous search (for example only changes a filter), false when it is a new request
- "lang": the BCP-47 code of the language the user wrote in

A complaint or a description of a problem without asking for help is "chat".

Examples (the user's language varies; copy city and service values exactly from the lists below):
User: "I need help with {ex_service} in {ex_city}" -> {"intent":"search","city":"{ex_city}","service":"{ex_service}","gender":null,"age_min":null,"age_max":null,"sort":null,"targets":[],"user_name":null,"home_city":null,"keep":false,"lang":"en"}
User: "who are the best rated for {ex_service}?" -> {"intent":"search","city":null,"service":"{ex_service}","gender":null,"age_min":null,"age_max":null,"sort":"rating","targets":[],"user_name":null,"home_city":null,"keep":false,"lang":"en"}
User: "how many workers do {ex_service2} here?" -> {"intent":"count","city":null,"service":"{ex_service2}","gender":null,"age_min":null,"age_max":null,"sort":null,"targets":[],"user_name":null,"home_city":null,"keep":false,"lang":"en"}
User: "thanks, goodbye" -> {"intent":"chat","city":null,"service":null,"gender":null,"age_min":null,"age_max":null,"sort":null,"targets":[],"user_name":null,"home_city":null,"keep":false,"lang":"en"}
User (after a search): "only women, best rated" -> {"intent":"search","city":null,"service":null,"gender":"female","age_min":null,"age_max":null,"sort":"rating","targets":[],"user_name":null,"home_city":null,"keep":true,"lang":"en"}

AVAILABLE CITIES: {cities}
AVAILABLE SERVICES: {services}"""


def _catalog(items):
    names = [i["name"] for i in items[:settings.CATALOG_LIMIT]]
    return " | ".join(names) if names else "(none)"


def _pick(items, i):
    names = [x["name"] for x in items[:settings.CATALOG_LIMIT]]
    if not names:
        return "(none)"
    return names[i] if len(names) > i else names[0]


def _system(cities, services):
    return (
        PROMPT.replace("{ex_service2}", _pick(services, 1))
        .replace("{ex_service}", _pick(services, 0))
        .replace("{ex_city}", _pick(cities, 0))
        .replace("{cities}", _catalog(cities))
        .replace("{services}", _catalog(services))
    )


def _user(text, ctx, history, shown):
    recent = history[-settings.NLU_HISTORY:]
    lines = [f"user: {m['content']}" for m in recent if m["role"] == "user"] or ["(none)"]
    context = []
    if ctx.get("user_name"):
        context.append(f"user name: {ctx['user_name']}")
    if ctx.get("home_city"):
        context.append(f"home city: {ctx['home_city']}")
    if ctx.get("last_filters"):
        context.append(f"last filters: {ctx['last_filters']}")
    if shown:
        context.append("workers shown last: " + ", ".join(shown))
    return (
        "Context:\n" + ("\n".join(context) or "(none)")
        + "\n\nRecent messages:\n" + "\n".join(lines)
        + "\n\nUser message: " + text
    )


def _str(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() not in ("null", "none") else None


def _int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse(raw, text):
    data = parse_json(raw)
    if not isinstance(data, dict) or data.get("intent") not in INTENTS:
        return None
    gender = data.get("gender")
    sort = data.get("sort")
    age_min, age_max = _int(data.get("age_min")), _int(data.get("age_max"))
    if age_min is not None and age_max is not None and age_min > age_max:
        age_min, age_max = age_max, age_min
    targets = data.get("targets")
    return {
        "intent": data["intent"],
        "city": _str(data.get("city")),
        "service": _str(data.get("service")),
        "gender": gender if gender in GENDERS else None,
        "age_min": age_min,
        "age_max": age_max,
        "sort": sort if sort in SORTS else None,
        "targets": [t.strip() for t in targets if isinstance(t, str) and t.strip()]
        if isinstance(targets, list) else [],
        "user_name": _str(data.get("user_name")),
        "home_city": _str(data.get("home_city")),
        "keep": bool(data.get("keep")),
        "lang": _str(data.get("lang")),
        "raw": text,
    }


def _ask(messages):
    return llm.chat(
        messages, json_mode=True,
        max_tokens=settings.LLM_NLU_TOKENS, temperature=settings.LLM_TEMP_NLU,
    )


def understand(text, ctx, history, cities, services, shown, debug=None):
    if not llm.ready():
        return None
    system = _system(cities, services)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _user(text, ctx, history, shown)},
    ]
    if debug is not None:
        debug.update({"system_chars": len(system), "attempts": []})
    spec = None
    for attempt in range(2):
        try:
            raw = _ask(messages)
        except LLMError as e:
            if debug is not None:
                debug["attempts"].append({"error": str(e)})
            return None
        spec = parse(raw, text)
        logger.debug("attempt=%d text=%r raw=%r", attempt + 1, text, raw)
        if spec is None:
            logger.info("attempt=%d produced no valid request", attempt + 1)
        if debug is not None:
            debug["attempts"].append({"raw": raw, "valid": spec is not None})
        if spec is not None:
            break
        messages = messages + [
            {"role": "assistant", "content": raw or ""},
            {"role": "user", "content": 'That was not valid. Return ONLY one JSON object with the keys described, and a valid "intent".'},
        ]
    return spec

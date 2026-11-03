# Evaluating the language model

The app asks a small language model to turn a message such as *"محتاج كهربائي في المنصورة"* into a
structured request (intent, city, service, gender, sort, age range, ...). Everything after that
(search, ranking, the reply) is only as good as that first step, so it is the step worth measuring.

`run_eval.py` sends 69 hand-written messages to a **running** server and checks what it understood.
Because it talks to the server, it measures the model you actually selected, not a fake one.

```bash
python run.py                       # start the app and load the demo data (Workers panel)
python eval/run_eval.py             # in another terminal
```

Only the standard library is needed. The demo data must be loaded: the cases refer to its cities
and services.

| Option | Meaning |
|---|---|
| `--skip-followups` | Run only single-message cases (much faster; follow-ups need several model calls each) |
| `--category search --category count` | Run only these categories |
| `--limit 10` | First 10 selected cases, for a quick smoke test |
| `--min-accuracy 0.8` | Exit with code 1 if exact-match accuracy is below 80 % |
| `--url`, `--token` | Server address (default `http://127.0.0.1:8000`) and admin token (also read from `FINDPRO_ADMIN_TOKEN`) |
| `--report path.json` | Where to write the report (default `eval/last_report.json`) |

The evaluation uses `POST /api/explain`, which is an admin endpoint. From the same machine and
without a token it just works; if you set `FINDPRO_ADMIN_TOKEN`, pass it the same way.

## Reading the result

* **accuracy** is the share of cases where *every* checked field was right. It is strict on purpose.
* **By field** shows which part is weak: for example `service` low and `city` high means the model
  finds places but maps professions badly.
* **By category** shows which kind of message is weak: dialect search, follow-ups, names, and so on.
* **Failures** list each wrong case with what the model produced instead.

Run it on every model you consider and keep the reports. Compare two of them:

```bash
cp eval/last_report.json eval/qwen3b.json          # after running with model A
cp eval/last_report.json eval/qwen7b.json          # after running with model B
python eval/compare_reports.py eval/qwen3b.json eval/qwen7b.json
```

## What a case looks like

One JSON object per line in `nlu_cases.jsonl`:

```json
{"id": "followup-01", "category": "followup",
 "turns": ["عايز نجار في أكتوبر", "بس الستات"],
 "expect": {"intent": "search", "keep": true, "gender": "female"}}
```

`turns` are sent in order inside one chat session; only the **last** one is checked, so earlier
turns set up the context for a follow-up. Only the fields listed in `expect` are scored.

| `expect` key | Checked against | Notes |
|---|---|---|
| `intent` | what the model produced | `search`, `count`, `person`, `comparison`, `chat` |
| `event` | what the app decided to do | `unknown_entity` (service not in the data), `clarify` (nothing to search for), or the intent name |
| `city`, `service` | the value the app resolved, else the model's | must exist in the data; `null` means "no city/service" |
| `gender` | model | `male`, `female` or `null` |
| `sort` | model | `rating`, `rate_asc`, `rate_desc`, `experience`, or `null` |
| `age_min`, `age_max` | model | integers or `null` |
| `keep` | model | `true` when the message refines the previous search |
| `targets_has` | model's list of names | every listed word must appear inside some name it extracted |
| `user_name`, `home_city` | model | what the user said about themselves |
| `ignored_city` | the app | `true` when the city is not in the data (e.g. القاهرة in the demo set) |
| `is_empty` | the app | `true` when a name search should find nobody |

Spelling variants of the same Arabic letter (أ/إ/ا, ى/ي, ة/ه) count as equal.

### Adding a case

1. Append a line to `nlu_cases.jsonl` with a new unique `id` (`category-NN`).
2. Take cities and services from your data, exactly as written there.
3. For a name that should be found, use a name that exists in the data; for one that should not,
   use one that does not. `tests/test_eval_harness.py` fails if the data and the case disagree.
4. Run `pytest tests/test_eval_harness.py`. It validates the file and also runs every case against
   a perfect fake model; if your new case cannot pass even then, the expectation is wrong or the
   app needs a fix. (For a new follow-up case, add the messages to `MANUAL` in that test.)

Aim for messages that real users would type: dialect, typos, English, mixed phrasing. A set made
only of clean sentences will flatter the model.

## Tuning the two search knobs

Two numbers in `src/settings.py` were chosen by reasoning, not measured against a real model. Both
can be changed without editing code:

| Variable | Default | What it does |
|---|---|---|
| `FINDPRO_NAME_SIM_MIN` | `0.90` | A person's name that matches no stored name by spelling is looked up by embedding; the closest worker is returned only if the cosine similarity reaches this value. Lower = more guesses, higher = more "not found". Use `-1` to accept anything (the old behaviour) |
| `FINDPRO_RANK_SIM_WEIGHT` | `0.35` | Share of the semantic match in the final order of search results; the rest is the worker's rating. `0` = rating only, `1` = meaning only |

How to tune `FINDPRO_NAME_SIM_MIN`: start the app with different values and run
`python eval/run_eval.py --category person`. The two cases with `"is_empty": true` must stay empty
(otherwise the app invents a person) and the others must still be found. That is only a handful
of cases; add more names of both kinds before trusting a value.

The evaluation does **not** measure the ranking weight, because that needs cases of the form
"for this message, worker X should come first". Until those exist, treat `0.35` as a sensible
starting point rather than a tuned value.

## What this does not measure

* The quality of the final written reply (tone, whether it invents a phone number). It checks the
  understanding step only.
* Speed and memory.
* Anything on data other than the demo set.

`tests/test_eval_harness.py` also shows what the surrounding code is worth: with a model that
always answers correctly, every case passes. Any gap you see in a real report is therefore the
model's, not the search code's.

# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses [semantic versioning](https://semver.org/).

## [1.1.0] - 2026-11-03

First release under the name **FindPro** (the project was called "Workers AI" before; that name is
also a Cloudflare product, which made it hard to find and easy to confuse).

### Breaking changes

* **Environment variables were renamed** from `WORKERSAI_*` to `FINDPRO_*`
  (`FINDPRO_HOST`, `FINDPRO_PORT`, `FINDPRO_ADMIN_TOKEN`, `FINDPRO_AUTO`). The old names are ignored.
* **The server refuses to start on a non-local address without `FINDPRO_ADMIN_TOKEN`.**
  `FINDPRO_HOST=0.0.0.0` now needs a token. The default `127.0.0.1` is unchanged.
* **Endpoints that change data now require authorisation.** `POST/PUT/DELETE /api/workers`,
  `POST /api/upload`, `/api/seed_demo`, `/api/reindex`, `/api/system/restart/*`, `/api/explain` and
  `/api/admin/*` accept a request from the same machine only, or any request carrying the
  `X-Admin-Token` header when `FINDPRO_ADMIN_TOKEN` is set. Cross-site browser requests are refused.
  The web interface asks for the token once if one is needed.
* **`session_id` must be 8 to 64 characters from `A-Za-z0-9_-`**, and a message can be at most
  2000 characters. Other values get `422`. The web interface already generates valid ids.
* **`/static/*` is no longer served.** It exposed everything in `data/static/`, including the
  workers Excel file with phone numbers.

### Added

* **Evaluation set and runner** (`eval/`): 69 messages in Egyptian dialect, Modern Standard Arabic,
  English and transliterated Arabic, with `run_eval.py` to score a running server and
  `compare_reports.py` to compare two models. See `eval/README.md`.
* **Built-in column and gender dictionaries for the Excel import** (Arabic and English). Ordinary
  files, including the demo workbook, now import instantly and without a language model; the model
  is only asked about headers or values the dictionaries do not know.
* `POST /api/explain` can replay a follow-up: send the `session_id` of an existing conversation
  (nothing is added to its history).
* `FINDPRO_NAME_SIM_MIN`, `FINDPRO_RANK_SIM_WEIGHT` and `FINDPRO_DEBUG` settings.
* `LICENSE` (MIT), GitHub Actions workflow running the tests on Python 3.10 and 3.12, and
  `requirements-ci.txt`.
* 56 new tests (117 in total).

### Changed

* **Result ordering.** Candidates that already match the filters are ordered by a blend of
  semantic similarity to the user's message and the worker's rating. Before, the semantic part
  compared the workers with the words "service city", which are the same for all of them, so it
  added noise and ignored the rating. A worker with no rating counts as average.
* **Searching a person by name is stricter.** Each typed word must be close to a word of the stored
  name (ى/ي and ة/ه are treated as the same letter), or the whole name must match once spaces are
  ignored. The old whole-string cutoff matched unrelated people (an invented name returned a real
  worker). The embedding fallback now answers "not found" unless the similarity reaches
  `NAME_SIM_MIN` (0.90) instead of always returning the nearest worker. Approximate matches are now
  reported as approximate.
* Text taken from the database is cleaned and shortened before it goes into the model prompt, and
  the prompt tells the model to ignore anything inside it that looks like an instruction.
* The shared secret between the gateway and the model workers is random per start
  (it was the constant `local-dev-token`).
* `/api/admin/select` validates its body and accepts only models from the catalog; choosing a local
  file by path is possible only from the terminal wizard.
* Model workers use FastAPI lifespan handlers instead of the deprecated `on_event`; the supervisor
  and the language-understanding step log through `logging`. Messages and raw model output are
  logged only with `FINDPRO_DEBUG=1`.

### Fixed

* `/api/explain` no longer returns Python tracebacks unless `FINDPRO_DEBUG=1`.
* The wildcard characters `%` and `_` in a searched name were treated as wildcards.
* Loading the demo data failed when no language model was ready (see the import dictionaries above).

## [1.0.0]

Initial release: chat over a SQLite database of workers, with a language model for understanding
and replying, an embedding model for semantic search, Excel import, and a supervisor that starts
and monitors the two model processes.

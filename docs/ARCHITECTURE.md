# FindPro - Architecture

Three views of the system: how a chat message flows through it, how the data is stored, and how the runtime classes relate.

## 1. Workflow - life of a chat message

![Workflow animation](assets/workflow.gif)

A message enters through `POST /api/chat/stream`. The gateway asks the local LLM worker to turn it into a structured request (`nlu.understand`), then `search.run` picks a route:

- **SQL route** for sorted queries ("best rated", "cheapest").
- **Semantic route** filters rows in SQLite, then the embedder worker re-ranks the candidates.
- **Fallback** widens the city when nothing matches.
- **Clarify** when the message has no filters at all.

`chat.handle_stream` then composes the reply with the LLM worker, streams it back over SSE, and saves the session.

Interactive version (open locally in a browser): [`diagrams/workflow.html`](diagrams/workflow.html) - press **Live** to play the animation.

## 2. ERD - SQLite schema

![ERD](assets/erd.png)

Interactive version: [`diagrams/erd.html`](diagrams/erd.html). Source of truth: `src/db.py` (`SCHEMA`).

```mermaid
erDiagram
    cities ||--o{ workers : "in city"
    services ||--o{ workers : "offers"
    workers ||--o| worker_vectors : "embeds"
    chat_sessions ||--o{ chat_messages : "has"

    cities {
        INTEGER city_id PK
        TEXT name
        TEXT name_norm UK
    }
    services {
        INTEGER service_id PK
        TEXT name
        TEXT name_norm UK
    }
    workers {
        INTEGER worker_id PK
        TEXT name
        TEXT name_norm
        INTEGER age
        TEXT gender
        TEXT nationality
        INTEGER city_id FK
        INTEGER service_id FK
        INTEGER experience_years
        REAL rating
        INTEGER hourly_rate
        TEXT phone
        INTEGER completed_projects
        TEXT skills
        TEXT created_at
    }
    worker_vectors {
        INTEGER worker_id PK, FK
        TEXT model_key
        INTEGER dim
        BLOB vec
        TEXT text_hash
    }
    chat_sessions {
        TEXT session_id PK
        TEXT context_json
        TEXT created_at
        TEXT last_active
    }
    chat_messages {
        INTEGER message_id PK
        TEXT session_id FK
        TEXT role
        TEXT content
        TEXT intent
        TEXT created_at
    }
    jobs {
        INTEGER job_id PK
        TEXT kind
        TEXT status
        TEXT label
        INTEGER total
        INTEGER done
        TEXT result_json
        TEXT error
        TEXT created_at
        TEXT finished_at
    }
```

Delete behaviour: removing a city or service sets `workers.city_id` / `service_id` to `NULL`; removing a worker cascades to `worker_vectors`; removing a chat session cascades to its messages. `jobs` has no foreign keys.

## 3. UML - runtime classes

![UML class diagram](assets/uml-class.png)

Interactive version: [`diagrams/uml-class.html`](diagrams/uml-class.html). Sources: `src/runtime.py`, `src/clients.py`, `src/hardware.py`, `src/db.py`.

```mermaid
classDiagram
    class _Remote {
        <<abstract>>
        #str name
        +ready() bool
        #_open(path, payload, timeout, stream)
    }
    class Embedder {
        +embed(texts, kind) list
    }
    class LLM {
        +chat(messages, json_mode) str
        +stream(messages) Iterator
    }
    class Supervisor {
        -dict _workers
        +start_all(llm_enabled)
        +get_worker(name) ManagedProcess
        +restart(name)
        +status() dict
        +stop_all()
    }
    class ManagedProcess {
        +str name
        +int port
        +Status status
        +start()
        +stop()
        +restart()
        +is_ready() bool
    }
    class Status {
        <<record>>
        +State state
        +int progress
        +int pid
        +int restarted_count
    }
    class State {
        <<enumeration>>
        NOT_INSTALLED
        STARTING
        LOADING
        READY
        CRASHED
        BACKOFF
        FAILED
        STOPPED
    }
    class EventBus {
        +publish(kind, source, payload)
        +subscribe()
        +unsubscribe(q)
    }
    class Event {
        <<record>>
        +str kind
        +str source
        +dict payload
        +float ts
    }
    class Hardware {
        <<record>>
        +float ram_total
        +float vram_total
        +bool gpu_available
        +safe_ram() float
        +safe_vram() float
    }
    class Store {
        +workers(ids) list
        +filter_ids(...) list
        +search_vectors(query_vec, ids, model_key, k) list
        +load_ctx(session_id) dict
        +save_ctx(session_id, ctx)
        +job_begin(...)
        +job_finish(...)
    }

    Embedder --|> _Remote
    LLM --|> _Remote
    _Remote ..> Supervisor : get_worker()
    Supervisor *-- ManagedProcess : owns
    ManagedProcess *-- Status
    Status --> State
    ManagedProcess ..> EventBus : publishes
    EventBus *-- Event : keeps
    Supervisor ..> Hardware : sizes
```

`Status`, `Event` and `Hardware` are dataclasses (shown as records). `Store` is the SQLite access layer and is used by the gateway modules rather than by the runtime classes, so it has no arrows here.

## Regenerating the diagrams

The JSON files in [`diagrams/src`](diagrams/src) are the inputs for [Archify](https://github.com/tt-a1i/archify):

```bash
node archify/bin/archify.mjs finalize erd      docs/diagrams/src/erd.json       docs/diagrams/erd.html       --quality standard
node archify/bin/archify.mjs finalize class    docs/diagrams/src/uml-class.json docs/diagrams/uml-class.html --quality standard
node archify/bin/archify.mjs finalize workflow docs/diagrams/src/workflow.json  docs/diagrams/workflow.html  --quality showcase
```

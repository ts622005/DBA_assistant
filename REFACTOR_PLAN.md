# DBA Assistant — Refactor Plan

**From an AWR summarizer to an evidence-grounded DBA copilot**

Version 1.0 · Target codebase: `C:\technorise` · Author: architecture review

---

## Implementation Priority

> This is the authoritative build order. Do **not** attempt to build everything at
> once, and do not start with agent frameworks. Solve the grounding/data problem
> first; add intelligence on top of facts. (The detailed, code-level phasing in §10
> follows this same order.)

**Phase 1 (Must Have)**
- AWR Parser
- Structured JSON extraction
- Timestamped storage
- Historical snapshot storage

**Phase 2**
- Evidence Verification Layer
- Confidence Scoring
- Question Classification

**Phase 3**
- RCA Agent
- Recommendation Agent

**Phase 4** — _(none for now)_
- Reserved. The Historical Comparison Engine is deferred to Phase 5 (future);
  see below. No work is scheduled here until a comparison/trending need is real.

**Phase 5 (Future)**
- Historical Comparison Engine
- Regression Detection (incl. plan-hash change detection)
- Execution Plans
- SQL Monitor Reports
- Conversational memory (bounded multi-turn context)
- Agent Orchestration

This prevents spending days building an agent framework before the actual problem
(facts + grounding) is solved.

---

## Agent Guidance

Do **not** implement a supervisor/orchestrator agent in v1.

Use a simple sequential pipeline:

```
Question
  → Question Classifier
  → Evidence Agent
  → RCA Agent
  → Recommendation Agent
```

Only introduce orchestration after multiple heterogeneous data sources exist
(AWR, ASH, SQL Monitor, DBMS_XPLAN, Incident History, etc.).

Do not reach for LangGraph / CrewAI-style frameworks because the brief says
"agentic." For v1 the four steps above are plain function calls in sequence; the
Metrics Agent and Comparison Engine are invoked inline as the RCA Agent needs them,
not coordinated by a separate supervisor. (Where §8 / §10 mention an
"orchestrator," read it as this thin sequential caller — **not** a framework — until
Phase 5.)

---

## Deliverables

This refactor must be delivered as an architect would deliver it — design and a
migration path, not just generated files. Provide:

1. Updated architecture diagram — §2.2
2. Database schema changes — §3.3
3. Vector DB schema changes — §3.1 (narrowed to prose), §10 layout
4. Structured JSON schema for parsed AWR data — §4
5. Agent interfaces — §8 (contracts per agent)
6. API contract changes — §5.3, §6.3, §11 (answer contract)
7. Step-by-step migration plan from current architecture — §10
8. Sample outputs for:
   - Root Cause Analysis — §11 (answer contract) + §11 behavioral test
   - Historical Comparison — §7.3 (comparison output contract)
   - Evidence Verification — §5.3 (`can_answer` contract)

---

## 1. Executive summary

The current system is described informally as "AWR → chunk → vector → LLM," but the
code is already further along than that. There is a real structured parser
(`awr_parser.py`), a derived-metrics layer (`derived_metrics.py`), a YAML-driven
rule engine that already emits findings with evidence and confidence
(`rule_engine.py` + `rules.yaml`), and an ADDM parser (`addm_parser.py`). The
problem is not that structure is missing — it is that **the structure never reaches
the answer**.

Three architectural facts explain every symptom you listed (generic
recommendations, invented full table scans, assumed index/plan problems):

1. **Parsed metrics are thrown away after ingest.** `pipeline.analyze()` produces a
   rich JSON object, but it is either written to a flat file (`awr_analysis.json`)
   or flattened into a handful of text documents for ChromaDB
   (`vector_store.build_documents`). There is **no queryable, timestamped metrics
   store**, so historical and time-comparison questions are impossible to answer
   from facts — the model is forced to guess.

2. **The answer path bypasses the deterministic layer.** `app.py` retrieves the top
   *k* text chunks and hands them straight to Llama 3 with a free-form system
   prompt. The rule engine's findings, confidences, and evidence are reduced to
   prose snippets the model may ignore or contradict. Nothing forces the model to
   answer *only* from retrieved facts, and nothing stops it from inventing an
   execution-plan cause that AWR never contained.

3. **There is no notion of "answerable."** A question like "Is SQL_ID X doing a full
   table scan?" cannot be answered from an AWR report alone (it needs an execution
   plan), but the system has no gate that recognizes this. It always answers.

This plan keeps everything that already works (the parser, derived metrics, rule
engine, ADDM normalization) and adds the four missing pieces that turn it into a
copilot: a **timestamped structured store**, an **evidence-verification gate**, a
**historical comparison engine**, and a **small set of specialized agents** that
together enforce: *state findings → cite evidence → state confidence → explain
reasoning → name missing evidence → recommend.*

### Bugs and gaps found during review (fix as part of the refactor)

- **Embedding-space mismatch (correctness bug).** `app.py` builds the store with
  `OllamaEmbedder(model="nomic-embed-text")` (768-dim), while the CLI path in
  `pipeline.py` / `vector_store.py` `__main__` uses `BGEEmbedder`
  (`BAAI/bge-small-en-v1.5`, 384-dim). Querying a collection that was embedded with
  a different model returns semantically meaningless neighbors. The embedder must
  be **pinned in one place** and the collection tagged with the model+dim it was
  built with.
- **No timestamps in the store.** `meta` carries `begin_time` / `end_time` as
  strings, but they are not parsed to datetimes nor indexed for range queries.
- **No plan hash, SQL stats, or segment data parsed.** `awr_parser` deliberately
  extracts only three Top-SQL categories and omits Plan Hash Value, Top SQL by
  Buffer Gets, Segments, and per-SQL execution counts needed for trend/plan-change
  detection.
- **`report_id` is referenced but never set.** `vector_store.build_documents` reads
  `meta["report_id"]` (to scope `/ask` to one report), but `pipeline.py` never
  populates it, so all reports share an id namespace and can overwrite each other.
- **Confidence exists at the rule level but never at the answer level.** The user
  sees raw model prose with no confidence, no evidence list, no "missing evidence."

---

## 2. Current state (grounded assessment)

### 2.1 What exists and should be kept

| Module | Responsibility | Keep / change |
|---|---|---|
| `awr_parser.py` | HTML → structured sections (header, snapshot, load profile, waits, time model, host/instance CPU, IO, memory, top SQL ×3) | **Keep**, extend (see §4.4) |
| `derived_metrics.py` | Raw → ratios + a flat `signals` namespace | **Keep**, extend with deltas |
| `rule_engine.py` + `rules.yaml` | Correlated, multi-signal findings with evidence + confidence | **Keep** — this is the anti-hallucination core; reuse, don't replace |
| `addm_parser.py` | Oracle ADDM prose → normalized findings tagged `source: oracle_addm` | **Keep** |
| `vector_store.py` | Embeddings + ChromaDB for findings / metrics summary / top SQL | **Keep for text only** (see §3) |
| `pipeline.py` | Orchestrates parse → derive → rules → assemble JSON | **Keep**, becomes the writer into the new structured store |
| `app.py` | Streamlit UI + retrieve-then-prompt Llama 3 | **Rework** — this is where grounding is lost |

### 2.2 The data-flow problem, drawn

```
CURRENT
  AWR HTML
    └─ pipeline.analyze() ─► rich JSON (header, snapshot, signals, findings…)
                               ├─► awr_analysis.json        (flat file, not queried)
                               └─► vector_store.ingest()    (flattened to ~12 text docs)
                                                                    │
  Question ───────────────────────────────────────────────────────┘
            └─ Chroma top-k text ─► Llama 3 (free prose) ─► answer (ungrounded)
```

The rich object exists for a few milliseconds and is then either filed away or
lossily compressed into prose. Numeric trend questions ("CPU over 30 days") have
nothing to read from; grounding questions have only prose to read from.

```
TARGET
  AWR HTML
    └─ pipeline.analyze() ─► canonical snapshot object
                               ├─► Structured Metrics Store  (SQLite/Postgres, timestamped)   ← facts & trends
                               └─► Vector Store              (ADDM text, RCA notes, recs)      ← semantics only
                                          │                                  │
  Question ─► Evidence Agent (can_answer? confidence? missing?) ─┐
                               │                                 │
                               ├─ Metrics Agent  (facts only, from structured store)
                               ├─ Comparison Engine (deltas across snapshots)
                               ├─ RCA Agent      (reasons over cited facts only)
                               └─ Recommendation Agent (actions, never invents plans)
                                          │
                                          └─► answer: findings · evidence · confidence · reasoning · missing · recs
```

---

## 3. Separate structured storage from vector search

The single most important change: **stop asking the LLM to infer numbers from text
chunks.** Numbers live in a structured store and are *retrieved*, not *recalled*.

### 3.1 Division of responsibility

| Goes in the **Structured Metrics Store** | Goes in the **Vector Store** |
|---|---|
| Snapshot times, DB time, DB CPU % | ADDM finding prose |
| All Top-SQL rows (CPU, elapsed, reads, buffer gets) | Textual recommendations |
| Wait events & wait classes (with values) | Historical RCA summaries you generate |
| Plan hash values, SQL execution counts | Free-text explanations / notes |
| Top segments, instance metrics | Anything where *meaning* matters more than *value* |
| Derived metrics & signals | |

Rule of thumb: **if a question contains a number, a date range, a comparison, or an
aggregate, it must be answerable from the structured store without the LLM.** The
LLM explains; it does not compute.

### 3.2 Storage technology

Use **SQLite** as the default (zero-ops, single file `dba_assistant.db`, ships with
Python, fine for one DBA / one team and millions of rows). Keep the schema
Postgres-compatible so you can lift-and-shift later for multi-user concurrency. Do
not invent a bespoke time-series format — a normal relational schema with indexed
timestamps makes time-based queries first-class, which is exactly the requirement.

### 3.3 Relational schema (SQLite, Postgres-compatible)

```sql
-- One row per ingested AWR report.
CREATE TABLE snapshot (
    snapshot_id     INTEGER PRIMARY KEY,          -- internal surrogate
    report_id       TEXT UNIQUE NOT NULL,         -- stable hash: db_id+inst+begin_snap+end_snap
    db_name         TEXT,
    db_id           TEXT,
    instance_number INTEGER,
    release         TEXT,
    is_rac          INTEGER,
    host            TEXT,
    begin_snap_id   INTEGER,
    end_snap_id     INTEGER,
    begin_time      TEXT NOT NULL,                -- ISO-8601, UTC-normalized
    end_time        TEXT NOT NULL,                -- ISO-8601
    elapsed_mins    REAL,
    db_time_mins    REAL,
    source_file     TEXT,
    ingested_at     TEXT NOT NULL
);
CREATE INDEX ix_snapshot_time ON snapshot (db_id, instance_number, begin_time);

-- Derived metrics + signals, one row per (snapshot, metric). Long/narrow so new
-- metrics never require a schema change.
CREATE TABLE metric (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id),
    name        TEXT NOT NULL,                    -- e.g. 'db_cpu_pct_db_time'
    value       REAL,
    unit        TEXT,                             -- 'pct','ms','per_sec','ratio','s'
    category    TEXT,                             -- 'CPU','IO','Parse','Commit'...
    PRIMARY KEY (snapshot_id, name)
);
CREATE INDEX ix_metric_name ON metric (name, snapshot_id);

-- Wait events / wait classes (kind distinguishes the two).
CREATE TABLE wait_event (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id),
    kind        TEXT NOT NULL,                    -- 'event' | 'class'
    name        TEXT NOT NULL,
    waits       REAL,
    time_s      REAL,
    avg_wait_ms REAL,
    pct_db_time REAL,
    wait_class  TEXT,
    PRIMARY KEY (snapshot_id, kind, name)
);

-- Top SQL — one row per (snapshot, sql_id, category). category lets one SQL appear
-- in CPU, elapsed, reads and buffer-gets lists.
CREATE TABLE sql_stat (
    snapshot_id    INTEGER NOT NULL REFERENCES snapshot(snapshot_id),
    sql_id         TEXT NOT NULL,
    category       TEXT NOT NULL,                 -- 'cpu_time','elapsed_time','physical_reads','buffer_gets'
    rank           INTEGER,
    cpu_time_s     REAL,
    elapsed_time_s REAL,
    physical_reads REAL,
    buffer_gets    REAL,
    executions     REAL,
    plan_hash_value TEXT,                         -- NULL until plan data is available
    module         TEXT,
    sql_text       TEXT,
    PRIMARY KEY (snapshot_id, sql_id, category)
);
CREATE INDEX ix_sqlstat_sqlid ON sql_stat (sql_id, snapshot_id);

-- Optional but recommended now: segments and ADDM findings as structured rows too,
-- so they participate in trend queries (vector store keeps the prose copy).
CREATE TABLE segment_stat (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id),
    owner       TEXT, segment_name TEXT, segment_type TEXT,
    category    TEXT,                              -- 'logical_reads','physical_reads','row_lock_waits'...
    value       REAL,
    PRIMARY KEY (snapshot_id, owner, segment_name, category)
);

CREATE TABLE addm_finding (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id),
    finding     TEXT, category TEXT, severity TEXT,
    impact_pct  REAL, recommendation TEXT,
    PRIMARY KEY (snapshot_id, finding)
);
```

Design notes:

- **`report_id` is a deterministic hash** of `db_id + instance + begin_snap +
  end_snap`. Re-ingesting the same report is idempotent (upsert), which fixes the
  current "reports overwrite each other" gap and makes ingest safe to retry.
- **`metric` is narrow (long format)** so adding a signal in `derived_metrics.py`
  needs no migration — it just appears as new rows. Every signal in the current
  flat `signals` dict maps to one `metric` row.
- **Timestamps are stored ISO-8601 and indexed.** This is what makes "last 30 days"
  and "between June 20 and June 23" first-class — they become `WHERE begin_time
  BETWEEN ? AND ?` queries, not LLM guesses.

### 3.4 Example time-based queries (now trivial and exact)

```sql
-- CPU trend over the last 30 days for one instance
SELECT s.begin_time, m.value AS db_cpu_pct
FROM snapshot s JOIN metric m ON m.snapshot_id = s.snapshot_id
WHERE m.name = 'db_cpu_pct_db_time'
  AND s.db_id = :db_id AND s.instance_number = :inst
  AND s.begin_time >= date('now','-30 days')
ORDER BY s.begin_time;

-- When did 'log file sync' first become significant (>10ms)?
SELECT MIN(s.begin_time) AS first_seen
FROM snapshot s JOIN wait_event w ON w.snapshot_id = s.snapshot_id
WHERE w.name = 'log file sync' AND w.avg_wait_ms > 10 AND s.db_id = :db_id;

-- Which SQL degraded most in CPU between two snapshots
SELECT a.sql_id,
       b.cpu_time_s - a.cpu_time_s              AS delta_cpu_s,
       (b.cpu_time_s - a.cpu_time_s) / NULLIF(a.cpu_time_s,0) * 100 AS pct_change
FROM sql_stat a JOIN sql_stat b
  ON a.sql_id = b.sql_id AND a.category='cpu_time' AND b.category='cpu_time'
WHERE a.snapshot_id = :prev AND b.snapshot_id = :curr
ORDER BY delta_cpu_s DESC;
```

### 3.5 New module: `metrics_store.py`

A thin data-access layer so nothing else writes SQL inline.

```python
class MetricsStore:
    def __init__(self, db_path="dba_assistant.db"): ...
    def upsert_snapshot(self, report: dict) -> int:        # returns snapshot_id; idempotent on report_id
    def metric_series(self, name, db_id, inst, start, end) -> list[tuple[str, float]]
    def snapshots_in_range(self, db_id, inst, start, end) -> list[dict]
    def sql_stat(self, sql_id, snapshot_ids=None) -> list[dict]
    def latest_snapshot(self, db_id, inst) -> dict | None
    def previous_snapshot(self, snapshot_id) -> dict | None
    def wait_events(self, snapshot_id) -> list[dict]
```

`pipeline.analyze_and_ingest()` is extended to call `metrics_store.upsert_snapshot()`
**and** `vector_store.ingest()` — structured facts to one, prose to the other.

---

## 4. The canonical snapshot object (structured parse target)

Before storing anything, `pipeline.analyze()` should emit one **canonical snapshot
object** — the contract every downstream component reads. It is a superset of what
`awr_parser` already returns, with the additions from §1's gap list and with
**parsed datetimes**. This is the JSON the user asked for, made concrete.

```jsonc
{
  "report_id": "ORCL_1_48210_48211",          // deterministic hash (NEW)
  "db_name": "ORCL", "db_id": "1547304573",
  "instance_number": 1, "release": "19.0.0.0.0", "is_rac": false,
  "host": "dbhost01",

  "snapshot_begin": "2026-06-23T10:00:00",     // ISO-8601, parsed (NEW: was a raw string)
  "snapshot_end":   "2026-06-23T11:00:00",
  "elapsed_mins": 60.02,
  "db_time_mins": 1380.16,

  "db_cpu_pct": 72.66,                          // promoted from derived_metrics for convenience
  "top_wait_event": "db file sequential read",

  "wait_events": [
    {"event":"db file sequential read","waits":1.2e6,"time_s":3300,
     "avg_wait_ms":2.7,"pct_db_time":39.8,"wait_class":"User I/O"}
  ],
  "wait_classes": [ {"wait_class":"User I/O","pct_db_time":41.2,"avg_wait_ms":2.6} ],

  "top_sql": {
    "cpu_time":      [ {"sql_id":"0m1p56431tx7t","rank":1,"cpu_time_s":3285,
                        "elapsed_time_s":3324,"executions":118000,
                        "plan_hash_value":"2853973971"} ],   // plan hash (NEW)
    "elapsed_time":  [ ... ],
    "physical_reads":[ ... ],
    "buffer_gets":   [ ... ]                                  // 4th category (NEW)
  },

  "top_segments": [ {"owner":"APP","segment_name":"ORDERS","category":"logical_reads","value":4.1e8} ], // NEW
  "instance_metrics": { "logical_reads_per_sec": 55595.3, "physical_reads_per_sec": 812.0 },

  "derived_metrics": { "db_cpu_pct_db_time": 72.66, "logical_to_physical_ratio": 68.4, "...": "..." },
  "signals":         { "...": "flat namespace consumed by the rule engine (unchanged)" },

  "addm_findings": [ {"finding":"...","impact_pct":31.0,"category":"IO","severity":"HIGH"} ],

  "findings": [ {"category":"IO","severity":"HIGH","finding":"I/O Bottleneck",
                 "evidence":["User I/O is 41.2% of DB time", "..."],
                 "recommendation":"...","confidence":0.83,
                 "rule_id":"io_bottleneck","source":"rule_engine"} ]
}
```

### 4.4 Parser extensions required

| Add to `awr_parser.py` | Why |
|---|---|
| **Plan Hash Value** column in `_parse_sql_table` | Required for plan-change detection (§7). It sits in the Top-SQL tables already. |
| **Top SQL by Buffer Gets** (`"sql by buffer gets"`) | The CPU-burn rule (`inefficient_logical_reads`) reasons about logical reads but has no SQL list to point at. |
| **Top Segments** section | Needed to attribute I/O / contention to objects, and for "which object" RCA. |
| **Parse `snapshot_begin/end` to `datetime`** | Time queries depend on real timestamps, not `"23-Jun-26 10:00:01"` strings. Handle the AWR `DD-Mon-YY HH24:MI:SS` format explicitly. |
| **`report_id` hash** in `pipeline.py` | Idempotent ingest; fixes the `meta["report_id"]` that `vector_store` already expects but never receives. |

These are additive — existing extraction is untouched, so current findings keep
working while the new fields fill in.

---

## 5. Evidence verification layer (the anti-hallucination gate)

This is the layer that produces *"I cannot determine because execution plans are
unavailable"* instead of *"it is probably an indexing problem."*

### 5.1 Principle

Every question is first classified by **what evidence it requires**, then checked
against **what evidence is present** in the structured + vector stores for the
relevant snapshot(s). The RCA and Recommendation agents are **only invoked if the
gate passes**. No evidence → no root cause. Ever.

### 5.2 Evidence requirements map (data, not code)

Keep this in a `evidence_requirements.yaml`, in the same spirit as `rules.yaml`:

```yaml
# question intent  ->  evidence that MUST be present to answer it
intents:
  full_table_scan:
    needs: [execution_plan]            # AWR alone CANNOT prove a FTS
    answerable_from_awr: false
    refusal: >
      Cannot determine from AWR alone whether SQL_ID {sql_id} performs a full
      table scan. Execution plan data (DBMS_XPLAN / SQL Monitor) is required.
  index_missing:
    needs: [execution_plan]
    answerable_from_awr: false
  why_query_slow:
    needs: [sql_stat]                  # partial: AWR shows cost, not cause
    answerable_from_awr: partial
    note: "AWR shows resource consumption and waits, not the plan operation."
  cpu_bottleneck:
    needs: [time_model, instance_cpu]
    answerable_from_awr: true
  io_bottleneck:
    needs: [wait_events, wait_classes]
    answerable_from_awr: true
  trend / comparison:
    needs: [two_or_more_snapshots]
    answerable_from_awr: true          # if >=2 snapshots exist in range
```

### 5.3 Output contract

The Evidence Agent returns exactly this shape (your spec, made precise):

```jsonc
{
  "can_answer": false,
  "answer_scope": "none",              // "none" | "partial" | "full"
  "confidence": 0.2,
  "evidence_present": ["sql_stat", "wait_events"],
  "missing_evidence": ["execution_plan"],
  "reason": "Full-table-scan detection requires an execution plan; AWR exposes resource use, not plan operations."
}
```

Classification can be a small, cheap LLM call **constrained to choose an intent
label from the YAML** (not free-form) — or pure keyword routing for the common
intents. Either way the *decision to refuse* is made by the rules + store contents,
not by the generating model.

---

## 6. Confidence scoring (answer-level)

Today confidence exists only inside individual rule findings. Promote it to the
**answer**, and make it a transparent function of evidence — never a number the LLM
makes up.

### 6.1 Three-tier classification

Every statement in an answer is tagged:

- **Confirmed** — directly read from the structured store (a measured value, a
  fired multi-signal rule, or an ADDM finding). High confidence.
- **Inferred** — a reasoned conclusion that combines confirmed facts but is not
  itself measured (e.g., "CPU is the bottleneck because DB CPU is 73% of DB time
  *and* AAS > #CPUs"). Medium confidence.
- **Speculative** — a possibility consistent with the data but unproven (e.g., "this
  *could* be a plan regression"). Low confidence, and only shown when explicitly
  hedged.

### 6.2 Scoring model

```
answer_confidence =
      w_evidence  * (evidence_present / evidence_required)
    + w_rules     * mean(confidence of fired rules used)
    + w_corrob    * corroboration_bonus     # ADDM and rule engine agree → +
    - w_missing   * (missing_evidence / evidence_required)
```

Bounded to [0, 1]; report the band, not false precision:

| Score | Band | Meaning |
|---|---|---|
| ≥ 0.80 | High | Confirmed by measured facts and/or agreeing sources |
| 0.55–0.79 | Medium | Inferred from confirmed facts; some evidence missing |
| 0.30–0.54 | Low | Partial evidence; treat as a lead, not a conclusion |
| < 0.30 | Insufficient | Refuse; state missing evidence |

### 6.3 Output contract (per answer)

```jsonc
{
  "confidence": 0.87,
  "band": "High",
  "evidence": ["DB CPU 72.66% of DB time", "Top SQL by CPU: 0m1p56431tx7t", "ADDM: 'Top SQL by CPU'"],
  "evidence_sources": ["metric", "sql_stat", "addm_finding"],
  "missing_evidence": [],
  "classification": "confirmed"
}
```

---

## 7. Historical comparison engine

A new module `comparison_engine.py` that reads the structured store and computes
**deltas between snapshots** — no LLM involved in the math.

### 7.1 Inputs

- A baseline snapshot and a current snapshot (explicit ids, or resolved from dates:
  "June 20" → nearest snapshot, "yesterday" → previous day's snapshot).
- Or a window (last N days) for trend extraction.

### 7.2 Detectors (thresholds in YAML, tunable per environment)

| Detector | Condition | Output |
|---|---|---|
| Metric regression | `abs(pct_change) ≥ threshold` for any `metric` | "DB CPU rose from 48% → 73% (+52%)" |
| Wait-event growth | wait class/event `pct_db_time` or `avg_wait_ms` up ≥ threshold | "User I/O avg read latency 2.6ms → 11.4ms" |
| New top SQL | `sql_id` present in current top-N, absent in baseline | "SQL abc123 entered Top SQL by CPU (rank 2, new)" |
| Dropped top SQL | present in baseline top-N, gone in current | context for "what improved" |
| SQL regression | per-`sql_id` `delta_cpu` / `delta_elapsed` / `delta_reads` ≥ threshold | "SQL abc123 CPU +240% vs previous snapshot" |
| **Plan change** | same `sql_id`, different `plan_hash_value` | "Plan changed from 12345 → 98765" (strong RCA signal) |
| Execution spike | `executions` up ≥ threshold (distinguishes "more work" from "slower work") | "abc123 executions 10k → 95k; per-exec cost flat → it's volume, not regression" |

### 7.3 Output contract

```jsonc
{
  "baseline": {"snapshot_id": 41, "begin_time": "2026-06-20T10:00:00"},
  "current":  {"snapshot_id": 44, "begin_time": "2026-06-23T10:00:00"},
  "changes": [
    {"type":"metric_regression","name":"db_cpu_pct_db_time","from":48.1,"to":72.66,
     "pct_change":51.1,"severity":"HIGH","direction":"worse"},
    {"type":"plan_change","sql_id":"2bfwrkh7ttm3n","from":"12345","to":"98765","severity":"HIGH"},
    {"type":"sql_regression","sql_id":"abc123","metric":"cpu_time_s","from":120,"to":408,
     "pct_change":240.0,"executions_pct_change":4.2,"severity":"HIGH"}
  ],
  "summary_facts": ["3 regressions, 1 plan change, 1 new top-SQL"]
}
```

The engine emits **facts**; the RCA agent turns the most severe changes into a
narrative. "Why is today worse than yesterday?" becomes: run comparison → feed the
ranked `changes` to the RCA agent → narrate with citations.

---

## 8. Agent design (only where it adds value)

Four specialized agents, each with a **narrow contract**. No generic "do
everything" agent. An **orchestrator** (extends `app.py`'s query path) sequences
them. Three of the four are mostly deterministic — only RCA and Recommendation use
the LLM, and both are constrained to cited evidence.

```
Question
   │
   ▼
Evidence Agent ──► can_answer? ──no──► refuse with missing_evidence  (STOP)
   │ yes
   ▼
Metrics Agent  ──► facts (structured store) ┐
Comparison Eng ──► deltas (if temporal)     ├─► evidence bundle
Vector retrieve──► ADDM/RCA prose           ┘
   │
   ▼
RCA Agent ──► root cause(s) + reasoning, each citing the bundle
   │
   ▼
Recommendation Agent ──► prioritized actions (never invents plan data)
   │
   ▼
Answer assembler ──► findings · evidence · confidence · reasoning · missing · recs
```

### 8.1 Evidence Agent

- **Responsibility:** decide answerability, identify missing data, assign initial
  confidence. (§5)
- **Reads:** the intent map + store contents for the target snapshot(s).
- **Writes:** the §5.3 contract. **Gates everything downstream.**
- **LLM use:** at most a constrained intent-classification call.

### 8.2 Metrics Agent

- **Responsibility:** retrieve structured + historical metrics and **return facts
  only**. No RCA, no recommendations.
- **Reads:** `metrics_store`.
- **Writes:** a flat evidence bundle of `(name, value, unit, source, snapshot_time)`.
- **LLM use:** none. Pure queries. This is what kills numeric hallucination.

### 8.3 RCA Agent

- **Responsibility:** determine root cause(s) from the evidence bundle, explain the
  reasoning chain, identify the contributing metrics.
- **Hard constraint:** **must cite** at least one item from the evidence bundle for
  every claim; any claim it cannot cite is dropped or demoted to *speculative*.
- **Reuses:** the fired `rules.yaml` findings as its backbone — the rule engine has
  already done correlated diagnosis; the agent narrates and ranks, it does not
  re-derive from raw chunks.
- **LLM use:** yes, with a system prompt that forbids introducing facts not in the
  bundle and requires the three-tier tag (confirmed/inferred/speculative).

### 8.4 Recommendation Agent

- **Responsibility:** generate actions, prioritize them, estimate impact/effort.
- **Hard constraint:** **never invents execution-plan information.** If a
  recommendation would require a plan ("add index on column X"), it must either be
  conditioned ("*if* the plan shows a full scan of ORDERS, …") or downgraded to
  "next step: capture `DBMS_XPLAN` for SQL_ID X to confirm."
- **Reuses:** `recommendation` text from fired rules and ADDM as priors.
- **LLM use:** yes, constrained to the confirmed root causes from the RCA agent.

### 8.5 Why this is "agentic only where useful"

The expensive, error-prone step (free generation) is fenced into two agents that
can only speak about cited facts. Everything that can be deterministic — fact
retrieval, delta computation, answerability — is deterministic. That is the
measurable value: fewer LLM calls on the numeric path, and the LLM is structurally
prevented from fabricating the things it currently fabricates.

---

## 9. Future extensions (design for them now)

The architecture should get *progressively smarter* as richer sources arrive, with
no rewrite. The key is that the evidence layer already speaks in terms of evidence
*types* — so adding a source means (a) a new parser, (b) new structured tables, and
(c) flipping `answerable_from_awr` for the intents that source unlocks.

| Future source | What it unlocks | How it plugs in |
|---|---|---|
| **SQL execution plans / `DBMS_XPLAN`** | The big one: full-table-scan detection, index-usage, join-order RCA — exactly the questions the system must currently refuse | New `plan` table keyed by `(sql_id, plan_hash_value)` with operations; intents `full_table_scan` / `index_missing` flip to `answerable_from_awr: true` |
| **SQL Monitor reports** | Per-execution actuals (rows, time per operation, parallelism) | Joins to `sql_stat` on `sql_id` + `plan_hash_value`; feeds RCA with operation-level evidence |
| **ASH reports** | Sub-snapshot, time-sampled session activity; pinpoints *when* and *who* | New `ash_sample` table; lets the comparison engine localize a regression to a time-of-day / session |
| **AWR baselines** | Statistical "normal" bands per metric | Stored as `metric_baseline (name, p50, p90, p95)`; confidence scoring compares current vs baseline instead of fixed thresholds |
| **Historical incident reports** | Past RCA + resolution narratives | Embedded into the **vector store**; RCA agent retrieves "we've seen this pattern before" priors |

**Extensibility contract:** every new source must (1) write structured rows for its
numbers, (2) write prose to the vector store for its narratives, and (3) declare the
evidence types it provides so the Evidence Agent can widen what is answerable. The
evidence-gating design means new sources *automatically* reduce the number of "I
cannot determine" refusals — the system visibly improves as data is added.

---

## 10. Phased migration plan

Each phase is independently shippable and leaves the system working. Ordered so the
highest-impact, lowest-risk change (grounding the answer path) lands first.

### Phase 0 — Fixes and foundations (low risk)
- Pin the embedder in **one** place; tag the Chroma collection with model + dim;
  rebuild the collection once so `app.py` and the CLI agree. *(fixes the
  embedding-space bug)*
- Add `report_id` hashing in `pipeline.py`. *(fixes overwrite + idempotency)*
- Parse `snapshot_begin/end` into real datetimes.

### Phase 1 — Structured metrics store
- Add `metrics_store.py` + the §3.3 schema.
- Extend `pipeline.analyze_and_ingest()` to write structured rows **and** prose.
- Backfill: re-run the two existing AWR HTML files through the new path.
- *Deliverable:* exact answers to "CPU over last 30 days" without the LLM.

### Phase 2 — Evidence gate + answer contract
- Add `evidence_requirements.yaml` + Evidence Agent.
- Rework `app.py`'s query path into the orchestrator (§8) and render the §11
  answer contract (findings / evidence / confidence / reasoning / missing / recs).
- *Deliverable:* the system now refuses full-table-scan questions correctly. This
  phase alone removes most reported hallucinations.

### Phase 3 — RCA + Recommendation agents
- Add the RCA Agent and Recommendation Agent as **plain sequential steps** (no
  orchestrator — see Agent Guidance), both constrained to cite the evidence bundle.
- Promote confidence to the answer level with the three-tier classification.
- *Deliverable:* every answer carries a confidence band, cited evidence, and
  cleanly refuses when the evidence gate fails.

### Phase 4 — (reserved / no work scheduled)
Deferred by decision: the Historical Comparison Engine is in Phase 5. Nothing is
planned for Phase 4 until a concrete comparison/trending need arises. (Plan-hash and
buffer-gets parsing were already delivered in Phase 1, so the data is ready when the
engine is eventually built.)

### Phase 5 — Future sources & comparison
- **Historical Comparison Engine + regression detection** — `comparison_engine.py`
  + thresholds YAML, date-resolution ("June 20", "yesterday") → snapshot lookup,
  plan-hash change + per-SQL regression. ("what changed between X and Y".)
- Execution-plan parsing (highest unlock — flips index/FTS intents to answerable),
  then SQL Monitor reports, AWR baselines, incident history.
- **Conversational memory** — bounded multi-turn context (carry last sql_id /
  intent / snapshot; feed only the last 1-2 turns to narration). Deliberately
  deferred; the deterministic gate/RCA stay stateless for grounding.

### Suggested module layout after refactor

```
awr_parser.py          # extended: plan hash, buffer-gets SQL, segments, datetimes
addm_parser.py         # unchanged
derived_metrics.py     # extended: emits metric rows
rule_engine.py         # unchanged (reused by RCA agent)
rules.yaml             # unchanged
metrics_store.py       # NEW  — structured/timestamped store (SQLite)
comparison_engine.py   # NEW  — snapshot deltas, plan-change, regressions
evidence_requirements.yaml  # NEW — intent → required evidence
agents/
  evidence_agent.py    # NEW  — answerability gate
  metrics_agent.py     # NEW  — facts only
  rca_agent.py         # NEW  — cited reasoning
  recommendation_agent.py  # NEW — cited actions
orchestrator.py        # NEW  — sequences agents, assembles answer
vector_store.py        # narrowed to prose: ADDM, RCA notes, recommendations
pipeline.py            # writes to BOTH stores; sets report_id
app.py                 # UI calls orchestrator; renders answer contract
```

---

## 11. Desired end-state — the answer contract

Every answer the copilot returns has the same six-part shape. This is the single
behavioral guarantee that distinguishes a copilot from a summarizer.

```jsonc
{
  "findings":    ["User I/O is the dominant bottleneck (41.2% of DB time)."],
  "evidence":    ["wait_class User I/O = 41.2% DB time",
                  "db file sequential read avg 11.4ms",
                  "Top SQL by physical reads: b6usrg82hwsa3"],
  "confidence":  {"score": 0.83, "band": "High", "classification": "confirmed"},
  "reasoning":   "User I/O dominates DB time and single-block read latency is >10ms, which the io_bottleneck rule and the ADDM 'User I/O' finding independently corroborate.",
  "missing_evidence": ["execution_plan — needed to confirm whether b6usrg82hwsa3 is doing a full scan vs. index range scan"],
  "recommendations": [
    {"action":"Capture DBMS_XPLAN for b6usrg82hwsa3 to confirm access path","priority":1,"requires":"execution_plan"},
    {"action":"Review storage latency; 11.4ms single-block reads indicate slow storage","priority":2,"impact":"high"}
  ]
}
```

### The behavioral test

For the three transcripts you shared, the refactored system must behave like this:

| Question (from transcripts) | Old behavior | Required new behavior |
|---|---|---|
| "Which SQL would benefit from indexing?" | Asserted indexing fixes for two `sql_id`s with no plan evidence | List the high-physical-read SQL as *candidates*, **confidence Medium**, with `missing_evidence: [execution_plan]` and a next step to capture the plan |
| "Are any SQL doing full table scans?" | Hedged but still speculated; even garbled metrics ("CPU usage of 'Nones'") | **Refuse cleanly:** "Cannot determine from AWR alone — execution plan data required," then offer to identify *candidates* by high physical reads |
| "Why might 2bfwrkh7ttm3n not use an index?" | Invented "excessive logical reads / plan issues," misattributed CPU% as physical I/O | State only what's measured (its rank, logical reads/sec), tag the index claim **speculative**, and list `execution_plan` + `plan_hash` history as the missing evidence |

The governing rule, restated: the system must prefer

> "I cannot determine because execution plans are unavailable."

over

> "It is probably an indexing problem."

whenever evidence is insufficient — and the evidence gate (§5) is the mechanism that
makes that the *default*, not a matter of prompt wording.

---

## Appendix A — Mapping your 8 requirements to this plan

| Your requirement | Addressed in |
|---|---|
| 1. Structured AWR parser → JSON | §4 (canonical object), §4.4 (parser extensions) — extends existing `awr_parser.py` |
| 2. Store historical data with timestamps | §3.3 schema (indexed ISO timestamps), §3.4 queries |
| 3. Separate structured storage from vector search | §3 (division of responsibility), §10 layout |
| 4. Evidence verification layer | §5 (gate, intent map, `can_answer` contract) |
| 5. Confidence scoring | §6 (answer-level, three-tier, scoring model) |
| 6. Historical comparison engine | §7 (`comparison_engine.py`, detectors incl. plan change) |
| 7. Agent design (Evidence/Metrics/RCA/Recommendation) | §8 (four constrained agents + orchestrator) |
| 8. Future extensions | §9 (plans, SQL Monitor, ASH, baselines, incidents) |

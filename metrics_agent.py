#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_agent.py
================
Metrics Agent (Phase 3) — FACTS ONLY.

Its single job is to retrieve structured metrics, waits, top SQL and fired
findings from the MetricsStore and assemble a citeable "evidence bundle". It
performs NO root-cause analysis and makes NO recommendations, and it never calls
an LLM. This is the component that keeps numbers out of the model's imagination:
the RCA and Recommendation agents may only reason over what this bundle contains.

Output bundle:
    {
      "snapshot":      {snapshot_id, begin_time, end_time, db_name},
      "metrics":       {name: {value, unit}},      # key metrics present
      "top_wait_classes": [...], "top_wait_events": [...],
      "top_sql":       [rows in the category relevant to the intent],
      "sql_focus":     {sql_id: [...rows across categories]} | None,
      "findings":      [structured fired findings with evidence],
      "facts":         ["DB CPU is 75.68% of DB time", ...]   # one-line citations
    }
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("metrics_agent")

# Which Top-SQL category is most relevant per intent.
_INTENT_SQL_CATEGORY = {
    "cpu_bottleneck": "cpu_time",
    "io_bottleneck": "physical_reads",
    "index_benefit": "physical_reads",
    "index_usage": "buffer_gets",
    "full_table_scan": "physical_reads",
    "parse_issue": "cpu_time",
    "commit_latency": "elapsed_time",
    "concurrency": "elapsed_time",
    "trend_comparison": "cpu_time",
    "general": "elapsed_time",
}

# Key metrics surfaced as facts, with human label + formatting.
_KEY_METRICS = [
    ("db_cpu_pct_db_time",        "DB CPU is {v}% of DB time"),
    ("user_io_pct_db_time",       "User I/O is {v}% of DB time"),
    ("average_active_sessions",   "Average active sessions: {v}"),
    ("num_cpus",                  "Host CPUs: {v}"),
    ("aas_to_cpu_ratio",          "Active-sessions-to-CPU ratio: {v}"),
    ("hard_parse_pct",            "Hard parses are {v}% of parses"),
    ("logical_to_physical_ratio", "Logical-to-physical read ratio: {v}"),
    ("log_file_sync_ms",          "log file sync averages {v} ms"),
    ("db_file_seq_read_ms",       "db file sequential read averages {v} ms"),
    ("top_wait_class_pct",        "Top wait class accounts for {v}% of DB time"),
]

_SQL_ID_RE = re.compile(r"\b[0-9a-z]{13}\b")


def extract_sql_id(question: str) -> str | None:
    """Pull an Oracle sql_id (13 lowercase alphanumerics) from the question."""
    m = _SQL_ID_RE.search(question or "")
    return m.group(0) if m else None


class MetricsAgent:
    def __init__(self, store):
        self.store = store

    def gather(self, question: str, intent: str = "general",
               snapshot_id: int | None = None) -> dict | None:
        snap = (self.store.conn.execute(
                    "SELECT * FROM snapshot WHERE snapshot_id=?", (snapshot_id,)).fetchone()
                if snapshot_id else None)
        if snap is None:
            latest = self.store.latest_snapshot()
            if not latest:
                return None
            snapshot_id = latest["snapshot_id"]
            snap = dict(latest)
        else:
            snap = dict(snap)

        metrics = self.store.metrics(snapshot_id)
        wait_classes = self.store.wait_events(snapshot_id, kind="class", limit=5)
        wait_events = self.store.wait_events(snapshot_id, kind="event", limit=5)
        category = _INTENT_SQL_CATEGORY.get(intent, "elapsed_time")
        top_sql = self.store.top_sql(snapshot_id, category, limit=5)
        # Ranked Top-SQL across the main categories so the RCA agent can name the
        # single biggest offender directly from structured data (not ADDM prose).
        top_sql_by = {cat: self.store.top_sql(snapshot_id, cat, limit=5)
                      for cat in ("cpu_time", "elapsed_time", "physical_reads", "buffer_gets")}
        findings = self.store.findings(snapshot_id)

        sql_focus = None
        sid_q = extract_sql_id(question)
        if sid_q:
            rows = self.store.sql_stat(sid_q, snapshot_ids=[snapshot_id])
            sql_focus = {"sql_id": sid_q, "rows": rows}

        facts = self._facts(metrics, wait_classes, top_sql, category, sql_focus)

        return {
            "snapshot": {
                "snapshot_id": snapshot_id,
                "begin_time": snap.get("begin_time"),
                "end_time": snap.get("end_time"),
                "db_name": snap.get("db_name"),
            },
            "intent": intent,
            "metrics": metrics,
            "top_wait_classes": wait_classes,
            "top_wait_events": wait_events,
            "sql_category": category,
            "top_sql": top_sql,
            "top_sql_by": top_sql_by,
            "sql_focus": sql_focus,
            "findings": findings,
            "facts": facts,
        }

    @staticmethod
    def _facts(metrics, wait_classes, top_sql, category, sql_focus) -> list[str]:
        facts: list[str] = []
        for name, tmpl in _KEY_METRICS:
            if name in metrics and metrics[name]["value"] is not None:
                facts.append(tmpl.format(v=_fmt(metrics[name]["value"])))
        if wait_classes:
            wc = wait_classes[0]
            if wc.get("pct_db_time") is not None:
                facts.append(f"Top wait class is '{wc['name']}' at "
                             f"{_fmt(wc['pct_db_time'])}% of DB time")
        label = category.replace("_", " ")
        for s in top_sql[:3]:
            metric_val = s.get(_CATEGORY_FIELD.get(category, "elapsed_time_s"))
            facts.append(f"Top SQL by {label}: {s['sql_id']} "
                         f"({label} {_fmt(metric_val)}, execs {_fmt(s.get('executions'))})")
        if sql_focus and sql_focus["rows"]:
            for r in sql_focus["rows"]:
                facts.append(f"{sql_focus['sql_id']} [{r['category']}]: "
                             f"cpu {_fmt(r.get('cpu_time_s'))}s, reads {_fmt(r.get('physical_reads'))}, "
                             f"buffer_gets {_fmt(r.get('buffer_gets'))}, execs {_fmt(r.get('executions'))}, "
                             f"plan_hash {r.get('plan_hash_value') or 'n/a'}")
        return facts


_CATEGORY_FIELD = {
    "cpu_time": "cpu_time_s",
    "elapsed_time": "elapsed_time_s",
    "physical_reads": "physical_reads",
    "buffer_gets": "buffer_gets",
}


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    return str(v)

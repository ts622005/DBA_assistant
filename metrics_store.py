#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_store.py
================
Structured, timestamped metrics store for the DBA Assistant (Phase 1).

Parsed AWR metrics are stored in a relational schema with INDEXED timestamps so
numeric, historical and time-comparison questions are answered from facts, not
from the LLM's memory:

    "CPU over the last 30 days"      -> SELECT ... WHERE begin_time >= ...
    "what degraded between X and Y"  -> JOIN sql_stat a, sql_stat b ...
    "when did log file sync spike"   -> SELECT MIN(begin_time) WHERE avg_wait_ms > ...

SQLite by default (zero-ops, single file), schema kept Postgres-compatible.

Design rules:
  * Numbers live here and are RETRIEVED, never recalled by the LLM.
  * Ingest is IDEMPOTENT on report_id: re-ingesting a report replaces its rows.
  * The `metric` table is long/narrow so adding a derived metric needs no schema
    change — it just appears as new rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

log = logging.getLogger("metrics_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id       TEXT UNIQUE NOT NULL,
    db_name         TEXT,
    db_id           TEXT,
    instance_number INTEGER,
    release         TEXT,
    is_rac          TEXT,
    host            TEXT,
    begin_snap_id   INTEGER,
    end_snap_id     INTEGER,
    begin_time      TEXT NOT NULL,
    end_time        TEXT NOT NULL,
    elapsed_mins    REAL,
    db_time_mins    REAL,
    source_file     TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_snapshot_time ON snapshot (db_id, instance_number, begin_time);

CREATE TABLE IF NOT EXISTS metric (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    value       REAL,
    unit        TEXT,
    category    TEXT,
    PRIMARY KEY (snapshot_id, name)
);
CREATE INDEX IF NOT EXISTS ix_metric_name ON metric (name, snapshot_id);

CREATE TABLE IF NOT EXISTS wait_event (
    snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    waits       REAL,
    time_s      REAL,
    avg_wait_ms REAL,
    pct_db_time REAL,
    wait_class  TEXT,
    PRIMARY KEY (snapshot_id, kind, name)
);

CREATE TABLE IF NOT EXISTS sql_stat (
    snapshot_id     INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    sql_id          TEXT NOT NULL,
    category        TEXT NOT NULL,
    rank            INTEGER,
    cpu_time_s      REAL,
    elapsed_time_s  REAL,
    physical_reads  REAL,
    buffer_gets     REAL,
    executions      REAL,
    plan_hash_value TEXT,
    module          TEXT,
    sql_text        TEXT,
    PRIMARY KEY (snapshot_id, sql_id, category)
);
CREATE INDEX IF NOT EXISTS ix_sqlstat_sqlid ON sql_stat (sql_id, snapshot_id);

CREATE TABLE IF NOT EXISTS segment_stat (
    snapshot_id  INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    owner        TEXT,
    segment_name TEXT,
    segment_type TEXT,
    category     TEXT,
    value        REAL,
    PRIMARY KEY (snapshot_id, owner, segment_name, category)
);

CREATE TABLE IF NOT EXISTS addm_finding (
    snapshot_id    INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    finding        TEXT,
    category       TEXT,
    severity       TEXT,
    impact_pct     REAL,
    recommendation TEXT,
    PRIMARY KEY (snapshot_id, finding)
);

CREATE TABLE IF NOT EXISTS finding (
    snapshot_id    INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
    source         TEXT,
    rule_id        TEXT,
    category       TEXT,
    severity       TEXT,
    finding        TEXT,
    evidence       TEXT,
    recommendation TEXT,
    confidence     REAL,
    impact_pct     REAL,
    PRIMARY KEY (snapshot_id, source, finding)
);
"""


def _unit_for(name: str) -> Optional[str]:
    n = name.lower()
    if n.endswith("_pct") or "pct" in n:
        return "pct"
    if n.endswith("_ms"):
        return "ms"
    if "per_sec" in n:
        return "per_sec"
    if "ratio" in n:
        return "ratio"
    if n.endswith("_s") or n.endswith("_seconds"):
        return "s"
    return None


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


class MetricsStore:
    """SQLite-backed structured/timestamped metrics store."""

    def __init__(self, db_path: str = "dba_assistant.db"):
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        log.info("MetricsStore ready at %s", db_path)

    def upsert_snapshot(self, report: dict) -> int:
        """Insert (or replace) one analysed AWR report. Idempotent on report_id."""
        meta = report.get("meta", {}) or {}
        report_id = meta.get("report_id")
        if not report_id:
            raise ValueError("report['meta']['report_id'] is required (set by pipeline).")

        begin = meta.get("begin_time_iso") or meta.get("begin_time") or ""
        end = meta.get("end_time_iso") or meta.get("end_time") or ""

        cur = self.conn.cursor()
        row = cur.execute("SELECT snapshot_id FROM snapshot WHERE report_id=?",
                          (report_id,)).fetchone()
        if row:
            cur.execute("DELETE FROM snapshot WHERE snapshot_id=?", (row["snapshot_id"],))

        cur.execute(
            "INSERT INTO snapshot (report_id, db_name, db_id, instance_number, release, "
            "is_rac, host, begin_snap_id, end_snap_id, begin_time, end_time, elapsed_mins, "
            "db_time_mins, source_file) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (report_id, meta.get("db_name"), meta.get("db_id"),
             _num(meta.get("instance_number")), meta.get("release"),
             str(meta.get("is_rac")) if meta.get("is_rac") is not None else None,
             meta.get("host"), _num(meta.get("begin_snap")), _num(meta.get("end_snap")),
             begin, end, _num(meta.get("elapsed_mins")), _num(meta.get("db_time_mins")),
             meta.get("source_file")),
        )
        sid = cur.lastrowid

        self._write_metrics(cur, sid, report)
        self._write_waits(cur, sid, report)
        self._write_sql(cur, sid, report)
        self._write_segments(cur, sid, report)
        self._write_addm(cur, sid, report)
        self._write_findings(cur, sid, report)

        self.conn.commit()
        log.info("Stored snapshot %s (report_id=%s).", sid, report_id)
        return sid

    def _write_metrics(self, cur, sid, report):
        merged = {}
        merged.update(report.get("meta", {}).get("signals", {}) or {})
        merged.update(report.get("derived_metrics", {}) or {})
        rows = [(sid, name, _num(val), _unit_for(name), None)
                for name, val in merged.items() if _num(val) is not None]
        cur.executemany(
            "INSERT OR REPLACE INTO metric (snapshot_id,name,value,unit,category) "
            "VALUES (?,?,?,?,?)", rows)

    def _write_waits(self, cur, sid, report):
        rows = []
        for ev in report.get("wait_events", []) or []:
            rows.append((sid, "event", ev.get("event"), _num(ev.get("waits")),
                         _num(ev.get("time_s")), _num(ev.get("avg_wait_ms")),
                         _num(ev.get("pct_db_time")), ev.get("wait_class")))
        for wc in report.get("wait_classes", []) or []:
            rows.append((sid, "class", wc.get("wait_class"), _num(wc.get("waits")),
                         _num(wc.get("total_wait_s")), _num(wc.get("avg_wait_ms")),
                         _num(wc.get("pct_db_time")), wc.get("wait_class")))
        cur.executemany(
            "INSERT OR REPLACE INTO wait_event "
            "(snapshot_id,kind,name,waits,time_s,avg_wait_ms,pct_db_time,wait_class) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [r for r in rows if r[2]])

    def _write_sql(self, cur, sid, report):
        rows = []
        for cat, items in (report.get("top_sql", {}) or {}).items():
            for rank, s in enumerate(items, start=1):
                rows.append((
                    sid, s.get("sql_id"), cat, rank,
                    _num(s.get("cpu_time")), _num(s.get("elapsed_time")),
                    _num(s.get("physical_reads")), _num(s.get("buffer_gets")),
                    _num(s.get("executions")), s.get("plan_hash_value"),
                    s.get("module"), s.get("sql_text"),
                ))
        cur.executemany(
            "INSERT OR REPLACE INTO sql_stat "
            "(snapshot_id,sql_id,category,rank,cpu_time_s,elapsed_time_s,"
            "physical_reads,buffer_gets,executions,plan_hash_value,module,sql_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [r for r in rows if r[1]])

    def _write_segments(self, cur, sid, report):
        rows = []
        for cat, items in (report.get("top_segments", {}) or {}).items():
            for s in items or []:
                rows.append((sid, s.get("owner") or "", s.get("segment_name") or "",
                             s.get("segment_type"), cat, _num(s.get("value"))))
        cur.executemany(
            "INSERT OR REPLACE INTO segment_stat "
            "(snapshot_id,owner,segment_name,segment_type,category,value) "
            "VALUES (?,?,?,?,?,?)",
            [r for r in rows if r[2]])

    def _write_addm(self, cur, sid, report):
        rows = []
        for f in report.get("findings", []) or []:
            if f.get("source") == "oracle_addm":
                rows.append((sid, f.get("finding"), f.get("category"),
                             f.get("severity"), _num(f.get("impact_pct")),
                             f.get("recommendation")))
        cur.executemany(
            "INSERT OR REPLACE INTO addm_finding "
            "(snapshot_id,finding,category,severity,impact_pct,recommendation) "
            "VALUES (?,?,?,?,?,?)",
            [r for r in rows if r[1]])

    def _write_findings(self, cur, sid, report):
        rows = []
        for f in report.get("findings", []) or []:
            rows.append((
                sid, f.get("source"), f.get("rule_id"), f.get("category"),
                f.get("severity"), f.get("finding"),
                json.dumps(f.get("evidence") or []),
                f.get("recommendation"), _num(f.get("confidence")),
                _num(f.get("impact_pct")),
            ))
        cur.executemany(
            "INSERT OR REPLACE INTO finding "
            "(snapshot_id,source,rule_id,category,severity,finding,evidence,"
            "recommendation,confidence,impact_pct) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [r for r in rows if r[5]])

    # ── queries (facts only — what the Metrics Agent calls) ───────────────────

    def metric_series(self, name, db_id=None, instance_number=None, start=None, end=None):
        sql = ["SELECT s.begin_time, s.end_time, m.value, m.unit "
               "FROM snapshot s JOIN metric m ON m.snapshot_id = s.snapshot_id "
               "WHERE m.name = ?"]
        args = [name]
        if db_id is not None:
            sql.append("AND s.db_id = ?"); args.append(db_id)
        if instance_number is not None:
            sql.append("AND s.instance_number = ?"); args.append(instance_number)
        if start is not None:
            sql.append("AND s.begin_time >= ?"); args.append(start)
        if end is not None:
            sql.append("AND s.begin_time <= ?"); args.append(end)
        sql.append("ORDER BY s.begin_time")
        return [dict(r) for r in self.conn.execute(" ".join(sql), args).fetchall()]

    def snapshots_in_range(self, db_id=None, instance_number=None, start=None, end=None):
        sql = ["SELECT * FROM snapshot WHERE 1=1"]
        args = []
        if db_id is not None:
            sql.append("AND db_id = ?"); args.append(db_id)
        if instance_number is not None:
            sql.append("AND instance_number = ?"); args.append(instance_number)
        if start is not None:
            sql.append("AND begin_time >= ?"); args.append(start)
        if end is not None:
            sql.append("AND begin_time <= ?"); args.append(end)
        sql.append("ORDER BY begin_time")
        return [dict(r) for r in self.conn.execute(" ".join(sql), args).fetchall()]

    def sql_stat(self, sql_id, snapshot_ids=None):
        sql = ["SELECT ss.*, s.begin_time FROM sql_stat ss "
               "JOIN snapshot s ON s.snapshot_id = ss.snapshot_id WHERE ss.sql_id = ?"]
        args = [sql_id]
        if snapshot_ids:
            ph = ",".join("?" * len(snapshot_ids))
            sql.append("AND ss.snapshot_id IN (" + ph + ")")
            args.extend(snapshot_ids)
        sql.append("ORDER BY s.begin_time, ss.category")
        return [dict(r) for r in self.conn.execute(" ".join(sql), args).fetchall()]

    def wait_events(self, snapshot_id, kind="event", limit=10):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM wait_event WHERE snapshot_id=? AND kind=? "
            "ORDER BY pct_db_time DESC LIMIT ?", (snapshot_id, kind, limit)).fetchall()]

    def findings(self, snapshot_id):
        rows = self.conn.execute(
            "SELECT * FROM finding WHERE snapshot_id=? "
            "ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 "
            "WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 ELSE 4 END, confidence DESC",
            (snapshot_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["evidence"] = json.loads(d.get("evidence") or "[]")
            except (ValueError, TypeError):
                d["evidence"] = []
            out.append(d)
        return out

    def top_sql(self, snapshot_id, category, limit=5):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sql_stat WHERE snapshot_id=? AND category=? "
            "ORDER BY rank LIMIT ?", (snapshot_id, category, limit)).fetchall()]

    def metrics(self, snapshot_id, names=None):
        if names:
            ph = ",".join("?" * len(names))
            rows = self.conn.execute(
                "SELECT name,value,unit FROM metric WHERE snapshot_id=? AND name IN (" + ph + ")",
                (snapshot_id, *names)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT name,value,unit FROM metric WHERE snapshot_id=?",
                (snapshot_id,)).fetchall()
        return {r["name"]: {"value": r["value"], "unit": r["unit"]} for r in rows}

    def latest_snapshot(self, db_id=None, instance_number=None):
        sql = ["SELECT * FROM snapshot WHERE 1=1"]
        args = []
        if db_id is not None:
            sql.append("AND db_id = ?"); args.append(db_id)
        if instance_number is not None:
            sql.append("AND instance_number = ?"); args.append(instance_number)
        sql.append("ORDER BY begin_time DESC LIMIT 1")
        row = self.conn.execute(" ".join(sql), args).fetchone()
        return dict(row) if row else None

    def previous_snapshot(self, snapshot_id):
        this = self.conn.execute(
            "SELECT * FROM snapshot WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if not this:
            return None
        row = self.conn.execute(
            "SELECT * FROM snapshot WHERE db_id IS ? AND instance_number IS ? "
            "AND begin_time < ? ORDER BY begin_time DESC LIMIT 1",
            (this["db_id"], this["instance_number"], this["begin_time"])).fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Ingest a pipeline analysis JSON into the structured store.")
    ap.add_argument("analysis_json")
    ap.add_argument("--db", default="dba_assistant.db")
    args = ap.parse_args()
    store = MetricsStore(args.db)
    store.upsert_snapshot(json.load(open(args.analysis_json)))

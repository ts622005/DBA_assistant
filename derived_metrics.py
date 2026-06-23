#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derived_metrics.py
==================
Derived Metrics Layer for the AI-powered DBA Assistant.

After raw extraction (awr_parser), this stage computes second-order metrics
that the Rule Engine correlates. It returns two things:

  derived_metrics : the compact ratios that go into the final JSON
                    (cpu_ratio, io_ratio, parse_ratio, ...).
  signals         : a FLAT, normalised namespace consumed by the rule engine,
                    so rules can reference clean names (e.g. db_cpu_pct_db_time)
                    instead of digging through nested raw sections.

Every computation is null-safe: missing inputs yield None rather than raising,
because real AWR reports omit sections depending on edition / RAC / version.
"""

from __future__ import annotations

from typing import Optional


# ── safe arithmetic ─────────────────────────────────────────────────────────

def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        if b == 0:
            return None
        return round(a / b, 4)
    except (TypeError, ZeroDivisionError):
        return None


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    r = _div(a, b)
    return round(r * 100, 2) if r is not None else None


# ── fuzzy lookups against the parsed report ─────────────────────────────────

def _lp(report: dict, *needles: str, field: str = "per_second") -> Optional[float]:
    """Get a Load Profile value by fuzzy key match (all needles must appear)."""
    lp = report.get("load_profile", {}) or {}
    for key, val in lp.items():
        kl = key.lower()
        if all(n in kl for n in needles):
            if isinstance(val, dict):
                return val.get(field)
            return val
    return None


def _tm(report: dict, *needles: str, field: str = "time_s") -> Optional[float]:
    """Get a Time Model value by fuzzy key match."""
    tm = report.get("time_model", {}) or {}
    for key, val in tm.items():
        kl = key.lower()
        if all(n in kl for n in needles):
            if isinstance(val, dict):
                return val.get(field)
            return val
    return None


def _ie(report: dict, *needles: str) -> Optional[float]:
    """Get an Instance Efficiency ratio by fuzzy key match."""
    ie = report.get("instance_efficiency", {}) or {}
    for key, val in ie.items():
        if all(n in key.lower() for n in needles):
            return val
    return None


def _event(report: dict, name_contains: str, field: str) -> Optional[float]:
    """Get a field from the first wait event whose name contains a substring."""
    for ev in report.get("wait_events", []) or []:
        if name_contains.lower() in str(ev.get("event", "")).lower():
            return ev.get(field)
    return None


def _wait_class(report: dict, name: str, field: str = "pct_db_time") -> Optional[float]:
    for wc in report.get("wait_classes", []) or []:
        if str(wc.get("wait_class", "")).lower() == name.lower():
            return wc.get(field)
    return None


def _user_io_wait_seconds(report: dict) -> Optional[float]:
    """Total User I/O wait time (s): prefer wait-class rollup, else sum events."""
    cls = _wait_class(report, "User I/O", "total_wait_s")
    if cls is not None:
        return cls
    total = None
    for ev in report.get("wait_events", []) or []:
        if str(ev.get("wait_class", "")).lower() == "user i/o" and ev.get("time_s") is not None:
            total = (total or 0) + ev["time_s"]
    return total


def _db_time_seconds(report: dict) -> Optional[float]:
    """DB time in seconds — Time Model first, else snapshot minutes * 60."""
    t = _tm(report, "db time")
    if t is not None:
        return t
    mins = (report.get("snapshot", {}) or {}).get("DB Time (mins)")
    return round(mins * 60, 2) if mins is not None else None


def _db_cpu_seconds(report: dict) -> Optional[float]:
    """DB CPU in seconds — Time Model 'DB CPU'."""
    return _tm(report, "db cpu")


# ════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════

def compute(report: dict) -> tuple[dict, dict]:
    """Return (derived_metrics, signals) for a parsed AWR report."""

    # --- raw inputs (null-safe) ---
    db_cpu_s = _db_cpu_seconds(report)
    db_time_s = _db_time_seconds(report)
    user_io_s = _user_io_wait_seconds(report)

    hard_parses = _lp(report, "hard parse")
    total_parses = _lp(report, "parse") if _lp(report, "parse") is not None else _lp(report, "parses")
    # avoid matching "hard parses" when we wanted total parses:
    total_parses = None
    lp = report.get("load_profile", {}) or {}
    for k, v in lp.items():
        kl = k.lower()
        if "parse" in kl and "hard" not in kl:
            total_parses = v.get("per_second") if isinstance(v, dict) else v
            break

    logical_reads = _lp(report, "logical read")
    physical_reads = _lp(report, "physical read")
    executes = _lp(report, "execute")
    db_cpu_per_sec = _lp(report, "db cpu")
    db_time_per_sec = _lp(report, "db time")  # == Average Active Sessions

    # --- required derived metrics ---
    cpu_ratio = _div(db_cpu_s, db_time_s)
    io_ratio = _div(user_io_s, db_time_s)
    parse_ratio = _div(hard_parses, total_parses)
    logical_to_physical_ratio = _div(logical_reads, physical_reads)
    cpu_per_exec = _div(db_cpu_per_sec, executes)
    reads_per_exec = _div(physical_reads, executes)

    derived_metrics = {
        "cpu_ratio": cpu_ratio,
        "io_ratio": io_ratio,
        "parse_ratio": parse_ratio,
        "logical_to_physical_ratio": logical_to_physical_ratio,
        "cpu_per_exec": cpu_per_exec,
        "reads_per_exec": reads_per_exec,
        # useful extras (clearly derived):
        "db_cpu_pct_db_time": round(cpu_ratio * 100, 2) if cpu_ratio is not None else None,
        "user_io_pct_db_time": round(io_ratio * 100, 2) if io_ratio is not None else None,
        "hard_parse_pct": round(parse_ratio * 100, 2) if parse_ratio is not None else None,
        "average_active_sessions": db_time_per_sec,
        "db_cpu_per_sec": db_cpu_per_sec,
    }

    # --- top wait class (highest %DB time) ---
    top_wc, top_wc_pct = None, None
    for wc in report.get("wait_classes", []) or []:
        p = wc.get("pct_db_time")
        if p is not None and (top_wc_pct is None or p > top_wc_pct):
            top_wc, top_wc_pct = wc.get("wait_class"), p

    # --- %DB time accounted for by top events (CPU-starvation tell) ---
    pct_accounted = None
    for ev in report.get("wait_events", []) or []:
        if ev.get("pct_db_time") is not None:
            pct_accounted = (pct_accounted or 0) + ev["pct_db_time"]

    host = report.get("host_cpu", {}) or {}
    mem = report.get("memory", {}) or {}

    def _mem_pct_host() -> Optional[float]:
        for k, v in mem.items():
            if "host" in k.lower() and "mem" in k.lower():
                if isinstance(v, dict):
                    return v.get("end") or v.get("begin")
                return v
        return None

    num_cpus = host.get("CPUs")

    # --- flat signal namespace for the rule engine ---
    signals = {
        # CPU
        "db_cpu_pct_db_time": derived_metrics["db_cpu_pct_db_time"],
        "db_cpu_per_sec": db_cpu_per_sec,
        "num_cpus": num_cpus,
        "num_cores": host.get("Cores"),
        "host_cpu_idle_pct": host.get("%Idle"),
        "host_cpu_user_pct": host.get("%User"),
        "host_cpu_busy_pct": (round(100 - host["%Idle"], 2) if host.get("%Idle") is not None else None),
        "average_active_sessions": db_time_per_sec,
        "aas_to_cpu_ratio": _div(db_time_per_sec, num_cpus),
        "pct_db_time_accounted": round(pct_accounted, 2) if pct_accounted is not None else None,
        # I/O
        "user_io_pct_db_time": (
            _wait_class(report, "User I/O", "pct_db_time")
            if _wait_class(report, "User I/O", "pct_db_time") is not None
            else derived_metrics["user_io_pct_db_time"]
        ),
        "logical_to_physical_ratio": logical_to_physical_ratio,
        "physical_reads_per_sec": physical_reads,
        "logical_reads_per_sec": logical_reads,
        "reads_per_exec": reads_per_exec,
        "db_file_seq_read_ms": _event(report, "db file sequential read", "avg_wait_ms"),
        "db_file_scat_read_ms": _event(report, "db file scattered read", "avg_wait_ms"),
        "direct_path_read_ms": _event(report, "direct path read", "avg_wait_ms"),
        "user_io_class_avg_ms": _wait_class(report, "User I/O", "avg_wait_ms"),
        # Parse / SQL reuse
        "hard_parse_ratio": parse_ratio,
        "hard_parse_pct": derived_metrics["hard_parse_pct"],
        "hard_parses_per_sec": hard_parses,
        "execute_to_parse_pct": _ie(report, "execute to parse"),
        "soft_parse_pct": _ie(report, "soft parse"),
        "parse_cpu_to_parse_elapsed_pct": _ie(report, "parse cpu"),
        # Commit / redo
        "log_file_sync_ms": _event(report, "log file sync", "avg_wait_ms"),
        "log_file_sync_pct": _event(report, "log file sync", "pct_db_time"),
        "commit_class_pct": _wait_class(report, "Commit"),
        # Concurrency / contention
        "concurrency_class_pct": _wait_class(report, "Concurrency"),
        "concurrency_class_avg_ms": _wait_class(report, "Concurrency", "avg_wait_ms"),
        "application_class_pct": _wait_class(report, "Application"),
        "cluster_class_pct": _wait_class(report, "Cluster"),
        "configuration_class_pct": _wait_class(report, "Configuration"),
        # Memory / cache
        "library_hit_pct": _ie(report, "library hit") or _ie(report, "library cache"),
        "buffer_hit_pct": _ie(report, "buffer hit"),
        "buffer_nowait_pct": _ie(report, "buffer nowait"),
        "mem_pct_host": _mem_pct_host(),
        # Wait-class summary
        "top_wait_class": top_wc,
        "top_wait_class_pct": top_wc_pct,
        "log_files_per_sec": None,
    }

    return derived_metrics, signals

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awr_parser.py
=============
Oracle AWR HTML report parser for the AI-powered DBA Assistant.

Design intent (see project architecture):
    AWR HTML -> Structured Metrics -> Derived Metrics -> Rule Engine
             -> Findings -> Embeddings -> Vector DB -> AI DBA Assistant

This module is *extraction only*. It deliberately parses ONLY the sections
that carry diagnostic value, rather than every table in the report. Derived
metrics, the rule engine and findings live in separate modules so each stage
of the pipeline stays single-responsibility and independently testable.

Sections extracted:
    header                 - DB identity + release + host
    snapshot               - begin/end snap ids, times, elapsed, DB time
    load_profile           - per-second / per-transaction counters
    instance_efficiency    - cache/parse hit ratios
    wait_events            - Top Foreground wait events (list)
    wait_classes           - Foreground wait-class rollup (list)
    time_model             - Time Model statistics (DB time breakdown)
    host_cpu               - Host CPU section (cores, %user/%sys/%idle, load)
    instance_cpu           - Instance CPU (%busy / %total CPU)
    io_profile             - IO Profile section (read/write reqs & MB per sec)
    memory                 - SGA / PGA / memory statistics
    top_sql                - Top 10 by Elapsed, CPU, Physical Reads (compact)

ADDM findings are intentionally NOT parsed here: a separate ADDM parser feeds
the RCA agent. This keeps the AWR metric surface clean and compact.

Compatible with Oracle 12c / 19c / 21c AWR HTML format.
"""

from __future__ import annotations

import re
import io
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("awr_parser")

# How many rows to keep per Top-SQL list, and how far to truncate SQL text.
TOP_SQL_LIMIT = 10
SQL_TEXT_TRUNCATE = 120


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

_UNIT_STRIP_RE = re.compile(r"[^\d.\-]")


def clean_number(raw) -> Optional[float]:
    """Convert an AWR cell string to float.

    Handles '1,234.56', '(123)' (negative), 'N/A', '95.4%', '1.1s', '-'.
    Returns None when the value is not numeric.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in ("N/A", "NA", "-", ".", "--"):
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").rstrip("%").replace(",", "")
    s = _UNIT_STRIP_RE.sub("", s)
    if not s or s in (".", "-"):
        return None
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _text(tag) -> str:
    return tag.get_text(separator=" ", strip=True) if tag else ""


# AWR snapshot time formats vary by version/locale. Try the common ones.
_AWR_TIME_FORMATS = (
    "%d-%b-%y %H:%M:%S",   # 23-Jun-26 10:00:01
    "%d-%b-%Y %H:%M:%S",   # 23-Jun-2026 10:00:01
    "%d-%b-%y %H:%M",      # 23-Jun-26 10:00
    "%d-%b-%Y %H:%M",
    "%d-%b-%y",            # 23-Jun-26
)


def to_iso(raw) -> Optional[str]:
    """Parse an AWR snapshot time string to ISO-8601, or None if unparseable.

    Time-based queries (trends, comparisons) depend on real timestamps rather than
    locale-formatted strings, so the parser normalises them once at extraction.
    """
    if raw is None:
        return None
    s = re.sub(r"\s+", " ", str(raw)).strip()
    if not s or s.lower() in ("nan", "n/a", "-"):
        return None
    from datetime import datetime
    for fmt in _AWR_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return None


def _find_table_by_summary(soup: BeautifulSoup, fragment: str) -> Optional[Tag]:
    """Find a table whose `summary` attribute contains `fragment`."""
    frag = fragment.lower()
    for tbl in soup.find_all("table"):
        if frag in (tbl.get("summary") or "").lower():
            return tbl
    return None


def _find_table_by_header_text(soup: BeautifulSoup, fragment: str) -> Optional[Tag]:
    """Fallback: find the table following a heading containing `fragment`."""
    frag = fragment.lower()
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "b", "th"]):
        if frag in _text(heading).lower():
            nxt = heading.find_next("table")
            if nxt is not None:
                return nxt
    return None


def _table_to_df(tbl: Tag) -> pd.DataFrame:
    try:
        dfs = pd.read_html(io.StringIO(str(tbl)), header=0)
        if dfs:
            df = dfs[0]
            df.columns = [str(c).strip() for c in df.columns]
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Metric"})
            return df
    except Exception as exc:  # noqa: BLE001
        log.debug("pd.read_html failed: %s", exc)
    return pd.DataFrame()


def _get_table_df(soup: BeautifulSoup, summary_frag: str,
                  heading_frag: str = "") -> pd.DataFrame:
    """Try summary-attribute lookup first, then a heading-text fallback."""
    tbl = _find_table_by_summary(soup, summary_frag)
    if tbl is None:
        tbl = _find_table_by_header_text(soup, heading_frag or summary_frag)
    if tbl is None:
        log.debug("Section not found: '%s'", summary_frag)
        return pd.DataFrame()
    return _table_to_df(tbl)


def _col(df: pd.DataFrame, *needles: str) -> Optional[str]:
    """Return the first column whose lowercased name contains ALL needles."""
    for c in df.columns:
        cl = str(c).lower()
        if all(n in cl for n in needles):
            return c
    return None


# ════════════════════════════════════════════════════════════════════════════
# Section parsers
# ════════════════════════════════════════════════════════════════════════════

def parse_header(soup: BeautifulSoup) -> dict:
    """DB identity: name, id, instance, release, RAC flag, host, platform."""
    result: dict = {}
    df = _get_table_df(soup, "database instance information", "DB Name")
    if df.empty:
        df = _get_table_df(soup, "this table displays database", "DB Name")

    if not df.empty:
        # AWR header is a single row keyed by columns.
        row = df.iloc[0]
        mapping = {
            "DB Name": ("db name",),
            "DB Id": ("db id",),
            "Instance": ("instance",),
            "Inst Num": ("inst num",),
            "Release": ("release",),
            "RAC": ("rac",),
            "CDB": ("cdb",),
            "Host": ("host",),
            "Platform": ("platform",),
        }
        for label, needles in mapping.items():
            c = _col(df, *needles)
            if c is not None:
                result[label] = str(row[c]).strip()

    # Fallback: scan label/value cell pairs across small header tables.
    if "DB Name" not in result:
        for tbl in soup.find_all("table"):
            cells = [_text(c) for c in tbl.find_all(["td", "th"])]
            for i, cell in enumerate(cells):
                lc = cell.lower()
                if i + 1 < len(cells):
                    if "db name" in lc and "DB Name" not in result:
                        result["DB Name"] = cells[i + 1].strip()
                    elif lc.strip() == "release" and "Release" not in result:
                        result["Release"] = cells[i + 1].strip()
            if "DB Name" in result:
                break
    return result


def _first_number(row):
    """Return the first numeric value in a row (handles label/value misalignment
    where the number sits in a column other than index 1, e.g. '1,380.16 (mins)')."""
    for cell in row:
        v = clean_number(cell)
        if v is not None:
            return v
    return None


def parse_snapshot(soup: BeautifulSoup) -> dict:
    """Snapshot range: begin/end snap ids and times, elapsed & DB time (mins)."""
    result: dict = {}
    df = _get_table_df(soup, "snapshot information", "Snap Id")
    if df.empty:
        df = _get_table_df(soup, "this table displays snapshot", "Snap Id")
    if df.empty:
        return result

    snap_col = _col(df, "snap", "id") or _col(df, "snap")
    time_col = _col(df, "snap", "time") or _col(df, "time")
    sess_col = _col(df, "session")
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip().lower()
        if "begin" in label:
            if snap_col is not None:
                result["Begin Snap Id"] = clean_number(row.get(snap_col))
            if time_col is not None:
                result["Begin Time"] = str(row.get(time_col)).strip()
                result["Begin Time ISO"] = to_iso(row.get(time_col))
            if sess_col is not None:
                result["Begin Sessions"] = clean_number(row.get(sess_col))
        elif "end" in label:
            if snap_col is not None:
                result["End Snap Id"] = clean_number(row.get(snap_col))
            if time_col is not None:
                result["End Time"] = str(row.get(time_col)).strip()
                result["End Time ISO"] = to_iso(row.get(time_col))
        elif "elapsed" in label:
            result["Elapsed (mins)"] = _first_number(row)
        elif "db time" in label:
            result["DB Time (mins)"] = _first_number(row)
    return result


def parse_load_profile(soup: BeautifulSoup) -> dict:
    """Load Profile: per-second & per-transaction for each counter."""
    result: dict = {}
    df = _get_table_df(soup, "load profile", "Load Profile")
    if df.empty:
        return result

    metric_col = _col(df, "metric") or df.columns[0]
    per_sec_col = _col(df, "per second")
    per_txn_col = _col(df, "per transaction")
    for _, row in df.iterrows():
        metric = str(row.get(metric_col, "")).strip().rstrip(":")
        if not metric or metric.lower() in ("metric", "nan", ""):
            continue
        result[metric] = {
            "per_second": clean_number(row.get(per_sec_col)) if per_sec_col else None,
            "per_transaction": clean_number(row.get(per_txn_col)) if per_txn_col else None,
        }
    return result


def parse_instance_efficiency(soup: BeautifulSoup) -> dict:
    """Instance Efficiency Percentages (buffer/library hit, parse ratios...)."""
    result: dict = {}
    df = _get_table_df(soup, "instance efficiency", "Instance Efficiency")
    if df.empty:
        return result
    # Laid out as side-by-side metric/value pairs across each row.
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        for i in range(0, len(vals) - 1, 2):
            metric = vals[i].rstrip(":").strip()
            value = clean_number(vals[i + 1])
            if metric and metric.lower() not in ("nan", "") and value is not None:
                result[metric] = value
    return result


def parse_wait_events(soup: BeautifulSoup) -> list:
    """Top Foreground wait events (also captures 'DB CPU' pseudo-event)."""
    events: list = []
    for frag in (
        "top 10 foreground events",
        "top 5 timed foreground events",
        "this table displays top 10 wait events",
        "foreground wait events",
        "top 10 timed events",
    ):
        df = _get_table_df(soup, frag, frag)
        if not df.empty:
            break
    else:
        return events

    ev_c = _col(df, "event")
    waits_c = _col(df, "waits")
    time_c = _col(df, "time") if _col(df, "time") and "avg" not in (_col(df, "time") or "") else _col(df, "total", "wait")
    # prefer an explicit "Time(s)"/"Total Wait Time" column over avg
    time_c = _col(df, "total", "wait") or _col(df, "time(s)") or _col(df, "time")
    avg_c = _col(df, "avg", "wait") or _col(df, "wait", "ms") or _col(df, "(ms)")
    pct_c = _col(df, "%", "db") or _col(df, "db", "time")
    class_c = _col(df, "wait", "class") or _col(df, "class")

    for _, row in df.iterrows():
        event = str(row.get(ev_c, "")).strip() if ev_c else ""
        if not event or event.lower() in ("event", "nan", ""):
            continue
        events.append({
            "event": event,
            "waits": clean_number(row.get(waits_c)) if waits_c else None,
            "time_s": clean_number(row.get(time_c)) if time_c else None,
            "avg_wait_ms": clean_number(row.get(avg_c)) if avg_c else None,
            "pct_db_time": clean_number(row.get(pct_c)) if pct_c else None,
            "wait_class": str(row.get(class_c)).strip() if class_c else None,
        })
    return events


def parse_wait_classes(soup: BeautifulSoup) -> list:
    """Foreground wait-class rollup (User I/O, Concurrency, Commit, ...)."""
    classes: list = []
    for frag in (
        "wait classes by total wait time",
        "foreground wait class",
        "this table displays wait class",
        "wait class",
    ):
        df = _get_table_df(soup, frag, frag)
        if not df.empty:
            break
    else:
        return classes

    cls_c = _col(df, "wait", "class") or df.columns[0]
    waits_c = _col(df, "waits")
    time_c = _col(df, "total", "wait") or _col(df, "time")
    avg_c = _col(df, "avg", "wait") or _col(df, "(ms)")
    pct_c = _col(df, "%", "db") or _col(df, "db", "time")
    for _, row in df.iterrows():
        name = str(row.get(cls_c, "")).strip()
        if not name or name.lower() in ("wait class", "nan", ""):
            continue
        classes.append({
            "wait_class": name,
            "waits": clean_number(row.get(waits_c)) if waits_c else None,
            "total_wait_s": clean_number(row.get(time_c)) if time_c else None,
            "avg_wait_ms": clean_number(row.get(avg_c)) if avg_c else None,
            "pct_db_time": clean_number(row.get(pct_c)) if pct_c else None,
        })
    return classes


def parse_time_model(soup: BeautifulSoup) -> dict:
    """Time Model statistics — DB time breakdown by statistic."""
    result: dict = {}
    df = _get_table_df(soup, "time model statistics", "Time Model")
    if df.empty:
        return result
    stat_c = _col(df, "stat") or df.columns[0]
    time_c = _col(df, "time", "s") or _col(df, "time")
    pct_c = _col(df, "%", "db") or _col(df, "%")
    for _, row in df.iterrows():
        metric = str(row.get(stat_c, "")).strip()
        if not metric or metric.lower() in ("statistic name", "stat name", "nan", ""):
            continue
        result[metric] = {
            "time_s": clean_number(row.get(time_c)) if time_c else None,
            "pct_db_time": clean_number(row.get(pct_c)) if pct_c else None,
        }
    return result


def parse_host_cpu(soup: BeautifulSoup) -> dict:
    """Host CPU: CPUs, cores, sockets, load avg, %user/%sys/%wio/%idle."""
    result: dict = {}
    df = _get_table_df(soup, "host cpu", "Host CPU")
    if df.empty:
        # Real 19c AWR labels this "system load statistics".
        df = _get_table_df(soup, "system load statistics", "%Idle")
    if df.empty:
        return result
    row = df.iloc[0]
    grab = {
        "CPUs": ("cpus",),
        "Cores": ("cores",),
        "Sockets": ("sockets",),
        "Load Begin": ("begin",),
        "Load End": ("end",),
        "%User": ("%user",),
        "%System": ("%system",),
        "%WIO": ("%wio",),
        "%Idle": ("%idle",),
    }
    for label, needles in grab.items():
        c = _col(df, *needles)
        if c is not None:
            result[label] = clean_number(row.get(c))
    # Load Average sometimes split into two unlabeled columns; best-effort only.
    return result


def parse_instance_cpu(soup: BeautifulSoup) -> dict:
    """Instance CPU: %Total CPU, %Busy CPU, %DB time waiting for CPU."""
    result: dict = {}
    df = _get_table_df(soup, "instance cpu", "Instance CPU")
    if df.empty:
        # Real 19c AWR labels this "CPU usage and wait statistics".
        df = _get_table_df(soup, "cpu usage and wait", "%Total CPU")
    if df.empty:
        return result
    row = df.iloc[0]
    grab = {
        "%Total CPU": ("%total", "cpu"),
        "%Busy CPU": ("%busy", "cpu"),
        "%DB time waiting for CPU": ("waiting", "cpu"),
    }
    for label, needles in grab.items():
        c = _col(df, *needles)
        if c is not None:
            result[label] = clean_number(row.get(c))
    if not result:  # row-oriented fallback
        for _, r in df.iterrows():
            k = str(r.iloc[0]).strip()
            v = clean_number(r.iloc[1] if len(r) > 1 else None)
            if k and v is not None:
                result[k] = v
    return result


def parse_io_profile(soup: BeautifulSoup) -> dict:
    """IO Profile: read+write/s, read/s, write/s, redo MB/s, etc."""
    result: dict = {}
    df = _get_table_df(soup, "io profile", "IO Profile")
    if df.empty:
        return result
    per_sec_c = _col(df, "per second") or (df.columns[1] if len(df.columns) > 1 else None)
    metric_c = _col(df, "metric") or df.columns[0]
    for _, row in df.iterrows():
        metric = str(row.get(metric_c, "")).strip().rstrip(":")
        if not metric or metric.lower() in ("metric", "nan", ""):
            continue
        val = clean_number(row.get(per_sec_c)) if per_sec_c else None
        if val is not None:
            result[metric] = val
    return result


def parse_memory(soup: BeautifulSoup) -> dict:
    """Memory statistics: SGA/PGA begin/end (MB), %host mem, cache hit ratios."""
    result: dict = {}
    df = _get_table_df(soup, "memory statistics", "Memory Statistics")
    if df.empty:
        df = _get_table_df(soup, "memory", "Memory Statistics")
    if df.empty:
        return result
    begin_c = _col(df, "begin")
    end_c = _col(df, "end")
    metric_c = _col(df, "metric") or df.columns[0]
    for _, row in df.iterrows():
        metric = str(row.get(metric_c, "")).strip().rstrip(":")
        if not metric or metric.lower() in ("metric", "nan", ""):
            continue
        entry = {}
        if begin_c is not None:
            entry["begin"] = clean_number(row.get(begin_c))
        if end_c is not None:
            entry["end"] = clean_number(row.get(end_c))
        if not entry and len(row) > 1:  # simple key/value rows
            entry = clean_number(row.iloc[1])
        if entry not in ({}, None):
            result[metric] = entry
    return result


# ── Top SQL (compact) ───────────────────────────────────────────────────────

# Top-SQL categories kept (most diagnostically useful). Buffer Gets was added so
# the "CPU burned on logical reads" finding has a SQL list to point at, and so
# logical-read regressions can be tracked per sql_id over time.
_SQL_CATEGORIES = {
    "elapsed_time": "sql by elapsed time",
    "cpu_time": "sql by cpu time",
    "physical_reads": "sql by physical reads",
    "buffer_gets": "sql by buffer gets",
}


def _parse_sql_table(soup: BeautifulSoup, summary_frag: str) -> list:
    """Extract a Top-SQL table -> compact rows, capped at TOP_SQL_LIMIT."""
    df = _get_table_df(soup, summary_frag, summary_frag)
    if df.empty:
        return []

    sqlid_c = _col(df, "sql id") or _col(df, "sql_id")
    # Absolute-time/count columns only: exclude "%..." and "per exec" variants,
    # so e.g. a "%CPU" column is never mistaken for absolute "CPU Time (s)".
    def _metric_col(*needles):
        for c in df.columns:
            cl = str(c).lower()
            if all(n in cl for n in needles) and "%" not in cl and "per" not in cl:
                return c
        return None
    elapsed_c = _metric_col("elapsed", "time")
    cpu_c = _metric_col("cpu", "time")
    preads_c = _metric_col("physical", "read")
    bgets_c = _metric_col("buffer", "gets") or _metric_col("gets")
    exec_c = _col(df, "executions") or _col(df, "execs")
    text_c = _col(df, "sql", "text") or _col(df, "sql text")
    module_c = _col(df, "module")
    # Plan Hash Value column (present in 12c/19c Top-SQL tables). Needed to detect
    # execution-plan changes for a sql_id across snapshots.
    plan_c = _col(df, "plan", "hash") or _col(df, "phv")

    rows: list = []
    for _, row in df.iterrows():
        sql_id = str(row.get(sqlid_c, "")).strip() if sqlid_c else ""
        if not sql_id or sql_id.lower() in ("sql id", "nan", ""):
            continue
        text = str(row.get(text_c, "")).strip() if text_c else ""
        phv = clean_number(row.get(plan_c)) if plan_c else None
        rows.append({
            "sql_id": sql_id,
            "elapsed_time": clean_number(row.get(elapsed_c)) if elapsed_c else None,
            "cpu_time": clean_number(row.get(cpu_c)) if cpu_c else None,
            "physical_reads": clean_number(row.get(preads_c)) if preads_c else None,
            "buffer_gets": clean_number(row.get(bgets_c)) if bgets_c else None,
            "executions": clean_number(row.get(exec_c)) if exec_c else None,
            # store plan hash as a string (it is an identifier, not a quantity)
            "plan_hash_value": (str(int(phv)) if phv is not None else None),
            "module": (str(row.get(module_c)).strip() if module_c else None),
            "sql_text": (text[:SQL_TEXT_TRUNCATE] + ("…" if len(text) > SQL_TEXT_TRUNCATE else "")) if text else "",
        })
        if len(rows) >= TOP_SQL_LIMIT:
            break
    return rows


def parse_top_sql(soup: BeautifulSoup) -> dict:
    """Top SQL by Elapsed Time, CPU Time, Physical Reads, Buffer Gets."""
    return {key: _parse_sql_table(soup, frag) for key, frag in _SQL_CATEGORIES.items()}


# ── Top Segments ─────────────────────────────────────────────────────────────

# AWR "Segments by ..." sections, mapped to a compact category name. Lets RCA
# attribute I/O / contention to specific objects, and track hot segments over time.
_SEGMENT_CATEGORIES = {
    "logical_reads": "segments by logical reads",
    "physical_reads": "segments by physical reads",
    "buffer_busy_waits": "segments by buffer busy waits",
    "row_lock_waits": "segments by row lock waits",
}

SEGMENT_LIMIT = 5


def _parse_segment_table(soup: BeautifulSoup, summary_frag: str) -> list:
    df = _get_table_df(soup, summary_frag, summary_frag)
    if df.empty:
        return []
    owner_c = _col(df, "owner")
    name_c = _col(df, "segment", "name") or _col(df, "object", "name")
    type_c = _col(df, "segment", "type") or _col(df, "object", "type")
    # The value column is the last numeric-ish one; prefer an explicit count/% column.
    val_c = _col(df, "%", "total") or _col(df, "logical") or _col(df, "physical") \
        or _col(df, "waits") or (df.columns[-1] if len(df.columns) else None)

    rows: list = []
    for _, row in df.iterrows():
        name = str(row.get(name_c, "")).strip() if name_c else ""
        if not name or name.lower() in ("segment name", "object name", "nan", ""):
            continue
        rows.append({
            "owner": (str(row.get(owner_c)).strip() if owner_c else None),
            "segment_name": name,
            "segment_type": (str(row.get(type_c)).strip() if type_c else None),
            "value": clean_number(row.get(val_c)) if val_c else None,
        })
        if len(rows) >= SEGMENT_LIMIT:
            break
    return rows


def parse_segments(soup: BeautifulSoup) -> dict:
    """Top segments per category (logical/physical reads, buffer-busy, row-lock)."""
    return {key: _parse_segment_table(soup, frag)
            for key, frag in _SEGMENT_CATEGORIES.items()}


# ════════════════════════════════════════════════════════════════════════════
# Master parse
# ════════════════════════════════════════════════════════════════════════════

def parse_awr_report(html_path: str) -> dict:
    """Parse an AWR HTML report into the structured (raw) metric sections.

    Returns the report WITHOUT derived_metrics / findings — those are added by
    the pipeline after this stage. The keys here match the final schema so the
    pipeline can simply attach derived_metrics + findings.
    """
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"AWR file not found: {html_path}")

    log.info("Loading AWR report: %s", path.name)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        soup = BeautifulSoup(fh, "lxml")
    log.info("HTML parsed — extracting diagnostic sections…")

    report = {
        "source_file": str(path),
        "report_name": path.stem,
        "header": parse_header(soup),
        "snapshot": parse_snapshot(soup),
        "load_profile": parse_load_profile(soup),
        "instance_efficiency": parse_instance_efficiency(soup),
        "wait_events": parse_wait_events(soup),
        "wait_classes": parse_wait_classes(soup),
        "time_model": parse_time_model(soup),
        "host_cpu": parse_host_cpu(soup),
        "instance_cpu": parse_instance_cpu(soup),
        "io_profile": parse_io_profile(soup),
        "memory": parse_memory(soup),
        "top_sql": parse_top_sql(soup),
        "top_segments": parse_segments(soup),
    }

    populated = sum(
        1 for k, v in report.items()
        if k not in ("source_file", "report_name") and bool(v)
    )
    log.info("Extraction complete — %d/%d sections populated.",
             populated, len(report) - 2)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse an Oracle AWR HTML report.")
    ap.add_argument("awr_html", help="Path to AWR HTML file")
    ap.add_argument("--json", default="awr_metrics.json", help="Output JSON path")
    args = ap.parse_args()
    rpt = parse_awr_report(args.awr_html)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(rpt, fh, indent=2, default=str)
    log.info("Wrote %s", args.json)
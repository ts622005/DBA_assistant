#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Orchestrates the deterministic stages of the AI-powered DBA Assistant:

    AWR HTML
      -> awr_parser            (structured metrics)
      -> derived_metrics       (ratios + signals)
      -> rule_engine           (correlated findings)
      -> assemble final JSON   (compact, retrieval-ready)

The output JSON follows the agreed schema exactly. It is intentionally compact
so it embeds and retrieves well downstream (embeddings -> ChromaDB -> agents).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import awr_parser
import addm_parser
import derived_metrics
from rule_engine import RuleEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# Final schema key order.
_SCHEMA_KEYS = [
    "header", "snapshot", "load_profile", "instance_efficiency",
    "wait_events", "wait_classes", "time_model", "host_cpu",
    "instance_cpu", "io_profile", "memory", "top_sql",
    "top_segments", "derived_metrics", "findings",
]


_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _make_report_id(hdr: dict, snap: dict) -> str:
    """Deterministic, human-readable id that is stable across re-ingests.

    Re-ingesting the same AWR report yields the same id, so the structured store
    and vector store can upsert idempotently instead of overwriting or duplicating.
    Falls back to a content hash when snap ids are missing.
    """
    db = (hdr.get("DB Name") or hdr.get("DB Id") or "DB")
    inst = hdr.get("Inst Num")
    b = snap.get("Begin Snap Id")
    e = snap.get("End Snap Id")
    if b is not None and e is not None:
        inst_part = int(inst) if inst is not None else "x"
        return f"{db}_{inst_part}_{int(b)}_{int(e)}"
    seed = "|".join(str(x) for x in (
        hdr.get("DB Id"), inst, snap.get("Begin Time"), snap.get("End Time")))
    return f"{db}_{hashlib.md5(seed.encode()).hexdigest()[:10]}"


def analyze(html_path: str, rules_path: str | None = None) -> dict:
    """Run the full deterministic pipeline and return the final report dict."""
    raw = awr_parser.parse_awr_report(html_path)

    derived, signals = derived_metrics.compute(raw)

    engine = RuleEngine(rules_path) if rules_path else RuleEngine()
    rule_findings = engine.evaluate(signals)
    for f in rule_findings:
        f.setdefault("source", "rule_engine")

    # Oracle's own findings, parsed from the ADDM text section (if present).
    addm_findings = addm_parser.parse_addm(html_path)

    # One unified, severity-sorted findings list. Source tag lets the agents
    # tell apart "the rules derived this" from "Oracle's ADDM said this", and
    # spot where the two independently agree.
    findings = rule_findings + addm_findings
    findings.sort(key=lambda f: (_SEV_ORDER.get(f.get("severity"), 9),
                                 -(f.get("confidence") or 0)))

    report: dict = {k: raw.get(k, {} if not k.endswith("s") else []) for k in _SCHEMA_KEYS}
    # fix list-typed sections explicitly
    report["wait_events"] = raw.get("wait_events", [])
    report["wait_classes"] = raw.get("wait_classes", [])
    report["derived_metrics"] = derived
    report["findings"] = findings

    # lightweight meta for retrieval/filtering (extra, outside the core schema)
    hdr = raw.get("header", {})
    snap = raw.get("snapshot", {})
    report["meta"] = {
        "report_id": _make_report_id(hdr, snap),   # stable id -> idempotent ingest
        "source_file": raw.get("source_file"),
        "report_name": raw.get("report_name"),
        "db_name": hdr.get("DB Name"),
        "db_id": hdr.get("DB Id"),
        "instance_number": hdr.get("Inst Num"),
        "release": hdr.get("Release"),
        "is_rac": hdr.get("RAC"),
        "host": hdr.get("Host"),
        "begin_snap": snap.get("Begin Snap Id"),
        "end_snap": snap.get("End Snap Id"),
        "begin_time": snap.get("Begin Time"),
        "end_time": snap.get("End Time"),
        # ISO-8601 timestamps for first-class time queries (fall back to raw string)
        "begin_time_iso": snap.get("Begin Time ISO") or snap.get("Begin Time"),
        "end_time_iso": snap.get("End Time ISO") or snap.get("End Time"),
        "elapsed_mins": snap.get("Elapsed (mins)"),
        "db_time_mins": snap.get("DB Time (mins)"),
        "signals": signals,  # kept for transparency / agent grounding
    }

    crit = sum(1 for f in findings if f["severity"] in ("CRITICAL", "HIGH"))
    log.info("Pipeline done — %d findings (%d high/critical).", len(findings), crit)
    return report


def analyze_to_file(html_path: str, out_json: str, rules_path: str | None = None) -> dict:
    report = analyze(html_path, rules_path)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("Wrote %s", out_json)
    return report


def analyze_and_ingest(html_path: str, store, rules_path: str | None = None,
                       metrics_store=None) -> dict:
    """Run full pipeline and ingest into the vector store, and (if provided) the
    structured metrics store. Structured facts -> metrics_store; prose -> vector."""
    report = analyze(html_path, rules_path)
    n = store.ingest(report)
    log.info("Ingested %d documents into vector store.", n)
    if metrics_store is not None:
        sid = metrics_store.upsert_snapshot(report)
        log.info("Stored structured snapshot %s into metrics store.", sid)
    return report


def analyze_and_store(html_path: str, metrics_store, rules_path: str | None = None) -> dict:
    """Run full pipeline and write ONLY the structured metrics store (no vector).

    Useful for backfills / environments without the embedding stack installed.
    """
    report = analyze(html_path, rules_path)
    sid = metrics_store.upsert_snapshot(report)
    log.info("Stored structured snapshot %s into metrics store.", sid)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AWR -> structured metrics + findings JSON")
    ap.add_argument("awr_html", help="Path to AWR HTML file")
    ap.add_argument("--out", default="awr_analysis.json", help="Output JSON path")
    ap.add_argument("--rules", default=None, help="Path to rules.yaml (optional)")
    ap.add_argument("--ingest", action="store_true", help="Also embed & store in ChromaDB")
    ap.add_argument("--chroma-dir", default="./chroma_awr", help="ChromaDB persist directory")
    ap.add_argument("--metrics-db", default=None,
                    help="Path to the structured SQLite store. With --ingest, writes both; "
                         "alone, writes ONLY the structured store (backfill, no embeddings).")
    args = ap.parse_args()

    if args.ingest:
        import config
        from vector_store import AWRVectorStore
        from metrics_store import MetricsStore
        store = AWRVectorStore(
            config.get_embedder(), persist_dir=args.chroma_dir,
            collection=config.COLLECTION,
            embedder_id=config.EMBEDDER_ID, embedder_dim=config.EMBEDDER_DIM,
        )
        mstore = MetricsStore(args.metrics_db) if args.metrics_db else None
        rpt = analyze_and_ingest(args.awr_html, store, args.rules, metrics_store=mstore)
    elif args.metrics_db:
        from metrics_store import MetricsStore
        rpt = analyze_and_store(args.awr_html, MetricsStore(args.metrics_db), args.rules)
    else:
        rpt = analyze_to_file(args.awr_html, args.out, args.rules)

    print("\nFindings:")
    for f in rpt["findings"]:
        print(f"  [{f['severity']:<8}] {f['finding']}  (conf {f['confidence']})")
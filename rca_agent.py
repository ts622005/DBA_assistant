#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rca_agent.py
============
RCA Agent (Phase 3) — root-cause analysis that can ONLY speak about cited facts.

Reuses the deterministic findings the rule engine and ADDM already produced
(retrieved structurally by the Metrics Agent) as the backbone of the diagnosis,
rather than re-deriving causes from raw text. Its guarantees:

  * Every root cause MUST cite at least one evidence item; an un-citeable claim
    is dropped, never shown.
  * Under a "partial" answer scope (e.g. index questions, where AWR lacks the
    execution plan), confirmed causes are demoted to inferred and any access-path
    conclusion is demoted to speculative.
  * It never invents an execution plan, index, or a number not in the bundle.

It is deterministic and LLM-free, so it is fully testable offline. An optional
narration step (in the assembler/app) may rephrase the structured output, but it
is constrained to these cited causes.

Three-tier classification per cause: confirmed | inferred | speculative.
"""

from __future__ import annotations

import logging

log = logging.getLogger("rca_agent")

# Intents where any access-path / index conclusion is inherently speculative
# without an execution plan.
_PLAN_DEPENDENT_INTENTS = {"index_benefit", "index_usage", "full_table_scan"}

# Map the question intent to the bottleneck dimension and the Top-SQL category to
# rank by, so the RCA can name the SINGLE biggest offender from structured data.
_INTENT_BOTTLENECK = {
    "cpu_bottleneck": ("CPU", "cpu_time"),
    "io_bottleneck": ("I/O", "physical_reads"),
    "index_benefit": ("excessive logical reads", "buffer_gets"),
    "index_usage": ("excessive logical reads", "buffer_gets"),
    "full_table_scan": ("I/O", "physical_reads"),
    "parse_issue": ("parsing", "cpu_time"),
    "commit_latency": ("commit/redo", "elapsed_time"),
    "concurrency": ("concurrency", "elapsed_time"),
    "general": (None, "cpu_time"),
}
_SQL_METRIC_FIELD = {"cpu_time": ("cpu_time_s", "CPU time", "s"),
                     "elapsed_time": ("elapsed_time_s", "elapsed time", "s"),
                     "physical_reads": ("physical_reads", "physical reads", ""),
                     "buffer_gets": ("buffer_gets", "buffer gets", "")}


def _fmtnum(v):
    if v is None:
        return "n/a"
    try:
        v = float(v)
        return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _headline_cause(intent: str, bundle: dict, scope: str) -> dict | None:
    """A crisp, data-ranked lead: the single biggest offender SQL_ID for the
    bottleneck implied by the question, taken straight from sql_stat (not ADDM)."""
    label, cat = _INTENT_BOTTLENECK.get(intent, (None, "cpu_time"))
    rows = (bundle.get("top_sql_by", {}) or {}).get(cat) or bundle.get("top_sql") or []
    if not rows:
        return None
    top = rows[0]
    field, fld_label, unit = _SQL_METRIC_FIELD.get(cat, ("cpu_time_s", "CPU time", "s"))
    val = top.get(field)
    sqlid = top.get("sql_id")
    if not sqlid:
        return None
    bottleneck = label or "database time"
    evid = [f"Top SQL by {fld_label} (rank 1, AWR): {sqlid} = {_fmtnum(val)}{unit} "
            f"over {_fmtnum(top.get('executions'))} executions"]
    # add the headline system metric for context if present
    for fct in bundle.get("facts", []):
        if fct.startswith("DB CPU is") or fct.startswith("User I/O is") \
           or fct.startswith("Logical-to-physical"):
            evid.append(fct)
            break
    return {
        "cause": f"{bottleneck.capitalize()} is the dominant bottleneck; the top "
                 f"{fld_label} consumer is SQL_ID {sqlid}",
        "category": "SQL", "severity": "HIGH",
        "evidence": evid, "contributing_metrics": evid,
        "confidence": 0.9 if scope == "full" else 0.6,
        "classification": "confirmed" if scope == "full" else "inferred",
        "source": "metrics_agent", "rule_id": None, "sql_id": sqlid,
        "base_recommendation": (
            f"Run SQL Tuning Advisor on SQL_ID {sqlid}, and capture its DBMS_XPLAN "
            f"to confirm the access path before any index, join-order or plan change."),
    }


def _classify(source: str | None, confidence: float | None, scope: str) -> str:
    conf = confidence or 0.0
    if scope == "partial":
        return "inferred" if conf >= 0.6 else "speculative"
    if source in ("rule_engine", "oracle_addm") and conf >= 0.7:
        return "confirmed"
    return "inferred"


class RCAAgent:
    def analyze(self, question: str, gate: dict, bundle: dict) -> dict:
        scope = gate.get("answer_scope", "full")
        intent = bundle.get("intent", "general")
        causes: list[dict] = []

        # 0) Headline: name the single biggest offender SQL_ID directly from the
        #    ranked sql_stat data (answers "which SQL_ID" crisply, not via ADDM prose).
        #    Skipped for plan-dependent intents, which surface candidates instead (step 2).
        headline = None
        if intent not in _PLAN_DEPENDENT_INTENTS:
            headline = _headline_cause(intent, bundle, scope)
            if headline:
                causes.append(headline)
        headline_sqlid = headline["sql_id"] if headline else None

        # 1) Root causes from fired findings (citation enforced).
        for f in bundle.get("findings", []):
            evidence = [e for e in (f.get("evidence") or []) if e]
            if not evidence:
                continue  # no citation -> drop
            conf = f.get("confidence")
            cls = _classify(f.get("source"), conf, scope)
            causes.append({
                "cause": f.get("finding"),
                "category": f.get("category"),
                "severity": f.get("severity"),
                "evidence": evidence,
                "contributing_metrics": evidence,
                "confidence": round((conf or 0.0) * (0.85 if scope == "partial" else 1.0), 2),
                "classification": cls,
                "source": f.get("source"),
                "rule_id": f.get("rule_id"),
                "base_recommendation": f.get("recommendation"),
            })

        # 2) For plan-dependent intents, surface high-resource SQL as CANDIDATES
        #    (explicitly speculative) — never asserted as confirmed causes.
        if intent in _PLAN_DEPENDENT_INTENTS and bundle.get("top_sql"):
            cand_ev = [fct for fct in bundle.get("facts", [])
                       if fct.startswith("Top SQL") or fct.startswith(
                           bundle.get("sql_focus", {}).get("sql_id", "\0") if bundle.get("sql_focus") else "\0")]
            if not cand_ev:
                cand_ev = [f"Top SQL by {bundle.get('sql_category','').replace('_',' ')}: "
                           f"{s['sql_id']}" for s in bundle["top_sql"][:3]]
            causes.append({
                "cause": "High-resource SQL are indexing/plan CANDIDATES (not confirmed)",
                "category": "SQL",
                "severity": "INFO",
                "evidence": cand_ev,
                "contributing_metrics": cand_ev,
                "confidence": 0.4,
                "classification": "speculative",
                "source": "metrics_agent",
                "rule_id": None,
                "base_recommendation": None,
                "requires": "execution_plan",
            })

        # 3) Fallback: nothing fired and nothing to speculate -> report facts only.
        if not causes:
            facts = bundle.get("facts", [])[:5]
            causes.append({
                "cause": "No rule-based root cause fired; reporting salient measured facts",
                "category": "INFO", "severity": "INFO",
                "evidence": facts or ["No diagnostic signals exceeded thresholds."],
                "contributing_metrics": facts,
                "confidence": 0.45 if facts else 0.2,
                "classification": "inferred" if facts else "speculative",
                "source": "metrics_agent", "rule_id": None, "base_recommendation": None,
            })

        reasoning = self._reasoning(causes, scope, gate)
        return {"scope": scope, "intent": intent, "root_causes": causes,
                "reasoning": reasoning}

    @staticmethod
    def _reasoning(causes: list[dict], scope: str, gate: dict) -> str:
        if not causes:
            return "Insufficient evidence to determine a root cause."
        top = causes[0]
        bits = [f"The leading {top['classification']} cause is '{top['cause']}', "
                f"supported by: {top['evidence'][0]}."]
        if len(causes) > 1:
            bits.append(f"{len(causes) - 1} additional contributing factor(s) identified.")
        if scope == "partial" and gate.get("missing_evidence"):
            bits.append("This is a partial analysis — "
                        f"{', '.join(gate['missing_evidence'])} would be needed to confirm "
                        "any access-path conclusion.")
        return " ".join(bits)

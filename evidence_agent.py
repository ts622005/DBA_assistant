#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evidence_agent.py
=================
Evidence Verification Layer / Evidence Agent (Phase 2).

This is the anti-hallucination gate. BEFORE any answer is generated, it decides
whether the question is answerable from the evidence actually present, and names
what is missing. It returns the §5.3 contract:

    {
      "can_answer": false,
      "answer_scope": "none",            # "none" | "partial" | "full"
      "confidence": 0.2,
      "evidence_present": [...],
      "missing_evidence": [...],
      "reason": "..."
    }

The decision is made by the intent map + what the structured store contains —
NOT by the generating model. Downstream RCA / Recommendation (Phase 3) must only
run when can_answer is True.

`available_evidence(store)` introspects the MetricsStore to build the set of
evidence types currently on hand. Sources we don't ingest yet (execution plans,
SQL Monitor, ASH) are simply never in the set, so any intent that needs them is
refused automatically — and will start being answerable the moment those
parsers are added (Phase 5), with no change here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from question_classifier import QuestionClassifier

log = logging.getLogger("evidence_agent")

_DEFAULT_REQS = Path(__file__).with_name("evidence_requirements.yaml")

# Evidence types that no current parser produces. Listed so reviewers can see the
# explicit gap, and so the gate refuses anything depending on them.
NOT_YET_AVAILABLE = frozenset({"execution_plan", "sql_monitor", "ash"})


def available_evidence(store) -> set[str]:
    """Introspect a MetricsStore and return the set of present evidence types.

    Global view across all ingested snapshots — sufficient for Phase 2. (A later
    refinement can scope this to the snapshot(s) a question targets.)
    """
    have: set[str] = set()
    if store is None:
        return have
    c = store.conn

    def _has(sql, args=()):
        return c.execute(sql, args).fetchone()[0] > 0

    if _has("SELECT COUNT(*) FROM sql_stat"):
        have.add("sql_stat")
    if _has("SELECT COUNT(*) FROM wait_event WHERE kind='event'"):
        have.add("wait_events")
    if _has("SELECT COUNT(*) FROM wait_event WHERE kind='class'"):
        have.add("wait_classes")
    if _has("SELECT COUNT(*) FROM metric WHERE name='db_cpu_pct_db_time'"):
        have.add("time_model")
    if _has("SELECT COUNT(*) FROM metric WHERE name IN ('num_cpus','host_cpu_idle_pct')"):
        have.add("instance_cpu")
    if _has("SELECT COUNT(*) FROM metric WHERE name LIKE '%hit_pct' OR name='mem_pct_host'"):
        have.add("memory")
    if _has("SELECT COUNT(*) FROM segment_stat"):
        have.add("segments")
    if _has("SELECT COUNT(*) FROM addm_finding"):
        have.add("addm")
    if _has("SELECT COUNT(*) FROM snapshot") and \
       c.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0] >= 2:
        have.add("two_or_more_snapshots")
    if _has("SELECT COUNT(*) FROM sql_stat WHERE plan_hash_value IS NOT NULL") and \
       "two_or_more_snapshots" in have:
        have.add("plan_hash_history")
    return have


class EvidenceAgent:
    def __init__(self, reqs_path: str | Path = _DEFAULT_REQS):
        with open(reqs_path, "r", encoding="utf-8") as fh:
            self.intents = (yaml.safe_load(fh) or {}).get("intents", {})
        self.classifier = QuestionClassifier(reqs_path)

    def assess(self, question: str, available: set[str]) -> dict:
        """Decide answerability for `question` given the `available` evidence set."""
        cls = self.classifier.classify(question)
        intent = cls["intent"]
        spec = self.intents.get(intent, {})

        needs = set(spec.get("needs") or [])
        recommends = set(spec.get("recommends") or [])
        answerable = spec.get("answerable_from_awr", True)

        needs_missing = needs - available
        recommends_missing = recommends - available

        # Case 1: AWR fundamentally cannot answer this.
        if answerable is False:
            return self._result(
                intent, can_answer=False, scope="none", confidence=0.2,
                present=sorted(available & (needs | recommends)),
                missing=sorted(needs | recommends),
                reason=_clean(spec.get("refusal") or
                              "This question cannot be answered from AWR data alone."))

        # Case 2: a HARD requirement is missing -> refuse, name what's missing.
        if needs_missing:
            return self._result(
                intent, can_answer=False, scope="none", confidence=0.25,
                present=sorted(available & needs),
                missing=sorted(needs_missing),
                reason=_clean(spec.get("refusal") or spec.get("note") or
                              f"Missing required evidence: {', '.join(sorted(needs_missing))}."))

        # Case 3: answerable. Partial if intent says so or soft evidence is missing.
        is_partial = (answerable == "partial") or bool(recommends_missing)
        scope = "partial" if is_partial else "full"
        base = 0.55 if is_partial else 0.75
        return self._result(
            intent, can_answer=True, scope=scope, confidence=base,
            present=sorted(available & (needs | recommends)) or sorted(needs),
            missing=sorted(recommends_missing),
            reason=_clean(spec.get("note") or "") or None)

    @staticmethod
    def _result(intent, *, can_answer, scope, confidence, present, missing, reason):
        return {
            "intent": intent,
            "can_answer": can_answer,
            "answer_scope": scope,
            "confidence": round(confidence, 2),
            "evidence_present": present,
            "missing_evidence": missing,
            "reason": reason,
        }


def _clean(s) -> str:
    return " ".join((s or "").split())


if __name__ == "__main__":
    import sys, json
    # Demo against a metrics store if given, else assume rich AWR evidence present.
    if len(sys.argv) > 1:
        from metrics_store import MetricsStore
        avail = available_evidence(MetricsStore(sys.argv[1]))
    else:
        avail = {"sql_stat", "wait_events", "wait_classes", "time_model",
                 "instance_cpu", "memory", "segments", "addm"}
    print("available:", sorted(avail), "\n")
    ea = EvidenceAgent()
    for q in [
        "Which SQL statements would benefit most from indexing?",
        "Are any SQL statements performing full table scans?",
        "Why might this UPDATE not be using an index even if one exists?",
        "What is causing high CPU usage?",
        "Show CPU trend over the last 30 days.",
    ]:
        print(q)
        print(json.dumps(ea.assess(q, avail), indent=2), "\n")

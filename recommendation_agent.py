#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommendation_agent.py
=======================
Recommendation Agent (Phase 3) — actions, prioritized, never inventing plans.

Turns the RCA agent's confirmed/inferred causes into concrete, prioritized
actions. Hard constraints:

  * It NEVER invents execution-plan information. If an action would require a plan
    (add/confirm an index, change an access path), and the plan is not available,
    the action is rephrased as "capture DBMS_XPLAN for SQL_ID X to confirm …" and
    tagged requires="execution_plan" — not stated as a definite fix.
  * It reuses each finding's own recommendation (from rules.yaml / ADDM) as the
    prior, so guidance stays consistent with the deterministic knowledge base.
  * Priority follows severity; speculative causes never outrank confirmed ones.

Output: a list of
    {action, priority, basis, impact, requires?}
"""

from __future__ import annotations

import logging

log = logging.getLogger("recommendation_agent")

_SEV_PRIORITY = {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

# Words that signal an action depends on the execution plan we don't have.
_PLAN_WORDS = ("index", "plan", "access path", "full scan", "full table scan",
               "join order", "cardinality")


def _needs_plan(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _PLAN_WORDS)


class RecommendationAgent:
    def recommend(self, rca: dict, gate: dict, bundle: dict) -> list[dict]:
        recs: list[dict] = []
        seen: set[str] = set()
        plan_missing = "execution_plan" in (gate.get("missing_evidence") or [])
        focus_sql = (bundle.get("sql_focus") or {}).get("sql_id")
        # A representative candidate sql_id for plan-capture suggestions.
        cand_sql = focus_sql or (bundle["top_sql"][0]["sql_id"] if bundle.get("top_sql") else "the top SQL")

        for cause in rca.get("root_causes", []):
            sev = cause.get("severity", "MEDIUM")
            base = cause.get("base_recommendation")
            classification = cause.get("classification")

            # Build the action text.
            if cause.get("requires") == "execution_plan" or (
                    classification == "speculative" and _needs_plan(cause.get("cause", ""))):
                action = (f"Capture DBMS_XPLAN / SQL Monitor for {cand_sql} to confirm the "
                          f"access path before any index or plan change.")
                requires = "execution_plan"
            elif base and plan_missing and _needs_plan(base):
                # The rule suggests indexing, but we can't confirm without a plan.
                action = (f"{_trim(base)} — but first capture DBMS_XPLAN for {cand_sql} "
                          f"to confirm an index/plan change is warranted.")
                requires = "execution_plan"
            elif base:
                action = _trim(base)
                requires = None
            else:
                action = f"Investigate: {cause.get('cause')}"
                requires = None

            key = action.lower()[:80]
            if key in seen:
                continue
            seen.add(key)

            recs.append({
                "action": action,
                "priority": _SEV_PRIORITY.get(sev, 3) + (1 if classification == "speculative" else 0),
                "basis": cause.get("cause"),
                "classification": classification,
                "impact": _impact(sev),
                **({"requires": requires} if requires else {}),
            })

        recs.sort(key=lambda r: r["priority"])
        for i, r in enumerate(recs, 1):
            r["priority"] = i  # normalize to 1..n after sort
        return recs


def _impact(sev: str) -> str:
    return {"CRITICAL": "high", "HIGH": "high", "MEDIUM": "medium",
            "LOW": "low", "INFO": "informational"}.get(sev, "medium")


def _trim(s: str) -> str:
    import re
    s = " ".join((s or "").split())
    s = re.sub(r"^\s*\d+\.\s*", "", s)  # drop a leading "1. " so it isn't double-numbered
    return (s[:300] + "…") if len(s) > 300 else s

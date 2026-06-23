#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confidence.py
=============
Answer-level confidence scoring (Phase 2, §6 of the refactor plan).

Today confidence lives only inside individual rule findings. This promotes it to
the whole ANSWER, and makes it a transparent function of evidence — never a number
the LLM invents.

Scoring model (§6.2):

    score = w_evidence * coverage
          + w_rules    * mean(fired-rule confidence used)
          + w_corrob   * corroboration_bonus      # ADDM and rules agree -> +
          - w_missing  * missing_fraction

Bounded to [0, 1]. We report the BAND, not false precision.

Three-tier classification of the answer:
    confirmed   — backed by measured facts / fired rules (and ideally corroborated)
    inferred    — reasoned from confirmed facts, some evidence missing / partial
    speculative — partial scope or weak evidence; a lead, not a conclusion
"""

from __future__ import annotations

# weights (sum of positive weights = 1.0; missing is a penalty)
W_EVIDENCE = 0.40
W_RULES = 0.40
W_CORROB = 0.20
W_MISSING = 0.30


def _band(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Medium"
    if score >= 0.30:
        return "Low"
    return "Insufficient"


def _classification(scope: str, score: float, corroborated: bool) -> str:
    if scope == "none":
        return "speculative"
    if scope == "full" and score >= 0.75 and corroborated:
        return "confirmed"
    if scope == "full" and score >= 0.70:
        return "confirmed"
    if scope == "partial" or score < 0.55:
        return "speculative" if score < 0.45 else "inferred"
    return "inferred"


def compute(evidence_result: dict, chunks: list[dict] | None = None) -> dict:
    """Combine the Evidence Agent result with retrieved findings into an
    answer-level confidence object.

    chunks: retrieved vector hits; finding-type hits may carry 'confidence' and
    'source' in their metadata, which we use as the rule-confidence and
    corroboration signals.
    """
    scope = evidence_result.get("answer_scope", "none")
    present = evidence_result.get("evidence_present", []) or []
    missing = evidence_result.get("missing_evidence", []) or []

    total = len(present) + len(missing)
    coverage = (len(present) / total) if total else (1.0 if scope != "none" else 0.0)
    missing_frac = (len(missing) / total) if total else 0.0

    # Rule confidences + corroboration from retrieved findings.
    rule_confs: list[float] = []
    sources: set[str] = set()
    for c in chunks or []:
        meta = c.get("metadata", {}) or {}
        if meta.get("doc_type") == "finding":
            if isinstance(meta.get("confidence"), (int, float)):
                rule_confs.append(float(meta["confidence"]))
            if meta.get("source"):
                sources.add(meta["source"])
    mean_rule = sum(rule_confs) / len(rule_confs) if rule_confs else 0.0
    corroborated = {"rule_engine", "oracle_addm"}.issubset(sources)

    if scope == "none":
        # Refusal: confidence in the *answer* is low by construction.
        score = min(evidence_result.get("confidence", 0.2), 0.3)
    else:
        score = (W_EVIDENCE * coverage
                 + W_RULES * mean_rule
                 + W_CORROB * (1.0 if corroborated else (0.4 if rule_confs else 0.0))
                 - W_MISSING * missing_frac)
        # Floor at the evidence agent's base so a fully-grounded answer with no
        # retrieved findings still scores reasonably.
        score = max(score, evidence_result.get("confidence", 0.0) * 0.9)
    score = round(max(0.0, min(1.0, score)), 2)

    return {
        "confidence": score,
        "band": _band(score),
        "classification": _classification(scope, score, corroborated),
        "coverage": round(coverage, 2),
        "corroborated": corroborated,
        "rules_used": len(rule_confs),
        "evidence_sources": sorted(sources),
    }

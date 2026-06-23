#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rule_engine.py
==============
Knowledge Rules Engine for the AI-powered DBA Assistant.

Loads correlated diagnostic rules from rules.yaml (data, not code) and
evaluates them against the flat `signals` produced by the derived metrics
layer. Emits structured findings:

    {
      "category": "CPU",
      "severity": "HIGH",
      "finding": "CPU Bottleneck",
      "evidence": ["DB CPU accounts for 82.4% of DB time", ...],
      "recommendation": "...",
      "confidence": 0.83,
      "rule_id": "cpu_bottleneck"
    }

Rules are evaluated dynamically: editing rules.yaml changes behaviour with no
code change. Findings are ordered by severity (CRITICAL→LOW) then confidence.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("rule_engine")

_DEFAULT_RULES = Path(__file__).with_name("rules.yaml")

_SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}

_TEMPLATE_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


# ── condition evaluation ─────────────────────────────────────────────────────

def _cmp(value, op: str, target) -> bool:
    """Evaluate a single condition operator, null-safe."""
    if op == "is_null":
        return value is None
    if op == "not_null":
        return value is not None
    if value is None:
        return False  # any numeric/comparison op on a missing signal is False
    try:
        if op == ">=":
            return value >= target
        if op == "<=":
            return value <= target
        if op == ">":
            return value > target
        if op == "<":
            return value < target
        if op == "==":
            return value == target
        if op == "!=":
            return value != target
        if op == "in":
            return value in target
    except TypeError:
        return False
    raise ValueError(f"Unknown operator: {op}")


def _eval_condition(cond: dict, signals: dict) -> bool:
    return _cmp(signals.get(cond["signal"]), cond["op"], cond.get("value"))


# ── severity & evidence ──────────────────────────────────────────────────────

def _resolve_severity(spec: Optional[dict], signals: dict) -> str:
    """Resolve severity from a static level or signal bands."""
    if not spec:
        return "MEDIUM"
    if "static" in spec:
        return spec["static"]
    val = signals.get(spec.get("signal"))
    if val is None:
        return "MEDIUM"
    for band in spec.get("bands", []):
        if "min" in band and val >= band["min"]:
            return band["level"]
        if "max" in band and val <= band["max"]:
            return band["level"]
    return "LOW"


def _render_evidence(templates: list, signals: dict) -> list:
    """Fill evidence templates; drop any line referencing a null signal."""
    out = []
    for tmpl in templates or []:
        refs = _TEMPLATE_RE.findall(tmpl)
        if any(signals.get(r) is None for r in refs):
            continue  # skip lines we can't fully populate
        line = _TEMPLATE_RE.sub(lambda m: str(signals.get(m.group(1))), tmpl)
        out.append(line)
    return out


# ── public API ───────────────────────────────────────────────────────────────

class RuleEngine:
    """Loads rules once and evaluates them against signal dicts."""

    def __init__(self, rules_path: str | Path = _DEFAULT_RULES):
        self.rules_path = Path(rules_path)
        with open(self.rules_path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        self.rules = doc.get("rules", [])
        log.info("Loaded %d rules from %s", len(self.rules), self.rules_path.name)

    def evaluate(self, signals: dict) -> list:
        findings = []
        for rule in self.rules:
            # All `when` conditions must hold.
            when = rule.get("when", [])
            if not all(_eval_condition(c, signals) for c in when):
                continue
            # `any_of` corroboration (if present, at least one must hold).
            any_of = rule.get("any_of")
            corroborations = 0
            if any_of:
                results = [_eval_condition(c, signals) for c in any_of]
                if not any(results):
                    continue
                corroborations = sum(results)

            evidence = _render_evidence(rule.get("evidence", []), signals)
            severity = _resolve_severity(rule.get("severity"), signals)

            # Confidence: more corroborating signals + more populated evidence
            # => higher confidence. Bounded to [0.5, 0.99].
            total_corr = len(any_of) if any_of else 0
            corr_frac = (corroborations / total_corr) if total_corr else 1.0
            ev_frac = len(evidence) / max(len(rule.get("evidence", [])), 1)
            confidence = round(min(0.99, 0.5 + 0.3 * corr_frac + 0.2 * ev_frac), 2)

            findings.append({
                "category": rule.get("category"),
                "severity": severity,
                "finding": rule.get("finding"),
                "evidence": evidence,
                "recommendation": " ".join(rule.get("recommendation", "").split()),
                "confidence": confidence,
                "rule_id": rule.get("id"),
            })

        findings.sort(
            key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 0), f["confidence"]),
            reverse=True,
        )
        log.info("Rule evaluation produced %d finding(s).", len(findings))
        return findings


def evaluate_signals(signals: dict, rules_path: str | Path = _DEFAULT_RULES) -> list:
    """Convenience one-shot wrapper."""
    return RuleEngine(rules_path).evaluate(signals)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
question_classifier.py
======================
Maps a user question to an INTENT label defined in evidence_requirements.yaml.

Deterministic by design: it is a keyword/regex router, not an LLM call. The
intent decides what evidence the question requires (see evidence_agent), and the
*decision to refuse* must not itself depend on a model that might hallucinate.
The first intent (in file order) with a matching keyword wins, so the YAML order
encodes priority — most specific intents first, `general` last as the catch-all.

An optional LLM fallback can be layered on later for ambiguous questions, but it
must still be constrained to choose a label that exists in the YAML.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

log = logging.getLogger("question_classifier")

_DEFAULT_REQS = Path(__file__).with_name("evidence_requirements.yaml")


class QuestionClassifier:
    def __init__(self, reqs_path: str | Path = _DEFAULT_REQS):
        with open(reqs_path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        self.intents: dict = doc.get("intents", {})
        # Pre-compile match patterns per intent, preserving file order.
        self._compiled: list[tuple[str, list[re.Pattern]]] = []
        for name, spec in self.intents.items():
            pats = [re.compile(p, re.I) for p in (spec.get("match") or [])]
            self._compiled.append((name, pats))
        log.info("Loaded %d intents from %s", len(self.intents), Path(reqs_path).name)

    def classify(self, question: str) -> dict:
        """Return {intent, matched, spec}. Falls back to 'general'."""
        q = question or ""
        for name, pats in self._compiled:
            for p in pats:
                if p.search(q):
                    return {"intent": name, "matched": p.pattern,
                            "spec": self.intents.get(name, {})}
        return {"intent": "general", "matched": None,
                "spec": self.intents.get("general", {})}


def classify(question: str, reqs_path: str | Path = _DEFAULT_REQS) -> dict:
    """One-shot convenience wrapper."""
    return QuestionClassifier(reqs_path).classify(question)


if __name__ == "__main__":
    import sys
    qc = QuestionClassifier()
    for q in sys.argv[1:] or [
        "Which SQL statements would benefit most from indexing?",
        "Are any SQL statements performing full table scans?",
        "Look at 2bfwrkh7ttm3n. Why might this UPDATE not be using an index even if one exists?",
        "What is causing high CPU usage?",
        "Show CPU trend over the last 30 days.",
    ]:
        r = qc.classify(q)
        print(f"[{r['intent']:<16}] {q}")

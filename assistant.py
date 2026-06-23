#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assistant.py
============
The sequential answer pipeline (Phase 3).

Per the Agent Guidance: NO supervisor/orchestrator. This is a plain, ordered
sequence of function calls — the four agents run one after another, and the
Metrics Agent is invoked inline as the RCA agent needs it:

    Question
      -> Question Classifier + Evidence Agent   (answerable? what's missing?)
      -> [refuse here if not answerable]
      -> Metrics Agent      (facts only)
      -> RCA Agent          (cited root causes)
      -> Recommendation Agent (prioritized actions, never invents plans)
      -> answer contract     (findings · evidence · confidence · reasoning ·
                              missing · recommendations)

It is LLM-free and deterministic, so the whole pipeline is unit-testable offline.
An optional narration step (in app.py) may rephrase the contract for chat, but it
is constrained to the contract's own facts.
"""

from __future__ import annotations

import logging

import confidence
from evidence_agent import EvidenceAgent, available_evidence
from metrics_agent import MetricsAgent
from rca_agent import RCAAgent
from recommendation_agent import RecommendationAgent

log = logging.getLogger("assistant")


class DBAAssistant:
    def __init__(self, metrics_store):
        self.store = metrics_store
        self.evidence = EvidenceAgent()
        self.metrics = MetricsAgent(metrics_store)
        self.rca = RCAAgent()
        self.recs = RecommendationAgent()

    def answer(self, question: str, vector_chunks: list[dict] | None = None) -> dict:
        # 1) Classify + gate.
        gate = self.evidence.assess(question, available_evidence(self.store))

        if not gate["can_answer"]:
            return self._contract(
                question, gate,
                findings=["Cannot determine from the available evidence."],
                evidence=[], reasoning=gate["reason"],
                recommendations=self._refusal_next_steps(gate),
                conf={"confidence": gate["confidence"], "band": "Insufficient",
                      "classification": "speculative", "evidence_sources": []},
                bundle=None)

        # 2) Facts. 3) RCA. 4) Recommendations.
        bundle = self.metrics.gather(question, intent=gate["intent"])
        if bundle is None:
            return self._contract(
                question, gate,
                findings=["No ingested snapshots to analyse."],
                evidence=[], reasoning="The metrics store is empty.",
                recommendations=[], conf={"confidence": 0.0, "band": "Insufficient",
                                          "classification": "speculative", "evidence_sources": []},
                bundle=None)

        rca = self.rca.analyze(question, gate, bundle)
        recommendations = self.recs.recommend(rca, gate, bundle)

        # 5) Confidence: use retrieved findings (or the bundle's findings offline).
        chunks = vector_chunks or [
            {"metadata": {"doc_type": "finding", "confidence": f.get("confidence"),
                          "source": f.get("source")}}
            for f in bundle.get("findings", [])
        ]
        conf = confidence.compute(gate, chunks)

        # Keep the answer concise: lead with the headline cause, cap the rest.
        findings = [c["cause"] for c in rca["root_causes"]][:5]
        evidence = _dedupe([e for c in rca["root_causes"] for e in c["evidence"]])[:8]
        return self._contract(question, gate, findings=findings, evidence=evidence,
                              reasoning=rca["reasoning"], recommendations=recommendations,
                              conf=conf, bundle=bundle, rca=rca)

    @staticmethod
    def _refusal_next_steps(gate) -> list[dict]:
        steps = []
        if "execution_plan" in (gate.get("missing_evidence") or []):
            steps.append({"action": "Capture DBMS_XPLAN or a SQL Monitor report for the "
                                    "SQL of interest, then re-ask.",
                          "priority": 1, "requires": "execution_plan",
                          "impact": "enables access-path analysis"})
        if "two_or_more_snapshots" in (gate.get("missing_evidence") or []):
            steps.append({"action": "Ingest at least one more AWR snapshot to enable "
                                    "historical comparison.", "priority": 1,
                          "impact": "enables trend/comparison"})
        return steps

    @staticmethod
    def _contract(question, gate, *, findings, evidence, reasoning, recommendations,
                  conf, bundle, rca=None) -> dict:
        snap = (bundle or {}).get("snapshot") if bundle else None
        return {
            "question": question,
            "intent": gate.get("intent"),
            "snapshot": snap,
            "findings": findings,
            "evidence": evidence,
            "confidence": {
                "score": conf.get("confidence"),
                "band": conf.get("band"),
                "classification": conf.get("classification"),
                "evidence_sources": conf.get("evidence_sources", []),
            },
            "reasoning": reasoning,
            "missing_evidence": gate.get("missing_evidence", []),
            "answer_scope": gate.get("answer_scope"),
            "recommendations": recommendations,
            "root_causes": (rca or {}).get("root_causes", []),
        }


def render_markdown(c: dict) -> str:
    """Render an answer contract as the six-part copilot answer."""
    out = []
    out.append("### Findings")
    if c["findings"]:
        out.extend(f"- {f}" for f in c["findings"])
    else:
        out.append("- (none)")

    out.append("\n### Supporting Evidence")
    if c["evidence"]:
        out.extend(f"- {e}" for e in c["evidence"])
    else:
        out.append("- (no measured evidence available)")

    out.append("\n### Reasoning")
    out.append(c["reasoning"] or "(none)")

    out.append("\n### Recommendations")
    if c["recommendations"]:
        for r in c["recommendations"]:
            req = f" _(requires {r['requires']})_" if r.get("requires") else ""
            out.append(f"{r['priority']}. {r['action']}{req}")
    else:
        out.append("- (none)")

    conf = c["confidence"]
    out.append("\n---")
    out.append(f"**Confidence:** {conf['band']} "
               f"({conf['score']:.2f} · *{conf['classification']}*) · "
               f"scope: {c['answer_scope']}")
    if conf.get("evidence_sources"):
        out.append(f"**Evidence sources:** {', '.join(conf['evidence_sources'])}")
    if c["missing_evidence"]:
        out.append(f"**Missing evidence:** {', '.join(c['missing_evidence'])}")
    return "\n".join(out)


def narration_context(c: dict) -> str:
    """Compact, fact-only context for an OPTIONAL LLM narration. The model must
    summarise these and add nothing."""
    lines = [f"INTENT: {c['intent']} | SCOPE: {c['answer_scope']}"]
    lines.append("ROOT CAUSES:")
    for rc in c.get("root_causes", []):
        lines.append(f"- ({rc['classification']}) {rc['cause']}")
        for e in rc["evidence"][:4]:
            lines.append(f"    evidence: {e}")
    lines.append("RECOMMENDATIONS:")
    for r in c["recommendations"]:
        lines.append(f"- {r['action']}")
    if c["missing_evidence"]:
        lines.append(f"MISSING EVIDENCE: {', '.join(c['missing_evidence'])}")
    return "\n".join(lines)


def _dedupe(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


if __name__ == "__main__":
    import sys, json
    from metrics_store import MetricsStore
    logging.basicConfig(level=logging.WARNING)
    store = MetricsStore(sys.argv[1] if len(sys.argv) > 1 else "dba_assistant.db")
    asst = DBAAssistant(store)
    for q in sys.argv[2:] or [
        "Are any SQL statements performing full table scans?",
        "Which SQL statements would benefit most from indexing?",
        "What is causing high CPU usage?",
    ]:
        print("=" * 70, "\nQ:", q)
        print(render_markdown(asst.answer(q)))
        print()

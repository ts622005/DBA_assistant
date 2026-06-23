#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
addm_parser.py
==============
Parses Oracle ADDM findings that live as a TEXT block inside the AWR HTML
report (normally a <pre> section produced by addmrpt.sql / DBMS_ADVISOR).

ADDM is fundamentally different from the rest of the AWR report. The other
sections are raw metric *tables* that our rule engine turns into findings.
ADDM is Oracle's OWN analysis: it already arrives as prose findings, each with
an impact %, recommendations, actions and rationale. So this module does NOT
compute anything — it lifts Oracle's findings out of the text and normalises
them into the SAME shape the rule engine emits:

    {category, severity, finding, evidence[], recommendation,
     source: "oracle_addm", impact_pct, confidence, rule_id}

Both ADDM findings and rule-engine findings then flow into one findings list
and one vector store, tagged by `source`, so the RCA agent can cross-check
Oracle's verdict against the rules' independent verdict.

Two ADDM text layouts are handled:
  * older 10g/11g:  "FINDING n: 67% impact (1234 seconds)"
  * newer  12c/19c: "Finding n: <title>"  +  "Impact is X active sessions, Y% of total activity."
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

log = logging.getLogger("addm_parser")

_FINDING_HEAD = re.compile(r"^\s*FINDING\s+\d+\s*:", re.I | re.M)


# ── normalisation helpers ───────────────────────────────────────────────────

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _severity_from_impact(pct):
    """Map ADDM impact% onto the rule engine's severity vocabulary."""
    if pct is None:
        return "MEDIUM"
    if pct >= 50:
        return "CRITICAL"
    if pct >= 25:
        return "HIGH"
    if pct >= 10:
        return "MEDIUM"
    return "LOW"


# Light keyword -> category map so ADDM findings unify with rule findings.
_CATEGORY_KEYWORDS = [
    ("Commit",      ("log file sync", "commit", "redo log")),
    ("Cluster",     ("interconnect", "global cache", "gc ", " rac")),
    ("Concurrency", ("buffer busy", "latch", "concurrency", "enqueue", " lock", "contention")),
    ("Parse",       ("hard pars", "soft pars", "cursor", "bind variable", "shared pool", "library cache")),
    ("IO",          ("i/o", "disk", "storage", "single block read", "sequential read", "wait class \"user i/o\"")),
    ("Memory",      ("buffer cache", "undersized", "pga", "sga", "memory")),
    ("SQL",         ("sql statement", "sql_id", "sql tuning", "top sql")),
    ("CPU",         ("cpu", "processor")),
]


def _classify(text: str) -> str:
    t = " " + text.lower() + " "
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(k in t for k in kws):
            return cat
    return "ADDM"


def _impact_pct(block: str):
    """Pull the impact percentage out of either ADDM layout."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*impact", block, re.I)          # older
    if m:
        return float(m.group(1))
    m = re.search(r"impact is[^.%]*?(\d+(?:\.\d+)?)\s*%\s*of total activity", block, re.I)  # newer
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of total activity", block, re.I)
    if m:
        return float(m.group(1))
    return None


# ── locate the ADDM text in the HTML ────────────────────────────────────────

def _extract_addm_text(soup: BeautifulSoup) -> str:
    # ADDM text is almost always inside <pre> (preserves the indented layout).
    chunks = []
    for pre in soup.find_all("pre"):
        txt = pre.get_text("\n")
        if _FINDING_HEAD.search(txt) or "ADDM Report" in txt:
            chunks.append(txt)
    if chunks:
        return "\n".join(chunks)
    # Fallback: the whole page text, only if it clearly contains ADDM findings.
    full = soup.get_text("\n")
    return full if _FINDING_HEAD.search(full) else ""


# ── parse one finding block ─────────────────────────────────────────────────

def _is_separator(s: str) -> bool:
    """True for rule/separator lines like '------', '~~~~~~', '======'."""
    return bool(s) and all(ch in "-~=_." for ch in s)


def _parse_block(block: str) -> dict:
    lines = block.splitlines()

    # Title from the FINDING/Finding header line (newer layout carries it here).
    title, start = "", 0
    for i, ln in enumerate(lines):
        m = re.match(r"\s*FINDING\s+\d+\s*:\s*(.*)", ln, re.I)
        if m:
            title = _clean(m.group(1))
            title = re.sub(r"\d+(?:\.\d+)?\s*%\s*impact.*$", "", title, flags=re.I).strip()
            start = i + 1
            break

    impact = _impact_pct(block)

    desc, actions, rationales, symptoms = [], [], [], []
    rec_types = []
    mode = "desc"
    for ln in lines[start:]:
        s = ln.strip()
        if not s or _is_separator(s):                       # blank or rule line
            continue
        up = s.upper()

        # Recommendation header — may carry a type after the colon.
        mrec = re.match(r"RECOMMENDATION\s+\d+\s*:?\s*(.*)$", up)
        if mrec:
            if mrec.group(1).strip():
                rec_types.append(_clean(s.split(":", 1)[1]) if ":" in s else "")
            mode = "rec"
            continue
        # Action / Rationale / Related Object — handle BOTH "Action: text" and a
        # bare "Action" header with the text on the following indented lines.
        if up == "ACTION" or up.startswith("ACTION:"):
            inline = s.split(":", 1)[1].strip() if ":" in s else ""
            actions.append(_clean(inline) if inline else "")
            mode = "action"; continue
        if up == "RATIONALE" or up.startswith("RATIONALE:"):
            inline = s.split(":", 1)[1].strip() if ":" in s else ""
            rationales.append(_clean(inline) if inline else "")
            mode = "rationale"; continue
        if up.startswith("RELATED OBJECT") or up.startswith("ADDITIONAL INFORMATION"):
            mode = "skip"; continue
        if up.startswith("SYMPTOMS THAT LED TO THE FINDING"):
            mode = "symptoms"; continue
        if up.startswith("IMPACT IS") or up.startswith("IMPACT:") or up.startswith("ESTIMATED BENEFIT"):
            continue                                        # captured / not needed
        # continuation / content line
        if mode == "desc":
            desc.append(s)
        elif mode == "action":
            actions[-1] = _clean((actions[-1] + " " + s).strip())
        elif mode == "rationale":
            rationales[-1] = _clean((rationales[-1] + " " + s).strip())
        elif mode == "symptoms":
            symptoms.append(_clean(s))
        # mode in ("rec", "skip"): ignore stray lines

    actions = [a for a in actions if a]                     # drop empties
    rationales = [r for r in rationales if r]
    symptoms = [s for s in symptoms if not _is_separator(s)]

    context = _clean(" ".join(desc))
    finding_text = title or context or "ADDM finding"

    evidence = []
    if impact is not None:
        evidence.append(f"ADDM impact: {impact}% of total DB activity")
    if title and context:
        evidence.append(context)
    evidence += [f"Symptom: {s}" for s in symptoms]
    evidence += [f"Rationale: {r}" for r in rationales]

    recommendation = (
        "  ".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
        if actions else "Refer to the ADDM report for the recommended action."
    )

    return {
        "category": _classify(f"{finding_text} {context} {' '.join(actions)}"),
        "severity": _severity_from_impact(impact),
        "finding": finding_text,
        "evidence": evidence,
        "recommendation": recommendation,
        "source": "oracle_addm",
        "impact_pct": impact,
        # ADDM is Oracle's own verdict -> high baseline confidence, nudged by impact.
        "confidence": round(min(0.99, 0.9 + (impact or 0) / 1000), 2),
        "rule_id": None,
    }


# ── public entry point ──────────────────────────────────────────────────────

def parse_addm(html_path: str) -> list[dict]:
    """Extract and normalise ADDM findings from an AWR HTML report."""
    html = Path(html_path).read_text(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_addm_text(soup)
    if not text:
        log.info("No ADDM section detected in %s", Path(html_path).name)
        return []

    parts = re.split(r"(?=^\s*FINDING\s+\d+\s*:)", text, flags=re.I | re.M)
    raw = [
        _parse_block(p.rstrip())
        for p in parts
        if re.match(r"\s*FINDING\s+\d+\s*:", p, re.I)
    ]

    # A 24h report contains one ADDM task per interval, so the same finding
    # recurs many times. Collapse by finding name, keeping the highest-impact
    # instance, and record how many intervals it appeared in.
    best: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for f in raw:
        key = f["finding"].lower()
        counts[key] = counts.get(key, 0) + 1
        if key not in best or (f.get("impact_pct") or 0) > (best[key].get("impact_pct") or 0):
            best[key] = f
    findings = []
    for key, f in best.items():
        if counts[key] > 1:
            f = dict(f)
            f["evidence"] = [f"Recurred in {counts[key]} ADDM intervals "
                             f"(peak impact shown)"] + f["evidence"]
        findings.append(f)
    findings.sort(key=lambda f: -(f.get("impact_pct") or 0))

    log.info("ADDM parser: %d raw finding(s) across intervals -> %d unique.",
             len(raw), len(findings))
    return findings


if __name__ == "__main__":
    import argparse
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Parse ADDM findings from an AWR HTML report.")
    ap.add_argument("awr_html")
    args = ap.parse_args()
    out = parse_addm(args.awr_html)
    print(json.dumps(out, indent=2))
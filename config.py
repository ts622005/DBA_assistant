#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Single source of truth for cross-cutting configuration of the DBA Assistant.

The most important job of this module is to PIN THE EMBEDDER IN ONE PLACE.

Background (the bug this fixes):
    app.py used OllamaEmbedder("nomic-embed-text")  -> 768-dim
    the CLI paths used BGEEmbedder("bge-small-en-v1.5") -> 384-dim
Querying a Chroma collection that was BUILT with one model using vectors from
ANOTHER model returns semantically meaningless neighbours. Both the ingest path
and the query path must use the SAME embedder. From now on, everyone calls
`config.get_embedder()` and nobody constructs an embedder directly.

Override the backend without touching code via the DBA_EMBEDDER env var:
    DBA_EMBEDDER=bge                    (default; local sentence-transformers)
    DBA_EMBEDDER=ollama:nomic-embed-text
    DBA_EMBEDDER=ollama:bge-m3
"""

from __future__ import annotations

import os

# ── Paths ────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.environ.get("DBA_CHROMA_DIR", "./chroma_awr")
METRICS_DB = os.environ.get("DBA_METRICS_DB", "./dba_assistant.db")
COLLECTION = os.environ.get("DBA_COLLECTION", "awr_dba")

# ── LLM (answer generation) ──────────────────────────────────────────────────
OLLAMA_MODEL = os.environ.get("DBA_OLLAMA_MODEL", "llama3")
OLLAMA_API = os.environ.get("DBA_OLLAMA_API", "http://localhost:11434")

# ── Embedder selection (the single pin) ──────────────────────────────────────
# Default to local BGE: self-contained, deterministic, no network egress.
_EMBEDDER_SPEC = os.environ.get("DBA_EMBEDDER", "ollama:nomic-embed-text")

# Known dimensions so we can tag/verify the Chroma collection.
_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
}


def _parse_spec(spec: str) -> tuple[str, str]:
    """Return (backend, model). 'bge' -> ('bge','bge-small-en-v1.5')."""
    spec = (spec or "bge").strip()
    if spec.lower() in ("bge", "bge-small", "bge-small-en-v1.5", "baai/bge-small-en-v1.5"):
        return "bge", "BAAI/bge-small-en-v1.5"
    if spec.lower().startswith("ollama:"):
        return "ollama", spec.split(":", 1)[1].strip()
    # Bare model name -> assume local sentence-transformers.
    return "bge", spec


EMBEDDER_BACKEND, EMBEDDER_MODEL = _parse_spec(_EMBEDDER_SPEC)

# Stable identifier stored in the Chroma collection metadata. If a collection was
# built with a different id, the store refuses to use it (prevents silent garbage).
EMBEDDER_ID = f"{EMBEDDER_BACKEND}:{EMBEDDER_MODEL}"
EMBEDDER_DIM = _KNOWN_DIMS.get(EMBEDDER_MODEL)  # may be None for unknown models


def get_embedder():
    """Construct the canonical embedder. Imported lazily so the structured-store
    path (SQLite only) never needs sentence-transformers / ollama installed."""
    from vector_store import BGEEmbedder, OllamaEmbedder  # lazy

    if EMBEDDER_BACKEND == "ollama":
        return OllamaEmbedder(model=EMBEDDER_MODEL, host=OLLAMA_API)
    return BGEEmbedder(model_name=EMBEDDER_MODEL)

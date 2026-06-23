#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vector_store.py
===============
Embeddings + Vector DB layer for the AI-powered DBA Assistant.

    findings + metrics + top SQL  --(BAAI/bge-small-en-v1.5)-->  ChromaDB
                                                                    |
                                              retrieval for the AI DBA agents

What gets embedded (in priority order, for retrieval quality):
  1. Each FINDING  — the primary unit the assistant retrieves to answer
     "what is wrong / how do I fix it". One document per finding.
  2. A compact METRICS SUMMARY — for grounding factual questions about the run.
  3. Top SQL statements — so the SQL agent can retrieve the offenders.

Embeddings: BAAI/bge-small-en-v1.5 (384-dim) via sentence-transformers, run
LOCALLY (no external API), matching the project's no-egress constraint. BGE
retrieval best practice is applied: the short query-instruction prefix is added
to QUERIES only, never to stored passages. Cosine space + normalised vectors.

The embedder is pluggable: swap in an Ollama embedding model, or inject a stub
for testing, without touching the store logic.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Protocol

import chromadb

log = logging.getLogger("vector_store")

BGE_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 retrieval instruction — applied to queries only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ════════════════════════════════════════════════════════════════════════════
# Embedders
# ════════════════════════════════════════════════════════════════════════════

class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class BGEEmbedder:
    """BAAI/bge-small-en-v1.5 via sentence-transformers, running locally."""

    def __init__(self, model_name: str = BGE_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # lazy import
        log.info("Loading embedding model %s on %s …", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # Passages: NO instruction prefix; normalise for cosine.
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            BGE_QUERY_INSTRUCTION + text,
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()


class OllamaEmbedder:
    """Alternative: embeddings served by a local Ollama model (no egress).

    Pull a compatible model first, e.g.:  ollama pull bge-m3   (or nomic-embed-text)
    """

    def __init__(self, model: str = "bge-m3", host: str = "http://localhost:11434"):
        import ollama  # lazy import
        self._client = ollama.Client(host=host)
        self.model = model

    def _embed(self, text: str) -> list[float]:
        return self._client.embeddings(model=self.model, prompt=text)["embedding"]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashEmbedder:
    """Deterministic, dependency-free embedder for offline tests ONLY.

    Not semantically meaningful — used to validate the store/retrieve plumbing
    where the real model weights aren't available.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        import math
        out = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ════════════════════════════════════════════════════════════════════════════
# Document construction
# ════════════════════════════════════════════════════════════════════════════

def _clean_meta(d: dict) -> dict:
    """Chroma metadata must be str/int/float/bool — drop None, coerce rest."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def build_documents(analysis: dict, top_sql_per_cat: int = 5) -> list[dict]:
    """Turn an analysis report into retrieval documents (id, text, metadata)."""
    meta = analysis.get("meta", {}) or {}
    base = {
        "report_id": meta.get("report_id"),     # scopes /ask queries to one report
        "db_name": meta.get("db_name"),
        "report": meta.get("report_name"),
        "begin_snap": meta.get("begin_snap"),
        "end_snap": meta.get("end_snap"),
        "source_file": meta.get("source_file"),
    }
    # Prefer report_id for the id prefix so reports never overwrite each other.
    rid = meta.get("report_id") or meta.get("report_name") or "awr"
    docs: list[dict] = []

    # 1) Findings — primary retrieval unit.
    for f in analysis.get("findings", []):
        evidence = " ".join(f.get("evidence", []))
        text = (
            f"Finding: {f['finding']} (category {f['category']}, severity {f['severity']}). "
            f"Evidence: {evidence} "
            f"Recommendation: {f.get('recommendation', '')}"
        )
        docs.append({
            "id": f"{rid}:finding:{f.get('rule_id') or f['finding']}",
            "text": text,
            "metadata": _clean_meta({
                **base, "doc_type": "finding", "category": f["category"],
                "severity": f["severity"], "rule_id": f.get("rule_id"),
                "confidence": f.get("confidence"),
                "source": f.get("source"), "impact_pct": f.get("impact_pct"),
            }),
        })

    # 2) Metrics summary — for factual grounding.
    dm = analysis.get("derived_metrics", {}) or {}
    sig = meta.get("signals", {}) or {}
    summary = (
        f"AWR performance summary for database {base.get('db_name')} "
        f"(snapshots {base.get('begin_snap')}-{base.get('end_snap')}). "
        f"DB CPU is {dm.get('db_cpu_pct_db_time')}% of DB time; "
        f"User I/O {sig.get('user_io_pct_db_time')}% of DB time; "
        f"hard parses {dm.get('hard_parse_pct')}% of parses; "
        f"average active sessions {dm.get('average_active_sessions')} on "
        f"{sig.get('num_cpus')} CPUs; "
        f"logical-to-physical read ratio {dm.get('logical_to_physical_ratio')}; "
        f"top wait class {sig.get('top_wait_class')} "
        f"({sig.get('top_wait_class_pct')}% of DB time); "
        f"library hit {sig.get('library_hit_pct')}%, buffer hit {sig.get('buffer_hit_pct')}%."
    )
    docs.append({
        "id": f"{rid}:metrics_summary",
        "text": summary,
        "metadata": _clean_meta({**base, "doc_type": "metrics_summary"}),
    })

    # 3) Top SQL offenders.
    for cat, rows in (analysis.get("top_sql", {}) or {}).items():
        for rank, s in enumerate(rows[:top_sql_per_cat]):
            text = (
                f"Top SQL by {cat.replace('_', ' ')} (rank {rank + 1}): "
                f"sql_id {s.get('sql_id')}, "
                f"elapsed {s.get('elapsed_time')}s, cpu {s.get('cpu_time')}s, "
                f"physical_reads {s.get('physical_reads')}, "
                f"executions {s.get('executions')}. "
                f"Text: {s.get('sql_text', '')}"
            )
            docs.append({
                "id": f"{rid}:sql:{cat}:{s.get('sql_id')}",
                "text": text,
                "metadata": _clean_meta({
                    **base, "doc_type": "sql", "sql_category": cat,
                    "sql_id": s.get("sql_id"), "rank": rank + 1,
                }),
            })
    return docs


# ════════════════════════════════════════════════════════════════════════════
# Store
# ════════════════════════════════════════════════════════════════════════════

class AWRVectorStore:
    """Persistent ChromaDB collection for AWR findings/metrics/SQL."""

    def __init__(self, embedder: Embedder,
                 persist_dir: str = "./chroma_awr",
                 collection: str = "awr_dba",
                 embedder_id: Optional[str] = None,
                 embedder_dim: Optional[int] = None):
        self.embedder = embedder
        self.embedder_id = embedder_id
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Tag the collection with the embedder identity so a collection built with
        # one model can never be queried with another (the 384 vs 768 dim bug).
        meta = {"hnsw:space": "cosine"}
        if embedder_id:
            meta["embedder_id"] = embedder_id
        if embedder_dim:
            meta["embedder_dim"] = embedder_dim
        self.col = self.client.get_or_create_collection(name=collection, metadata=meta)

        # Verify an EXISTING collection matches the current embedder.
        existing_id = (self.col.metadata or {}).get("embedder_id")
        if embedder_id and existing_id and existing_id != embedder_id:
            raise RuntimeError(
                f"Embedder mismatch for collection '{collection}': it was built with "
                f"'{existing_id}' but the current config uses '{embedder_id}'. "
                f"Rebuild the collection (delete '{persist_dir}' or set DBA_COLLECTION "
                f"to a new name) before ingesting/querying."
            )
        log.info("Chroma collection '%s' ready (%d docs, embedder=%s).",
                 collection, self.col.count(), existing_id or embedder_id or "untagged")

    def ingest(self, analysis: dict, top_sql_per_cat: int = 5) -> int:
        docs = build_documents(analysis, top_sql_per_cat=top_sql_per_cat)
        if not docs:
            return 0
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d["metadata"] for d in docs]
        embeddings = self.embedder.embed_passages(texts)
        self.col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        log.info("Ingested %d documents.", len(docs))
        return len(docs)

    def query(self, question: str, n_results: int = 4,
              where: Optional[dict] = None) -> list[dict]:
        q_emb = self.embedder.embed_query(question)
        res = self.col.query(
            query_embeddings=[q_emb], n_results=n_results, where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({
                "text": doc, "metadata": meta,
                "score": round(1 - dist, 4),  # cosine similarity
            })
        return hits


if __name__ == "__main__":
    import json
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Ingest an AWR analysis JSON into ChromaDB.")
    ap.add_argument("analysis_json", help="Output of pipeline.py")
    ap.add_argument("--persist", default="./chroma_awr")
    ap.add_argument("--query", default=None, help="Optional test query after ingest")
    args = ap.parse_args()

    import config
    analysis = json.load(open(args.analysis_json))
    store = AWRVectorStore(
        config.get_embedder(), persist_dir=args.persist,
        collection=config.COLLECTION,
        embedder_id=config.EMBEDDER_ID, embedder_dim=config.EMBEDDER_DIM,
    )
    store.ingest(analysis)
    if args.query:
        for h in store.query(args.query):
            print(f"\n[{h['score']}] {h['metadata'].get('doc_type')} "
                  f"{h['metadata'].get('category', '')}\n  {h['text'][:200]}")
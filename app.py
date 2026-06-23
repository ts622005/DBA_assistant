#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  —  AI DBA Assistant (Streamlit)

Flow:
    Upload AWR HTML  ->  pipeline  ->  ChromaDB
    User Question    ->  Retriever ->  Llama3 (Ollama)  ->  RCA + Recommendations
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path

import ollama
import requests
import streamlit as st

import config
import pipeline
from assistant import DBAAssistant, render_markdown, narration_context
from vector_store import AWRVectorStore
from metrics_store import MetricsStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")

st.set_page_config(page_title="AI DBA Assistant", page_icon="🔍", layout="wide")

OLLAMA_MODEL  = config.OLLAMA_MODEL
# Embedding model is pinned in config.py (single source of truth). app.py used to
# hardcode "nomic-embed-text" here while the CLI used BGE — that mismatch produced
# meaningless retrieval. We now always embed via config.get_embedder().
EMBED_MODEL   = config.EMBEDDER_MODEL if config.EMBEDDER_BACKEND == "ollama" else None
CHROMA_DIR    = config.CHROMA_DIR
OLLAMA_EXE    = r"C:\Users\Asus\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_API    = config.OLLAMA_API


# ── Ollama auto-start + model pull ───────────────────────────────────────────

def _ollama_ready() -> bool:
    try:
        return requests.get(f"{OLLAMA_API}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _model_available(model: str) -> bool:
    try:
        tags = requests.get(f"{OLLAMA_API}/api/tags", timeout=5).json()
        return any(model in m["name"] for m in tags.get("models", []))
    except Exception:
        return False


@st.cache_resource(show_spinner="Starting Ollama and pulling required models…")
def ensure_ollama() -> None:
    if not _ollama_ready():
        subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000,   # CREATE_NO_WINDOW — no console popup on Windows
        )
        for _ in range(30):             # wait up to 15 s
            time.sleep(0.5)
            if _ollama_ready():
                break
        else:
            st.error("Ollama did not start in time. Check the path or start it manually.")
            st.stop()

    # Pull the LLM, and the embedding model only if it is actually served by Ollama
    # (when config pins a local BGE embedder, EMBED_MODEL is None — nothing to pull).
    for model in (OLLAMA_MODEL, EMBED_MODEL):
        if not model:
            continue
        if not _model_available(model):
            for _ in ollama.pull(model, stream=True):  # streams progress, blocks until done
                pass

# The deterministic pipeline (assistant.py) produces the authoritative answer.
# The LLM is used ONLY to narrate that answer in fluent prose — it must add no new
# facts, causes, numbers, or recommendations beyond what the pipeline supplied.
NARRATION_SYSTEM = """You are an expert Oracle DBA assistant. You will be given a
STRUCTURED ANALYSIS (root causes, evidence, recommendations, missing evidence) that
was computed deterministically from an AWR report.

Your ONLY job is to summarise it in 2-4 sentences of clear prose for a DBA.

Strict rules:
- Add NOTHING that is not in the structured analysis: no new metrics, SQL IDs,
  causes, or recommendations.
- Never claim a full table scan, missing index, or plan problem unless it appears
  in the structured analysis (it won't, unless execution-plan data was available).
- If 'missing evidence' is listed, mention briefly that the conclusion is limited
  by it. Keep it short; the structured detail is shown to the user separately."""


# ── cached resources (survive Streamlit reruns) ──────────────────────────────

@st.cache_resource(show_spinner="Connecting to vector store…")
def get_store() -> AWRVectorStore:
    return AWRVectorStore(
        config.get_embedder(),
        persist_dir=CHROMA_DIR,
        collection=config.COLLECTION,
        embedder_id=config.EMBEDDER_ID,
        embedder_dim=config.EMBEDDER_DIM,
    )


@st.cache_resource(show_spinner="Opening metrics store…")
def get_metrics_store() -> MetricsStore:
    return MetricsStore(config.METRICS_DB)


@st.cache_resource(show_spinner="Loading reasoning pipeline…")
def get_assistant() -> DBAAssistant:
    return DBAAssistant(get_metrics_store())


# ── helpers ───────────────────────────────────────────────────────────────────

def run_pipeline_and_ingest(uploaded_file, store: AWRVectorStore,
                            metrics_store: MetricsStore | None = None) -> dict:
    suffix = Path(uploaded_file.name).suffix or ".html"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return pipeline.analyze_and_ingest(tmp_path, store, metrics_store=metrics_store)


def narrate(contract: dict):
    """Stream a short LLM narration constrained to the contract's own facts."""
    for chunk in ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": NARRATION_SYSTEM},
            {"role": "user", "content": narration_context(contract)},
        ],
        stream=True,
    ):
        yield chunk["message"]["content"]


# ── sidebar ───────────────────────────────────────────────────────────────────

ensure_ollama()   # no-op if already running; auto-starts otherwise

with st.sidebar:
    st.title("AI DBA Assistant")
    st.caption("AWR → ChromaDB → Llama3 → RCA")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload AWR HTML report(s)",
        type=["html", "htm"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process & Ingest", type="primary", use_container_width=True):
            store = get_store()
            metrics_store = get_metrics_store()
            if "ingested" not in st.session_state:
                st.session_state.ingested = {}

            for f in uploaded_files:
                if f.name in st.session_state.ingested:
                    st.info(f"{f.name} already ingested.")
                    continue
                with st.spinner(f"Processing {f.name}…"):
                    try:
                        report = run_pipeline_and_ingest(f, store, metrics_store)
                        findings = report.get("findings", [])
                        st.session_state.ingested[f.name] = {
                            "findings": len(findings),
                            "db": report.get("meta", {}).get("db_name", "—"),
                            "begin": report.get("meta", {}).get("begin_time", "—"),
                            "end":   report.get("meta", {}).get("end_time",   "—"),
                        }
                        st.success(f"Done — {len(findings)} finding(s)")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    st.divider()
    if "ingested" in st.session_state and st.session_state.ingested:
        st.subheader("Ingested Reports")
        for name, meta in st.session_state.ingested.items():
            with st.expander(name, expanded=False):
                st.write(f"**DB:** {meta['db']}")
                st.write(f"**Period:** {meta['begin']} → {meta['end']}")
                st.write(f"**Findings:** {meta['findings']}")
    else:
        st.info("No reports ingested yet.")

    st.divider()
    n_chunks = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=5)
    st.caption(f"Model: `{OLLAMA_MODEL}` via Ollama")


# ── main area ─────────────────────────────────────────────────────────────────

st.title("Ask the DBA Assistant")

no_data = "ingested" not in st.session_state or not st.session_state.ingested
if no_data:
    st.info("Upload and process an AWR HTML report from the sidebar to get started.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("Retrieved AWR context", expanded=False):
                for c in msg["chunks"]:
                    st.markdown(
                        f"**[{c['score']}] {c['metadata'].get('doc_type', '')}**  "
                        f"`{c['metadata'].get('category', c['metadata'].get('sql_category', ''))}`"
                    )
                    st.caption(c["text"][:300])
                    st.divider()

question = st.chat_input("e.g. What is causing high CPU usage? Which SQL is the top offender?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    store = get_store()
    assistant = get_assistant()

    # Retrieve vector context (best-effort) — used for confidence corroboration and
    # for the optional narration; the structured pipeline is the source of truth.
    chunks = []
    try:
        chunks = store.query(question, n_results=n_chunks)
    except Exception as e:
        logging.warning("vector query failed: %s", e)

    # ── Sequential pipeline (Phase 3): classify → gate → metrics → rca → rec ──
    contract = assistant.answer(question, vector_chunks=chunks or None)
    md = render_markdown(contract)

    with st.chat_message("assistant"):
        narrated = None
        # Optional LLM narration — constrained to the contract's own facts. Never
        # the source of truth; if Ollama is unavailable we just show the contract.
        if contract["answer_scope"] != "none":
            try:
                narrated = st.write_stream(narrate(contract))
                st.divider()
            except Exception as e:
                logging.info("narration skipped: %s", e)
        st.markdown(md)
        if chunks:
            with st.expander("Retrieved AWR context", expanded=False):
                for c in chunks:
                    st.markdown(
                        f"**[{c['score']}] {c['metadata'].get('doc_type', '')}**  "
                        f"`{c['metadata'].get('category', c['metadata'].get('sql_category', ''))}`"
                    )
                    st.caption(c["text"][:300])
                    st.divider()

    answer = (narrated + "\n\n" if narrated else "") + md
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "chunks": chunks,
    })

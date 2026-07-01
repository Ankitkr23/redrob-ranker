"""Redrob Ranker — hosted sandbox (Streamlit).

A small, self-contained demo that satisfies the Redrob submission-spec sandbox
requirement (Section 10.5): it accepts a candidate sample (<=100), runs the
*same* ranking pipeline end-to-end on CPU, offline, and returns a ranked CSV.

The sandbox uses the TF-IDF semantic backend so it stays fully portable — no
model download, no precomputed embeddings, no network — while exercising the
identical scoring / disqualifier / availability / reasoning code paths as the
full submission. Deploy on HuggingFace Spaces or Streamlit Cloud with
requirements-sandbox.txt.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker.pipeline import run  # noqa: E402

SAMPLE = ROOT / "data" / "sample_candidates.json"
JD = ROOT / "data" / "job_description.txt"

st.set_page_config(page_title="Redrob Candidate Ranker — Sandbox",
                   page_icon="🟦", layout="wide")

st.title("Redrob Intelligent Candidate Ranker — Sandbox")
st.caption(
    "Runs the real ranking pipeline end-to-end on a small candidate sample "
    "(≤100) — CPU-only, fully offline, TF-IDF backend. Same scoring, "
    "disqualifier, availability and grounded-reasoning code as the full "
    "100K submission."
)

with st.expander("Job description this sample is ranked against", expanded=False):
    st.text(JD.read_text(encoding="utf-8") if JD.exists() else "(JD file missing)")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload a candidate sample (JSON array or JSONL, ≤100 candidates). "
        "Leave empty to use the bundled 50-candidate sample.",
        type=["json", "jsonl"],
    )
with col2:
    top_n = st.number_input("Show top N", min_value=1, max_value=100, value=25, step=5)

if st.button("Rank candidates", type="primary"):
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "candidates.json"
        if uploaded is not None:
            inp.write_bytes(uploaded.getvalue())
            source = f"uploaded file ({uploaded.name})"
        else:
            inp.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
            source = "bundled sample_candidates.json (50 candidates)"

        out = Path(td) / "ranking.csv"
        try:
            t0 = time.time()
            rows = run(str(inp), str(JD), str(out), verbose=False,
                       semantic_backend="tfidf", top_n=int(top_n))
            dt = time.time() - t0
        except Exception as e:  # surface a clean error to the user
            st.error(f"Ranking failed: {e}")
            st.stop()

        st.success(
            f"Ranked {len(rows)} candidates from {source} in {dt:.2f}s "
            "(CPU · offline · TF-IDF)."
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ranked CSV",
            out.read_text(encoding="utf-8"),
            file_name="ranking.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "The graded 100K submission uses precomputed all-MiniLM-L6-v2 embeddings "
    "(`python rank.py --candidates ./candidates.jsonl --out ./submission.csv`). "
    "This sandbox demonstrates reproducibility on a small sample, as required by "
    "spec Section 10.5."
)

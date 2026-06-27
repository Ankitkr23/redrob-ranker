#!/usr/bin/env python3
"""Offline precomputation of candidate sentence embeddings.

This is the SLOW, SEPARATE step — it is NOT part of the 5-minute ranking
budget. It streams the candidate pool, builds each candidate's text document
(reusing features.build_text_doc so it matches ranking exactly), embeds them in
batches with a small CPU sentence-transformer, and writes:

    data/candidate_embeddings.npy   float32 matrix (N x 384), L2-normalized
    data/candidate_ids.json         candidate_id list in the SAME row order

The ranking step (rank.py --semantic-backend embeddings) then loads these and
only needs to embed the single JD string at run time.

Usage:
    python scripts/precompute_embeddings.py --candidates ./candidates.jsonl
    # ~52MB gzipped input also works directly:
    python scripts/precompute_embeddings.py --candidates ./candidates.jsonl.gz

Model: sentence-transformers/all-MiniLM-L6-v2 (~80MB, 384-dim, CPU-friendly).
The model is downloaded once on first run and cached locally by HuggingFace, so
subsequent runs (and the ranking step) work fully offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker.features import build_embedding_text  # noqa: E402
from redrob_ranker.io_utils import iter_candidates  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMB = ROOT / "data" / "candidate_embeddings.npy"
DEFAULT_IDS = ROOT / "data" / "candidate_ids.json"
# Compact embedding text is capped at ~64 words, so 96 tokens covers it with
# headroom while keeping transformer compute low. Must match
# retrieval_dense.MAX_SEQ_LENGTH.
MAX_SEQ_LENGTH = 96


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute candidate embeddings (offline).")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz].")
    ap.add_argument("--out-emb", default=str(DEFAULT_EMB))
    ap.add_argument("--out-ids", default=str(DEFAULT_IDS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    # Use all CPU cores for the matrix math; silence tokenizer fork warnings.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    torch.set_num_threads(args.threads)

    t0 = time.time()
    from sentence_transformers import SentenceTransformer  # heavy import, here on purpose

    print(f"loading model {args.model} (CPU, {args.threads} threads)...")
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH

    ids: list[str] = []
    docs: list[str] = []
    for raw in iter_candidates(args.candidates):
        ids.append(raw.get("candidate_id", ""))
        docs.append(build_embedding_text(raw))
    print(f"built {len(docs):,} candidate documents in {time.time()-t0:.1f}s; embedding...")

    t1 = time.time()
    emb = model.encode(
        docs,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    Path(args.out_emb).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, emb)
    with open(args.out_ids, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"embedded {emb.shape[0]:,} x {emb.shape[1]} in {time.time()-t1:.1f}s")
    print(f"saved -> {args.out_emb} ({emb.nbytes/1e6:.1f} MB) and {args.out_ids}")
    print(f"TOTAL precomputation: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

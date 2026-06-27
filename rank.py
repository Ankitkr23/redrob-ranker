#!/usr/bin/env python3
"""Produce the top-100 candidate ranking CSV from a candidate pool.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs CPU-only and fully offline (no network calls).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running directly from the repo without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from redrob_ranker.pipeline import run  # noqa: E402

DEFAULT_JD = str(Path(__file__).parent / "data" / "job_description.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Redrob intelligent candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl (or a JSON array file).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--jd", default=DEFAULT_JD,
                    help="Path to the job description text (defaults to bundled JD).")
    ap.add_argument("--semantic-backend", choices=["auto", "embeddings", "tfidf"],
                    default="auto",
                    help="Semantic similarity backend. 'auto' uses precomputed "
                         "embeddings if available, else falls back to TF-IDF.")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    args = ap.parse_args()

    run(args.candidates, args.jd, args.out, verbose=not args.quiet,
        semantic_backend=args.semantic_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

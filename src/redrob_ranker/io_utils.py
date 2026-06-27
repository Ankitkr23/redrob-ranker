"""Streaming IO for the candidate pool and the submission CSV.

The candidate file is ~465 MB / 100K lines, so we stream it line-by-line and
never hold raw JSON for the whole pool at once. The pipeline keeps only the
compact extracted features per candidate.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterator


def _open_text(path: Path):
    """Open a path as UTF-8 text, transparently decompressing .gz files.

    The official bundle ships the pool as candidates.jsonl.gz, so we accept it
    directly without requiring a separate `gunzip` step.
    """
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield candidate dicts from a JSONL, gzipped JSONL, or JSON array file.

    `candidates.jsonl[.gz]` is JSON-lines; `sample_candidates.json` is a JSON
    array. We auto-detect the structure so the same code path works for all.
    """
    p = Path(path)
    with _open_text(p) as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        f.seek(0)
        if first == "[":
            # JSON array (sample file).
            for obj in json.load(f):
                yield obj
            return
        # JSON-lines (full pool).
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_submission(rows: list[dict], out_path: str | Path) -> None:
    """Write the ranked rows to a spec-compliant CSV.

    `rows` must already be sorted by rank and contain keys:
    candidate_id, rank, score, reasoning.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([
                r["candidate_id"],
                int(r["rank"]),
                f"{float(r['score']):.4f}",
                r.get("reasoning", ""),
            ])

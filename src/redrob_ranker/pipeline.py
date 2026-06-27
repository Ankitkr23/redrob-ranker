"""End-to-end ranking pipeline.

  load JD rubric
   -> stream candidates: plausibility check + feature extraction
   -> semantic (TF-IDF) similarity to JD
   -> multi-signal scoring (+ disqualifiers + availability + honeypot demotion)
   -> deterministic ordering with spec-compliant tie-breaking
   -> top-100 with grounded reasoning
   -> CSV

Designed to run CPU-only, offline, within the 5-minute / 16 GB budget.
"""

from __future__ import annotations

import time

from . import features, plausibility, reasoning, retrieval, retrieval_dense, scoring
from .io_utils import iter_candidates, write_submission
from .jd_spec import RoleSpec

TOP_N = 100


def _semantic(docs, candidate_ids, role, backend: str, verbose: bool):
    """Select and run the semantic backend.

    backend: "auto" (embeddings if artifacts present + importable, else tfidf),
             "embeddings" (strict — errors if unavailable), or "tfidf".
    """
    if backend == "tfidf":
        return retrieval.semantic_scores(docs, role), "tfidf"
    if backend == "embeddings":
        return retrieval_dense.semantic_scores_dense(role, candidate_ids), "embeddings"
    # auto
    if retrieval_dense.embeddings_available():
        try:
            return retrieval_dense.semantic_scores_dense(role, candidate_ids), "embeddings"
        except Exception as e:  # not silent — explain the fallback
            if verbose:
                print(f"    (embeddings unavailable: {e}; falling back to tfidf)")
    return retrieval.semantic_scores(docs, role), "tfidf"


def run(candidates_path: str, jd_path: str, out_path: str, verbose: bool = True,
        semantic_backend: str = "auto") -> list[dict]:
    t0 = time.time()
    role = RoleSpec.load(jd_path)

    cands: list[features.Candidate] = []
    docs: list[str] = []
    n_honeypot = 0
    for raw in iter_candidates(candidates_path):
        is_hp, reasons = plausibility.detect(raw)
        c = features.extract(raw, role)
        c.is_honeypot = is_hp
        c.honeypot_reasons = reasons
        n_honeypot += int(is_hp)
        cands.append(c)
        docs.append(c.text_doc)

    if verbose:
        print(f"[1/4] parsed {len(cands):,} candidates "
              f"({n_honeypot} flagged as implausible) in {time.time()-t0:.1f}s")

    if len(cands) < TOP_N:
        raise ValueError(f"Need at least {TOP_N} candidates, got {len(cands)}.")

    t1 = time.time()
    candidate_ids = [c.candidate_id for c in cands]
    sem, used_backend = _semantic(docs, candidate_ids, role, semantic_backend, verbose)
    if verbose:
        print(f"[2/4] computed semantic similarity [{used_backend}] in {time.time()-t1:.1f}s")

    t2 = time.time()
    pool = scoring.PoolStats.from_candidates(cands)
    scored = []
    for c, s in zip(cands, sem):
        b = scoring.score(c, float(s), role, pool)
        scored.append((c, b))
    if verbose:
        print(f"[3/4] scored all candidates in {time.time()-t2:.1f}s")

    # primary order: score desc, then candidate_id asc (deterministic tie-break)
    scored.sort(key=lambda cb: (-cb[1].final, cb[0].candidate_id))
    top = scored[:TOP_N]

    # Enforce the validator's tie-break on the *rounded* score: within any block
    # of equal 4-dp scores, candidate_id must be ascending. Equal-rounded rows are
    # contiguous (full score is monotonic), so we reorder each block locally.
    rounded = [(c, b, round(b.final, 4)) for c, b in top]
    i = 0
    ordered: list[tuple] = []
    while i < len(rounded):
        j = i
        while j < len(rounded) and rounded[j][2] == rounded[i][2]:
            j += 1
        block = sorted(rounded[i:j], key=lambda x: x[0].candidate_id)
        ordered.extend(block)
        i = j

    rows = []
    for rank, (c, b, sc) in enumerate(ordered, start=1):
        rows.append({
            "candidate_id": c.candidate_id,
            "rank": rank,
            "score": sc,
            "reasoning": reasoning.generate(c, b, rank),
        })

    write_submission(rows, out_path)
    if verbose:
        hp_in_top = sum(1 for c, _, _ in ordered if c.is_honeypot)
        print(f"[4/4] wrote {len(rows)} rows to {out_path} "
              f"(honeypots in top-{TOP_N}: {hp_in_top}) — total {time.time()-t0:.1f}s")
    return rows

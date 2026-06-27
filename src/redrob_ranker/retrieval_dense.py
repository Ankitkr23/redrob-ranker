"""Dense (embeddings) semantic backend for the ranking step.

At ranking time this is cheap: it loads the precomputed candidate embedding
matrix (no per-candidate model inference) and embeds only the single JD string
once with a locally-cached sentence-transformer. Cosine similarity against 100K
x 384 vectors is a sub-second matrix-vector product.

Contract (per the challenge constraints):
  - Pure CPU, no network (the model is loaded from the local HuggingFace cache
    populated during precomputation).
  - If the precomputed artifacts are missing or don't match the candidate file,
    we raise a clear error rather than silently doing something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .jd_spec import RoleSpec

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Must match scripts/precompute_embeddings.MAX_SEQ_LENGTH so the JD vector lives
# in the same truncated space as the precomputed candidate vectors.
MAX_SEQ_LENGTH = 96

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMB = _ROOT / "data" / "candidate_embeddings.npy"
DEFAULT_IDS = _ROOT / "data" / "candidate_ids.json"

# Concise ideal-profile expansion, placed FIRST so it stays within the token
# budget (the full JD is long and would otherwise be truncated to its preamble).
_EXPANSION = (
    "Ideal profile: applied machine learning engineer with production experience in "
    "embeddings-based retrieval, ranking, recommendation systems, semantic search, "
    "vector databases and hybrid search, learning-to-rank, ranking evaluation "
    "(NDCG, MRR, MAP), A/B testing, NLP and information retrieval, LLM fine-tuning, "
    "shipped to real users at a product company, strong Python. "
)


def jd_query(role: RoleSpec) -> str:
    return _EXPANSION + (role.jd_text or "")


def embeddings_available(emb_path: str | Path = DEFAULT_EMB,
                         ids_path: str | Path = DEFAULT_IDS) -> bool:
    return Path(emb_path).exists() and Path(ids_path).exists()


def semantic_scores_dense(
    role: RoleSpec,
    candidate_ids: list[str],
    emb_path: str | Path = DEFAULT_EMB,
    ids_path: str | Path = DEFAULT_IDS,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Return cosine similarity in [0, 1] of each candidate to the JD, aligned
    to the order of `candidate_ids`."""
    emb_path, ids_path = Path(emb_path), Path(ids_path)
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Precomputed embeddings not found ({emb_path} / {ids_path}). "
            "Run:  python scripts/precompute_embeddings.py --candidates <pool> "
            "  (or use --semantic-backend tfidf)."
        )

    saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if saved_ids != candidate_ids:
        raise ValueError(
            f"Precomputed embeddings ({len(saved_ids)} ids) do not match the current "
            f"candidate file ({len(candidate_ids)} ids), or differ in order. "
            "Re-run scripts/precompute_embeddings.py on this exact candidate file."
        )

    mat = np.load(emb_path).astype("float32")  # expected pre-normalized, but be safe
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-9, None)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # explicit, not silent
        raise ImportError(
            "sentence-transformers is required for the embeddings backend "
            "(pip install -r requirements.txt), or use --semantic-backend tfidf."
        ) from e

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    q = model.encode([jd_query(role)], normalize_embeddings=True,
                     convert_to_numpy=True)[0].astype("float32")

    sims = mat @ q  # cosine in [-1, 1] (both sides L2-normalized)
    sims = np.clip(sims, 0.0, None)
    hi = float(np.quantile(sims, 0.999)) or 1.0
    return np.clip(sims / hi, 0.0, 1.0)

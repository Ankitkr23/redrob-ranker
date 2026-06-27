"""Semantic similarity between the JD and each candidate document.

We deliberately use a TF-IDF vector space rather than a neural embedding model:

  - It needs NO model weights and NO network, so the ranking step reproduces
    exactly inside the organizers' sandbox (CPU, 16 GB, no GPU, no network).
  - On 100K short professional documents it fits + scores in a few seconds.
  - Char-aware word n-grams capture the phrases that matter here ("recommendation
    system", "vector search", "learning to rank") which is precisely the
    "hidden gem" signal — evidence in free text, not just skill tags.

This is the recall/semantic backbone; precise role fit is handled by the
structured scorer. The two are fused in scoring.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .jd_spec import RoleSpec, normalize


def _build_query(role: RoleSpec) -> str:
    """Construct the semantic query: the JD text plus a distilled 'ideal profile'
    expansion so the vector emphasizes what the role *means*."""
    expansion = (
        " applied machine learning engineer embeddings retrieval ranking "
        "recommendation system semantic search vector database hybrid search "
        "learning to rank ndcg mrr evaluation a/b testing production ml at scale "
        "nlp information retrieval llm fine-tuning shipped to real users "
        "product company python "
    )
    return normalize(role.jd_text + " " + expansion)


def semantic_scores(docs: list[str], role: RoleSpec) -> np.ndarray:
    """Return cosine similarity in [0, 1] of each candidate doc to the JD query."""
    query = _build_query(role)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.6,
        max_features=120_000,
        sublinear_tf=True,
        norm="l2",
    )
    # Fit on the candidate corpus so the vocabulary reflects the pool; the JD is
    # appended last and transformed in the same space.
    matrix = vectorizer.fit_transform(docs + [query])
    cand_matrix = matrix[:-1]
    jd_vec = matrix[-1]
    sims = linear_kernel(cand_matrix, jd_vec).ravel()  # rows are L2-normed -> cosine
    # Scale to [0, 1] robustly (cosine is already >= 0 for tf-idf).
    hi = float(np.quantile(sims, 0.999)) or 1.0
    return np.clip(sims / hi, 0.0, 1.0)

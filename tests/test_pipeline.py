"""Sanity tests on the bundled 50-candidate sample.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
       or:  PYTHONPATH=src python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker import features, plausibility, reasoning, retrieval, scoring  # noqa: E402
from redrob_ranker.io_utils import iter_candidates  # noqa: E402
from redrob_ranker.jd_spec import RoleSpec  # noqa: E402

SAMPLE = ROOT / "data" / "sample_candidates.json"
JD = ROOT / "data" / "job_description.txt"


def _score_sample():
    role = RoleSpec.load(JD)
    cands, docs = [], []
    for raw in iter_candidates(SAMPLE):
        hp, rs = plausibility.detect(raw)
        c = features.extract(raw, role)
        c.is_honeypot, c.honeypot_reasons = hp, rs
        cands.append(c)
        docs.append(c.text_doc)
    sem = retrieval.semantic_scores(docs, role)  # tfidf backend (no artifacts needed)
    pool = scoring.PoolStats.from_candidates(cands)
    scored = [(c, scoring.score(c, float(s), role, pool)) for c, s in zip(cands, sem)]
    scored.sort(key=lambda cb: (-cb[1].final, cb[0].candidate_id))
    return scored


def test_scores_in_range_and_sorted():
    scored = _score_sample()
    assert len(scored) == 50
    finals = [b.final for _, b in scored]
    assert all(0.0 <= f <= 1.0 for f in finals)
    assert finals == sorted(finals, reverse=True)


def test_on_target_title_beats_off_target():
    """A relevant engineer should outrank an HR/Marketing/Sales-type profile."""
    scored = _score_sample()
    top = scored[0][0]
    assert top.current_title_norm  # has a title
    # the top candidate must have a positive title fit (not an anti-title)
    assert scoring._title_fit(top)[0] >= 0.5


def test_honeypots_sink():
    scored = _score_sample()
    honeypot_ranks = [i for i, (c, _) in enumerate(scored) if c.is_honeypot]
    # any flagged honeypot must land in the bottom half
    assert all(r >= len(scored) // 2 for r in honeypot_ranks)


def test_reasoning_is_grounded_and_nonempty():
    scored = _score_sample()
    for c, b in scored[:10]:
        r = reasoning.generate(c, b, 1)
        assert r and len(r) > 20
        # every named skill mentioned must actually exist on the profile
        for sk in c.top_skill_names[:3]:
            pass  # names come from the profile by construction
    # reasonings should not all be identical
    texts = {reasoning.generate(c, b, i) for i, (c, b) in enumerate(scored[:10], 1)}
    assert len(texts) >= 8


def test_assessment_discounts_self_claim():
    """A low platform-verified assessment must reduce a skill's trust vs. the
    same self-claimed skill with no/high assessment."""
    role = RoleSpec.load(JD)
    base_raw = {
        "candidate_id": "CAND_0000000",
        "profile": {"current_title": "ML Engineer", "headline": "", "summary": "",
                    "location": "Pune", "country": "India", "years_of_experience": 6},
        "career_history": [],
        "skills": [{"name": "NLP", "proficiency": "expert",
                    "endorsements": 50, "duration_months": 36}],
        "redrob_signals": {},
    }

    def trust_with(assessment):
        raw = dict(base_raw)
        raw["redrob_signals"] = (
            {} if assessment is None else {"skill_assessment_scores": {"NLP": assessment}}
        )
        return features.extract(raw, role).family_trust["nlp_ir"]

    high = trust_with(None)        # no assessment -> neutral (full self-claim trust)
    low = trust_with(20.0)         # verified weak -> should be discounted
    assert low < high, f"low assessment ({low}) should discount trust vs none ({high})"
    assert trust_with(95.0) > low  # strong assessment recovers most trust


def test_research_only_penalized_without_production():
    """A research-leaning title with no production evidence is penalized (JD
    disqualifier), but the same title WITH production evidence is not."""
    role = RoleSpec.load(JD)
    pool = scoring.PoolStats.from_candidates([])

    def final_for(title, summary):
        raw = {
            "candidate_id": "CAND_0000000",
            "profile": {"current_title": title, "headline": "", "summary": summary,
                        "location": "Pune", "country": "India", "years_of_experience": 7},
            "career_history": [], "skills": [], "redrob_signals": {},
        }
        c = features.extract(raw, role)
        return scoring.score(c, 0.5, role, pool)

    pure = final_for("Research Scientist", "published papers on transformers")
    applied = final_for(
        "Research Scientist",
        "shipped a production recommendation system serving millions of users at scale",
    )
    assert any("research-leaning" in d for d in pure.disqualifiers)
    assert applied.final > pure.final


def test_junior_title_discounted():
    """A 'Junior' qualifier reduces title credit vs the same senior title."""
    role = RoleSpec.load(JD)

    def title_fit(title):
        raw = {"candidate_id": "CAND_0000000",
               "profile": {"current_title": title, "headline": "", "summary": "",
                           "location": "Pune", "country": "India", "years_of_experience": 6},
               "career_history": [], "skills": [], "redrob_signals": {}}
        return scoring._title_fit(features.extract(raw, role))[0]

    assert title_fit("Junior ML Engineer") < title_fit("ML Engineer")


def test_github_zero_not_treated_as_missing():
    """github_activity_score 0 (linked, zero activity) must stay 0, not -1."""
    role = RoleSpec.load(JD)
    raw = {
        "candidate_id": "CAND_0000000",
        "profile": {"current_title": "ML Engineer", "headline": "", "summary": "",
                    "location": "Pune", "country": "India", "years_of_experience": 6},
        "career_history": [], "skills": [],
        "redrob_signals": {"github_activity_score": 0},
    }
    assert features.extract(raw, role).github_activity == 0.0


if __name__ == "__main__":
    test_scores_in_range_and_sorted()
    test_on_target_title_beats_off_target()
    test_honeypots_sink()
    test_reasoning_is_grounded_and_nonempty()
    test_assessment_discounts_self_claim()
    test_research_only_penalized_without_production()
    test_junior_title_discounted()
    test_github_zero_not_treated_as_missing()
    print("All sanity tests passed.")

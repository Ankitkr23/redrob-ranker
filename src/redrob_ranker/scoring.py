"""Transparent multi-signal scoring.

Final score = (weighted sum of fit components)
              x disqualifier_multiplier      (the JD's "do NOT want" list)
              x availability_multiplier       (behavioral signals)
              with honeypots forced to the bottom.

Every component is in [0, 1] and is kept in a breakdown object so the reasoning
generator (and a human reviewer at Stage 4/5) can see exactly why a candidate
landed where they did. This is the design choice that makes the system both
defensible and naturally resistant to the keyword-stuffer trap: title and
career-evidence carry as much weight as the raw skill list, so an "HR Manager
with 9 AI skills" cannot float to the top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import jd_spec
from .features import Candidate
from .jd_spec import RoleSpec

# Base component weights (sum to 1.0). Tuned toward the JD's decisive signals:
# "has shipped a ranking/search/recommendation system" (career_evidence) and a
# right-sized experience band matter more than the title string, which the JD
# warns is gameable.
WEIGHTS = {
    "title": 0.20,
    "skills": 0.22,
    "career_evidence": 0.22,
    "semantic": 0.14,
    "experience": 0.12,
    "location": 0.10,
}


@dataclass
class PoolStats:
    """Pool-relative reference points for market-interest signals.

    Absolute values of saved/search counts can vary by pool, so we score them
    relative to the pool's own distribution rather than hardcoded thresholds.
    """
    saved_p50: float = 0.0
    saved_p90: float = 1.0
    search_p50: float = 0.0
    search_p90: float = 1.0

    @classmethod
    def from_candidates(cls, cands: list[Candidate]) -> "PoolStats":
        if not cands:
            return cls()
        saved = np.array([c.saved_by_recruiters_30d for c in cands], dtype=float)
        search = np.array([c.search_appearance_30d for c in cands], dtype=float)
        return cls(
            saved_p50=float(np.percentile(saved, 50)),
            saved_p90=float(np.percentile(saved, 90)),
            search_p50=float(np.percentile(search, 50)),
            search_p90=float(np.percentile(search, 90)),
        )


@dataclass
class Breakdown:
    title: float = 0.0
    skills: float = 0.0
    career_evidence: float = 0.0
    semantic: float = 0.0
    experience: float = 0.0
    location: float = 0.0
    base: float = 0.0
    disqualifier_mult: float = 1.0
    availability_mult: float = 1.0
    # sub-factors of availability, surfaced separately for explainability
    market_mult: float = 1.0       # recruiter saves + search appearances (pool-relative)
    engagement_mult: float = 1.0   # profile completeness, interview reliability, github
    final: float = 0.0
    matched_target_title: str = ""
    disqualifiers: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


# Core AI/ML identities are credited unconditionally. "Generic" engineering
# titles (could be relevant, could not) are credited only in proportion to the
# candidate's actual AI signal, so a plain "Software Engineer" with no
# retrieval/ranking evidence doesn't get the same title credit as one who has it.
_GENERIC_TITLE_PATTERNS = frozenset({
    "data scientist", "research scientist", "staff engineer",
    "software engineer", "backend engineer", "data engineer", "platform engineer",
})


def _ai_signal(c: Candidate) -> float:
    """0-1 corroboration that this person actually does AI/ML/IR work."""
    must_trust = max((c.family_trust.get(f.name, 0.0) for f in jd_spec.MUST_FAMILIES),
                     default=0.0)
    evidence = min(1.0, 0.5 * c.evidence_strong_hits + 0.2 * c.evidence_medium_hits)
    return max(must_trust, evidence)


def _title_fit(c: Candidate) -> tuple[float, str]:
    cur = c.current_title_norm
    cur_target, matched = 0.0, ""
    for pat, w in jd_spec.TARGET_TITLE_WEIGHT.items():
        if pat in cur and w > cur_target:
            cur_target, matched = w, pat
    cur_anti = any(p in cur for p in jd_spec.ANTI_TITLE_PATTERNS) and cur_target == 0.0

    # "Junior/Intern/Trainee" contradicts a senior founding-team hire.
    junior = any(p in cur for p in jd_spec.JUNIOR_TITLE_PATTERNS)

    if cur_target > 0:
        credit = cur_target
        if matched in _GENERIC_TITLE_PATTERNS:
            # gate generic-title credit on real AI signal (floor at 40% so an
            # otherwise-relevant generic engineer isn't erased).
            credit *= 0.4 + 0.6 * _ai_signal(c)
        if junior:
            credit *= 0.45
        return credit, matched
    if cur_anti:
        # currently in an off-target role; small credit only if family signal exists
        return 0.05, ""
    return 0.35, ""  # neutral / unknown title


def _skills_fit(c: Candidate) -> float:
    must_num = must_den = 0.0
    for fam in jd_spec.MUST_FAMILIES:
        t = c.family_trust.get(fam.name, 0.0)
        must_num += t * fam.weight
        must_den += fam.weight
    must = must_num / must_den if must_den else 0.0

    nice_num = nice_den = 0.0
    for fam in jd_spec.NICE_FAMILIES:
        t = c.family_trust.get(fam.name, 0.0)
        nice_num += t * fam.weight
        nice_den += fam.weight
    nice = nice_num / nice_den if nice_den else 0.0

    return min(1.0, 0.85 * must + 0.15 * nice)


def _evidence_fit(c: Candidate) -> float:
    return min(1.0, 0.45 * c.evidence_strong_hits + 0.15 * c.evidence_medium_hits)


def _experience_fit(y: float) -> float:
    # Ideal 6-8; acceptable band 5-9. Below ~5 drops steeply (the JD's
    # disqualifiers and "5-9 years" band make early-career a poor fit), so
    # sub-band candidates don't outrank in-band ideal ones.
    if 6.0 <= y <= 8.0:
        return 1.0
    if 5.0 <= y < 6.0:
        return 0.82 + 0.18 * (y - 5.0)
    if 8.0 < y <= 9.0:
        return 1.0 - 0.12 * (y - 8.0)
    if 9.0 < y <= 12.0:
        return max(0.55, 0.88 - 0.11 * (y - 9.0))
    if 4.0 <= y < 5.0:
        return 0.45 + 0.30 * (y - 4.0)     # 4y -> 0.45, ~5y -> 0.75
    if 3.0 <= y < 4.0:
        return 0.25 + 0.20 * (y - 3.0)     # 3y -> 0.25, 4y -> 0.45
    if y > 12.0:
        return max(0.30, 0.55 - 0.03 * (y - 12.0))
    return max(0.05, 0.08 * y)             # y < 3


def _location_fit(c: Candidate) -> float:
    # Already where the role wants you: no relocation needed.
    if c.preferred_city:
        return 1.0
    if c.tier1_city:
        return 0.90

    base = 0.70 if c.in_india else 0.35  # outside India: case-by-case, no visa sponsorship

    # The JD explicitly welcomes relocation candidates. willing_to_relocate=True
    # partially closes the gap toward an in-place candidate; False leaves the
    # penalty unchanged (we never double-penalize an unwilling candidate).
    if c.willing_to_relocate:
        if c.in_india:
            base = base + 0.5 * (0.90 - base)   # -> 0.80
        else:
            base = base + 0.4 * (0.70 - base)   # -> ~0.49 (still no visa sponsorship)
    return base


def _disqualifiers(c: Candidate, b: Breakdown) -> float:
    mult = 1.0
    # entire/mostly services or consulting career (JD wants product-company ML)
    if c.consulting_share >= 0.99 or c.services_share >= 0.99:
        mult *= 0.55
        b.disqualifiers.append("career entirely at consulting/services firms")
    elif c.services_share >= 0.6:
        mult *= 0.80
        b.disqualifiers.append("mostly consulting/services background")

    # pure-research-leaning title with no production deployment evidence
    is_research = any(p in c.current_title_norm for p in jd_spec.RESEARCH_TITLE_PATTERNS)
    production = max(c.family_trust.get("ml_production", 0.0),
                    min(1.0, 0.5 * c.evidence_strong_hits))
    if is_research and production < 0.3:
        mult *= 0.60
        b.disqualifiers.append("research-leaning with no clear production deployment")
    # cv/speech/robotics primary without NLP/IR
    if c.cv_speech_share >= 0.6 and not c.has_nlp_ir_anchor:
        mult *= 0.50
        b.disqualifiers.append("CV/speech/robotics focus without NLP/IR exposure")
    elif c.cv_speech_share >= 0.6:
        mult *= 0.85
        b.disqualifiers.append("heavy CV/speech focus")
    # framework-hype only (LangChain-style) with no deeper retrieval/ranking signal
    deep = max(
        c.family_trust.get("embeddings_retrieval", 0),
        c.family_trust.get("vector_search", 0),
        c.family_trust.get("ranking_reco", 0),
    )
    if c.framework_hype_hits >= 1 and deep < 0.2 and c.evidence_strong_hits == 0:
        mult *= 0.70
        b.disqualifiers.append("framework-hype signal without deeper retrieval/ranking depth")
    # title-chasing (frequent short stints)
    if c.job_hops >= 4:
        mult *= 0.70
        b.disqualifiers.append("frequent short stints (possible title-chasing)")
    elif c.job_hops >= 3:
        mult *= 0.82
        b.disqualifiers.append("several short stints")
    return max(mult, 0.30)


def _market_factor(c: Candidate, pool: PoolStats, b: Breakdown) -> float:
    """Mild boost for candidates the market is actively interested in, scored
    relative to the pool's own distribution (median..p90)."""
    f = 1.0
    if c.saved_by_recruiters_30d > pool.saved_p50:
        frac = min(1.0, (c.saved_by_recruiters_30d - pool.saved_p50)
                   / max(1.0, pool.saved_p90 - pool.saved_p50))
        f += 0.04 * frac
        if frac >= 0.5:
            b.strengths.append(
                f"saved by {c.saved_by_recruiters_30d} recruiters in 30d (top of pool)")
    if c.search_appearance_30d > pool.search_p50:
        frac = min(1.0, (c.search_appearance_30d - pool.search_p50)
                   / max(1.0, pool.search_p90 - pool.search_p50))
        f += 0.03 * frac
    return f  # up to ~1.07


def _engagement_factor(c: Candidate, b: Breakdown) -> float:
    """Platform credibility / seriousness: profile completeness, interview
    reliability, and GitHub activity (the JD values open-source builders)."""
    f = 1.0
    f *= 0.97 + 0.03 * max(0.0, min(1.0, c.profile_completeness / 100.0))
    if c.interview_completion_rate > 0:
        f *= 0.96 + 0.04 * min(1.0, c.interview_completion_rate)
        if c.interview_completion_rate < 0.5:
            b.concerns.append(
                f"low interview completion rate ({c.interview_completion_rate:.0%})")
    # github_activity_score: -1 means no GitHub linked (neutral, not penalized).
    if c.github_activity >= 0:
        f *= 1.0 + 0.04 * min(1.0, c.github_activity / 100.0)
        if c.github_activity >= 60:
            b.strengths.append(f"active GitHub (activity score {c.github_activity:.0f}/100)")
    return f


def _availability(c: Candidate, pool: PoolStats, b: Breakdown) -> float:
    resp = 0.55 + 0.45 * max(0.0, min(1.0, c.response_rate))

    d = c.days_since_active
    recency = (1.0 if d <= 30 else 0.95 if d <= 90 else 0.85 if d <= 180
               else 0.70 if d <= 365 else 0.55)

    opentowork = 1.05 if c.open_to_work else 0.97

    n = c.notice_period_days
    notice = 1.0 if n <= 30 else 0.97 if n <= 60 else 0.93 if n <= 90 else 0.88

    if c.response_rate < 0.15:
        b.concerns.append(f"low recruiter response rate ({c.response_rate:.0%})")
    if d > 120:
        b.concerns.append(f"inactive for ~{d} days")
    if n > 60:
        b.concerns.append(f"long notice period ({n} days)")

    reach = resp * recency * opentowork * notice
    b.market_mult = _market_factor(c, pool, b)
    b.engagement_mult = _engagement_factor(c, b)
    return max(0.40, min(1.05, reach * b.market_mult * b.engagement_mult))


def score(c: Candidate, semantic: float, role: RoleSpec, pool: PoolStats) -> Breakdown:
    b = Breakdown()
    b.title, b.matched_target_title = _title_fit(c)
    b.skills = _skills_fit(c)
    b.career_evidence = _evidence_fit(c)
    b.semantic = float(semantic)
    b.experience = _experience_fit(c.years_experience)
    b.location = _location_fit(c)

    b.base = (
        WEIGHTS["title"] * b.title
        + WEIGHTS["skills"] * b.skills
        + WEIGHTS["career_evidence"] * b.career_evidence
        + WEIGHTS["semantic"] * b.semantic
        + WEIGHTS["experience"] * b.experience
        + WEIGHTS["location"] * b.location
    )

    b.disqualifier_mult = _disqualifiers(c, b)
    b.availability_mult = _availability(c, pool, b)

    final = b.base * b.disqualifier_mult * b.availability_mult
    if c.is_honeypot:
        final *= 0.02  # force implausible profiles to the bottom
        b.disqualifiers.append("profile failed plausibility checks (likely honeypot)")
    b.final = max(0.0, min(1.0, final))

    # collect human-readable strengths for the reasoning generator
    if b.matched_target_title:
        b.strengths.append(f"on-target role ({c.current_title})")
    if b.career_evidence >= 0.45:
        b.strengths.append("career history shows hands-on retrieval/ranking/ML work")
    if b.skills >= 0.5:
        b.strengths.append("solid trust-weighted coverage of core skills")
    # platform-verified assessment scores (independent of self-reported skills)
    if c.verified_skill_score >= 70:
        b.strengths.append(
            f"platform-verified skill assessments avg {c.verified_skill_score:.0f}/100")
    elif 0 <= c.verified_skill_score < 45:
        b.concerns.append(
            f"low platform-verified skill scores (avg {c.verified_skill_score:.0f}/100)")
    if 5.0 <= c.years_experience <= 9.0:
        b.strengths.append(f"{c.years_experience:.1f} yrs experience fits the band")
    if c.preferred_city or c.tier1_city:
        b.strengths.append(f"based in {c.location}")
    elif c.willing_to_relocate and not c.in_india:
        b.strengths.append("willing to relocate")
    return b

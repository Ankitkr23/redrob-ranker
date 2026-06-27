"""Turn a raw candidate JSON into a compact, ranking-ready feature record.

We do a single pass per candidate and extract:
  - identity + headline facts (title, years, location)
  - trust-weighted skill-family coverage (catches keyword stuffing)
  - free-text career evidence (catches "hidden gems")
  - disqualifier signals (consulting-only, cv/speech-only, framework-hype-only)
  - behavioral availability signals
  - a normalized text document used by the semantic component

Keeping this compact matters: we hold ~100K of these in RAM at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from . import jd_spec
from .jd_spec import RoleSpec, normalize

PROFICIENCY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# A reference "today" for recency math. The dataset is a static snapshot from
# mid-2026; using a fixed anchor keeps ranking deterministic and reproducible.
REFERENCE_DATE = date(2026, 6, 1)


@dataclass
class Candidate:
    candidate_id: str
    name: str
    current_title: str
    current_title_norm: str
    headline: str
    years_experience: float
    location: str
    country: str
    in_india: bool
    tier1_city: bool
    preferred_city: bool

    # skill-family coverage: family_name -> trust score in [0, 1]
    family_trust: dict[str, float]
    # named skills the candidate actually has, with light metadata (for reasoning)
    top_skill_names: list[str]
    n_skills: int
    # avg of Redrob's platform-verified assessment scores (0-100) over the
    # candidate's must-family skills that were assessed; -1 if none assessed.
    verified_skill_score: float

    # career evidence
    evidence_strong_hits: int
    evidence_medium_hits: int
    n_roles: int
    job_hops: int  # short stints (<= 18 months, non-current)

    # disqualifier signals
    consulting_share: float       # fraction of roles at named consulting firms
    services_share: float         # fraction of roles at consulting firms OR services industries
    cv_speech_share: float        # share of cv/speech/robotics signal
    has_nlp_ir_anchor: bool
    framework_hype_hits: int

    # behavioral availability
    response_rate: float
    open_to_work: bool
    days_since_active: int
    notice_period_days: int
    willing_to_relocate: bool
    # market-interest signals (how interested the market currently is)
    saved_by_recruiters_30d: int
    search_appearance_30d: int
    # platform credibility / seriousness signals
    github_activity: float          # 0-100, or -1 if no GitHub linked
    profile_completeness: float
    interview_completion_rate: float

    # semantic
    text_doc: str = ""

    # filled lazily by the plausibility module
    is_honeypot: bool = False
    honeypot_reasons: list[str] = field(default_factory=list)


def _parse_date(s) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _any_in(text: str, patterns) -> bool:
    return any(p in text for p in patterns)


def _count_in(text: str, patterns) -> int:
    return sum(1 for p in patterns if p in text)


def build_text_doc(raw: dict) -> str:
    """Build the normalized text document for a candidate.

    Shared by feature extraction (semantic via TF-IDF) and the offline
    embedding precomputation script, so both embed exactly the same text.
    """
    profile = raw.get("profile", {}) or {}
    history = raw.get("career_history", []) or []
    skills = raw.get("skills", []) or []
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for role_hist in history:
        parts.append(role_hist.get("title", ""))
        parts.append(role_hist.get("company", ""))
        parts.append(role_hist.get("description", ""))
        parts.append(role_hist.get("industry", ""))
    for sk in skills:
        parts.append(sk.get("name", ""))
    return normalize(" . ".join(p for p in parts if p))


def build_embedding_text(raw: dict, max_words: int = 64) -> str:
    """Build a compact, front-loaded document for the embedding backend.

    The full text_doc embeds every (often verbose) role description, which makes
    the transformer slow and pushes the high-signal fields (titles, skills) past
    the token budget. For semantic matching we instead front-load the most
    informative, low-token fields — headline, current + past titles, industries,
    and skill names — followed by a truncated summary. Role descriptions are
    dropped here because the career-evidence component already scans them in full
    (see EVIDENCE_* in jd_spec). This cuts embedding cost ~3-4x with negligible
    quality loss, since the dense signal is only one of six scoring components.
    """
    profile = raw.get("profile", {}) or {}
    history = raw.get("career_history", []) or []
    skills = raw.get("skills", []) or []

    parts: list[str] = [
        profile.get("headline", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for role_hist in history:
        parts.append(role_hist.get("title", ""))
        parts.append(role_hist.get("industry", ""))
    for sk in skills[:15]:
        parts.append(sk.get("name", ""))
    # summary last: informative but verbose, so it absorbs any remaining budget
    summary = " ".join((profile.get("summary", "") or "").split()[:40])
    parts.append(summary)

    text = normalize(" . ".join(p for p in parts if p))
    return " ".join(text.split()[:max_words])


def extract(raw: dict, role: RoleSpec) -> Candidate:
    profile = raw.get("profile", {}) or {}
    signals = raw.get("redrob_signals", {}) or {}
    history = raw.get("career_history", []) or []
    skills = raw.get("skills", []) or []

    title = profile.get("current_title", "") or ""
    headline = profile.get("headline", "") or ""
    location = profile.get("location", "") or ""
    country = profile.get("country", "") or ""

    # ---- build the normalized text document (used by semantic + scans) ----
    text_doc = build_text_doc(raw)

    # ---- platform-verified skill assessments ----
    # Redrob's own assessment of a skill (0-100), keyed by skill name. This is
    # independent of the candidate's self-reported proficiency, so it's the
    # strongest anti-keyword-stuffing signal in the dataset: a low assessment
    # discounts an inflated self-claim. Missing assessment -> neutral (no penalty).
    assessments = signals.get("skill_assessment_scores", {}) or {}
    assess_by_name = {normalize(k): float(v) for k, v in assessments.items()}
    verified_scores: list[float] = []

    # ---- skill-family trust ----
    # Trust blends *named-skill depth* (proficiency x endorsement x duration),
    # *platform-verified assessment*, and *evidence in text*. A skill claimed at
    # "expert" with 0 months used (or a low verified score) yields near-zero trust.
    family_trust: dict[str, float] = {}
    must_family_names = {f.name for f in jd_spec.MUST_FAMILIES}
    for fam in role.skill_families:
        # signal from named skills
        named = 0.0
        for sk in skills:
            sname = normalize(sk.get("name", ""))
            if _any_in(sname, fam.patterns):
                prof = PROFICIENCY_RANK.get(sk.get("proficiency", "beginner"), 1)
                dur = sk.get("duration_months", 0) or 0
                end = sk.get("endorsements", 0) or 0
                depth = (prof / 4.0)
                used = min(dur / 24.0, 1.0)        # 2+ years of use -> full
                social = min(end / 25.0, 1.0)      # endorsements as weak social proof
                # If claimed but never actually used, trust collapses.
                trust = depth * (0.25 + 0.55 * used + 0.20 * social)
                # Platform-verified assessment moderates the self-claim.
                assess = assess_by_name.get(sname)
                if assess is not None:
                    # factor in [0.4, 1.0]: a 40/100 assessment cuts trust by ~36%.
                    trust *= 0.4 + 0.6 * min(1.0, max(0.0, assess / 100.0))
                    if fam.name in must_family_names:
                        verified_scores.append(assess)
                named = max(named, trust)
        # signal from free text (career descriptions etc.)
        text_hit = 1.0 if _any_in(text_doc, fam.patterns) else 0.0
        # combine: text presence is corroborating, capped
        score = max(named, 0.55 * text_hit)
        if named > 0 and text_hit > 0:
            score = min(1.0, named + 0.20)
        family_trust[fam.name] = round(score, 4)

    verified_skill_score = (
        round(sum(verified_scores) / len(verified_scores), 1) if verified_scores else -1.0
    )

    # ---- named skills for reasoning (only those relevant to must-haves) ----
    must_patterns = tuple(p for f in jd_spec.MUST_FAMILIES for p in f.patterns)
    top_skill_names: list[str] = []
    for sk in skills:
        nm = sk.get("name", "")
        if nm and _any_in(normalize(nm), must_patterns):
            top_skill_names.append(nm)
    top_skill_names = top_skill_names[:8]

    # ---- career evidence + job-hopping ----
    evidence_strong = _count_in(text_doc, jd_spec.EVIDENCE_STRONG)
    evidence_medium = _count_in(text_doc, jd_spec.EVIDENCE_MEDIUM)

    job_hops = 0
    consulting_roles = 0
    services_roles = 0
    for role_hist in history:
        dur = role_hist.get("duration_months", 0) or 0
        if not role_hist.get("is_current", False) and 0 < dur <= 18:
            job_hops += 1
        comp = normalize(role_hist.get("company", ""))
        ind = normalize(role_hist.get("industry", ""))
        is_consulting = _any_in(comp, jd_spec.CONSULTING_FIRMS)
        if is_consulting:
            consulting_roles += 1
        if is_consulting or _any_in(ind, jd_spec.SERVICES_INDUSTRY_TOKENS):
            services_roles += 1
    n_roles = max(len(history), 1)
    consulting_share = consulting_roles / n_roles
    services_share = services_roles / n_roles

    # ---- cv/speech/robotics vs nlp/ir anchor ----
    cv_hits = _count_in(text_doc, jd_spec.CV_SPEECH_ROBOTICS)
    nlp_hits = _count_in(text_doc, jd_spec.NLP_IR_ANCHORS)
    cv_speech_share = cv_hits / (cv_hits + nlp_hits) if (cv_hits + nlp_hits) else 0.0
    has_nlp_ir_anchor = nlp_hits > 0
    framework_hype_hits = _count_in(text_doc, jd_spec.FRAMEWORK_HYPE)

    # ---- location ----
    loc_norm = normalize(location)
    country_norm = normalize(country)
    in_india = ("india" in country_norm) or _any_in(loc_norm, jd_spec.TIER1_INDIAN_CITIES)
    tier1_city = _any_in(loc_norm, jd_spec.TIER1_INDIAN_CITIES)
    preferred_city = _any_in(loc_norm, jd_spec.PREFERRED_CITIES)

    # ---- behavioral availability ----
    last_active = _parse_date(signals.get("last_active_date"))
    days_since_active = (REFERENCE_DATE - last_active).days if last_active else 365

    return Candidate(
        candidate_id=raw.get("candidate_id", ""),
        name=profile.get("anonymized_name", ""),
        current_title=title,
        current_title_norm=normalize(title),
        headline=headline,
        years_experience=float(profile.get("years_of_experience", 0) or 0),
        location=location,
        country=country,
        in_india=in_india,
        tier1_city=tier1_city,
        preferred_city=preferred_city,
        family_trust=family_trust,
        top_skill_names=top_skill_names,
        n_skills=len(skills),
        verified_skill_score=verified_skill_score,
        evidence_strong_hits=evidence_strong,
        evidence_medium_hits=evidence_medium,
        n_roles=len(history),
        job_hops=job_hops,
        consulting_share=consulting_share,
        services_share=services_share,
        cv_speech_share=cv_speech_share,
        has_nlp_ir_anchor=has_nlp_ir_anchor,
        framework_hype_hits=framework_hype_hits,
        response_rate=float(signals.get("recruiter_response_rate", 0) or 0),
        open_to_work=bool(signals.get("open_to_work_flag", False)),
        days_since_active=days_since_active,
        notice_period_days=int(signals.get("notice_period_days", 90) or 90),
        willing_to_relocate=bool(signals.get("willing_to_relocate", False)),
        saved_by_recruiters_30d=int(signals.get("saved_by_recruiters_30d", 0) or 0),
        search_appearance_30d=int(signals.get("search_appearance_30d", 0) or 0),
        # -1 means "no GitHub linked"; 0 is a valid (linked, zero-activity) score,
        # so we must NOT use `or -1` here (0 or -1 == -1 would corrupt real zeros).
        github_activity=float(signals.get("github_activity_score", -1)),
        profile_completeness=float(signals.get("profile_completeness_score", 0) or 0),
        interview_completion_rate=float(signals.get("interview_completion_rate", 0) or 0),
        text_doc=text_doc,
    )

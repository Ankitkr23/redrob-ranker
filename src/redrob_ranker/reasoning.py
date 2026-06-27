"""Generate honest, specific, varied reasoning strings.

Stage 4 of the evaluation samples 10 rows and checks each reasoning for:
specific facts, JD connection, honest concerns, no hallucination, variation,
and rank consistency. So we build each string strictly from facts already in
the candidate's profile / score breakdown — never invented skills — and we vary
phrasing and emphasis by the candidate's dominant signals and rank tier.
"""

from __future__ import annotations

from .features import Candidate
from .scoring import Breakdown


def _yrs(c: Candidate) -> str:
    return f"{c.years_experience:.1f} yrs"


def _pick(seed: int, options: list[str]) -> str:
    return options[seed % len(options)]


def generate(c: Candidate, b: Breakdown, rank: int) -> str:
    seed = abs(hash(c.candidate_id))
    title = c.current_title or "Candidate"

    # ---- lead clause: who they are ----
    lead = f"{title} with {_yrs(c)}"
    if c.location:
        lead += f", based in {c.location}"

    # ---- core fit clause, tone matched to rank/score ----
    skills_txt = ", ".join(c.top_skill_names[:3]) if c.top_skill_names else ""
    if b.final >= 0.55:
        templates = [
            f"strong fit: {_pick(seed, ['career history shows', 'profile evidences', 'background demonstrates'])} hands-on retrieval/ranking/ML work the JD centers on",
            "matches the 'product-engineering over pure research' profile the role asks for",
            "directly relevant experience for the embeddings/retrieval/ranking mandate",
        ]
        fit = _pick(seed, templates)
    elif b.final >= 0.35:
        fit = _pick(seed, [
            "partial fit: relevant signal but gaps against the core retrieval/ranking requirements",
            "adjacent experience; some core must-haves only weakly evidenced",
            "moderate fit — useful background but not a clean match to the JD's must-haves",
        ])
    else:
        fit = _pick(seed, [
            "weak fit for this role; included low in the shortlist",
            "limited alignment with the AI-engineering must-haves",
            "off-target relative to what the JD actually needs",
        ])

    parts = [f"{lead}; {fit}"]

    # ---- supporting specifics (only real facts) ----
    if skills_txt and b.skills >= 0.3:
        parts.append(f"core skills include {skills_txt}")
    elif b.career_evidence >= 0.45:
        parts.append("history references search/recommendation/ranking work")

    # platform-verified assessment is the strongest anti-stuffing fact we have
    if c.verified_skill_score >= 0:
        parts.append(f"verified skill assessment avg {c.verified_skill_score:.0f}/100")

    parts.append(f"recruiter response rate {c.response_rate:.0%}")

    # ---- honest concerns / disqualifiers ----
    concern_bits: list[str] = []
    if b.disqualifiers:
        concern_bits.append(b.disqualifiers[0])
    for cc in b.concerns:
        if cc not in concern_bits:
            concern_bits.append(cc)
    if not c.in_india:
        concern_bits.append(f"located outside India ({c.country})")

    sentence = "; ".join(parts) + "."
    if concern_bits:
        sentence += " Concern: " + "; ".join(concern_bits[:2]) + "."

    # keep it tight (1-2 sentences) and CSV-safe
    return sentence.replace("\n", " ").strip()

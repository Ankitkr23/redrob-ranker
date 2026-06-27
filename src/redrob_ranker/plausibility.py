"""Honeypot / internal-consistency detection.

The dataset seeds ~80 honeypots: profiles that look great to a keyword matcher
but are internally impossible (e.g. "8 years at a company founded 3 years ago",
or "expert in 10 skills with 0 months used"). The ground truth forces these to
relevance tier 0, and ranking >10% of them in your top 100 is disqualifying.

We don't special-case individual IDs; we apply general plausibility rules that a
careful human reviewer would notice. We flag a candidate only when the evidence
is confident, to avoid demoting genuine strong candidates (false positives are
costly because we'd drop a real fit).
"""

from __future__ import annotations

from datetime import date

from .features import PROFICIENCY_RANK, REFERENCE_DATE, _parse_date

# A skill is "claimed-deep" if proficiency is advanced/expert.
_DEEP = {"advanced", "expert"}


def detect(raw: dict) -> tuple[bool, list[str]]:
    """Return (is_honeypot, reasons)."""
    hard: list[str] = []
    soft: list[str] = []

    profile = raw.get("profile", {}) or {}
    history = raw.get("career_history", []) or []
    skills = raw.get("skills", []) or []

    yoe = float(profile.get("years_of_experience", 0) or 0)
    anchor = REFERENCE_DATE.replace(day=1)

    # ---- career-history date consistency ----
    earliest_start: date | None = None
    for r in history:
        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date"))
        dur = int(r.get("duration_months", 0) or 0)

        if sd and sd > anchor:
            hard.append("career role starts in the future")
        if sd and ed and ed < sd:
            hard.append("career role end_date precedes start_date")
        # tenure vs the actual elapsed window. For current roles the window runs
        # to "today". This catches "8 years at a company that only existed for 3".
        if sd:
            end_for_span = ed if ed else anchor
            span_months = (end_for_span.year - sd.year) * 12 + (end_for_span.month - sd.month)
            if dur > span_months + 9:
                hard.append(
                    f"role tenure ({dur}m) exceeds the elapsed window ({span_months}m)"
                )
        if sd and (earliest_start is None or sd < earliest_start):
            earliest_start = sd

    # claimed years of experience vs the career timeline
    if earliest_start is not None:
        career_months = (anchor.year - earliest_start.year) * 12 + (
            anchor.month - earliest_start.month
        )
        gap = yoe * 12 - career_months
        if gap > 48:        # >4y more experience than the timeline can support
            hard.append("years_of_experience far exceeds career timeline")
        elif gap > 30:
            soft.append("years_of_experience exceeds career timeline")

    # ---- skill plausibility (the "expert with 0 months" trap) ----
    deep_zero = 0
    for sk in skills:
        prof = (sk.get("proficiency") or "").lower()
        dur = int(sk.get("duration_months", 0) or 0)
        if prof in _DEEP and dur == 0:
            deep_zero += 1
    if deep_zero >= 3:
        hard.append(f"{deep_zero} advanced/expert skills claimed with 0 months of use")
    elif deep_zero == 2:
        soft.append("2 advanced/expert skills with 0 months of use")

    is_honeypot = bool(hard) or len(soft) >= 2
    return is_honeypot, (hard + soft)

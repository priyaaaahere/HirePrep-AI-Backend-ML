"""
skill_gap_engine.py — Structured skill-gap computation.

Given a matched job (from job_matcher) and the user's skill list,
this module compares required vs. user skills using normalized set
operations and returns a structured result ready for API responses.
"""

from utils import normalize


# ---------- INTERNAL HELPERS ----------
def _normalize_skill_set(skills: list[str]) -> set[str]:
    """Return a set of normalized skill strings."""
    return {normalize(s) for s in skills if s}


# ---------- PUBLIC API ----------
def compute_skill_gap(matched_job: dict, user_skills: list[str]) -> dict:
    """
    Compare *user_skills* against the skill set of *matched_job*.

    Parameters
    ----------
    matched_job : dict
        Must contain at least: job_title, category, job_skill_set.
    user_skills : list[str]
        Raw skill strings provided by the user.

    Returns
    -------
    dict with keys:
        matched_job_title, category, similarity_score (if present),
        required_skills, user_skills, matched_skills, missing_skills
    """
    required_raw = matched_job.get("job_skill_set", [])

    required_norm = _normalize_skill_set(required_raw)
    user_norm     = _normalize_skill_set(user_skills)

    matched_norm  = required_norm & user_norm       # intersection
    missing_norm  = required_norm - user_norm        # gap

    # Map normalized values back to their original-case versions
    norm_to_raw_required = {normalize(s): s for s in required_raw}
    norm_to_raw_user     = {normalize(s): s for s in user_skills}

    matched_original = sorted(norm_to_raw_required[n] for n in matched_norm)
    missing_original = sorted(norm_to_raw_required[n] for n in missing_norm)

    return {
        "matched_job_title":  matched_job["job_title"],
        "category":           matched_job["category"],
        "similarity_score":   matched_job.get("similarity_score"),
        "required_skills":    required_raw,
        "user_skills":        user_skills,
        "matched_skills":     matched_original,
        "missing_skills":     missing_original,
    }


import pandas as pd
from typing import Dict, List, Set

DATASET_PATH = "data/job_dataset.csv"


def normalize_skill(skill: str) -> str:
    """
    Normalize skill names for comparison
    """
    return skill.strip().lower()


def load_role_skills(role: str, level: str) -> Set[str]:
    """
    Load required skills for a given role & experience level
    """
    df = pd.read_csv(DATASET_PATH)

    # Normalize for matching (use actual column names from CSV)
    df["Title"] = df["Title"].str.lower()
    df["ExperienceLevel"] = df["ExperienceLevel"].str.lower()

    role = role.lower()
    level = level.lower()

    filtered = df[
        (df["Title"].str.contains(role, na=False)) &
        (df["ExperienceLevel"].str.contains(level, na=False))
    ]

    if filtered.empty:
        return set()

    skills_text = filtered.iloc[0]["Skills"]

    # Skills in CSV are separated by semicolons (;)
    skills = {
        normalize_skill(s)
        for s in skills_text.split(";")
        if s.strip()
    }

    return skills


def compute_skill_match(
    resume_skills: List[str],
    desired_role: str,
    experience_level: str
) -> Dict:
    """
    Compute skill match percentage and gaps
    """

    resume_skill_set = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    required_skills = load_role_skills(desired_role, experience_level)

    if not required_skills:
        return {
            "skill_match_percent": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "note": "No role data found"
        }

    matched = resume_skill_set.intersection(required_skills)
    missing = required_skills.difference(resume_skill_set)

    match_percent = (len(matched) / len(required_skills)) * 100

    return {
        "skill_match_percent": round(match_percent, 2),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }

import pandas as pd
from typing import Dict, List, Set, Optional

# Use aggregated dataset which has ALL skills combined per role+level
DATASET_PATH = "data/job_dataset_aggregated.csv"

# Flag to track if semantic matching is available
_semantic_matching_available = None


def is_semantic_matching_available() -> bool:
    """
    Check if sentence-transformers is installed for semantic role matching.
    """
    global _semantic_matching_available
    if _semantic_matching_available is None:
        try:
            from app.ml.role_matcher import smart_role_match
            _semantic_matching_available = True
        except ImportError:
            _semantic_matching_available = False
    return _semantic_matching_available


def normalize_skill(skill: str) -> str:
    """
    Normalize skill names for comparison
    """
    return skill.strip().lower()


def find_matching_role(role: str, level: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Find matching role in dataset using multiple strategies:
    1. Exact match
    2. Partial/substring match  
    3. Semantic similarity (NLP-based) - if available
    
    Returns filtered dataframe or None if no match found.
    """
    role_lower = role.lower()
    level_lower = level.lower()
    
    # Strategy 1: Exact match on title
    exact_match = df[
        (df["Title"].str.lower() == role_lower) &
        (df["ExperienceLevel"].str.lower().str.contains(level_lower, na=False))
    ]
    if not exact_match.empty:
        return exact_match
    
    # Strategy 2: Partial/substring match (original behavior)
    partial_match = df[
        (df["Title"].str.lower().str.contains(role_lower, na=False)) &
        (df["ExperienceLevel"].str.lower().str.contains(level_lower, na=False))
    ]
    if not partial_match.empty:
        return partial_match
    
    # Strategy 3: Semantic matching using NLP
    if is_semantic_matching_available():
        try:
            from app.ml.role_matcher import smart_role_match
            
            # Get semantically similar role
            match_result = smart_role_match(role, level)
            matched_role = match_result.get("matched_role", "")
            similarity = match_result.get("similarity", 0)
            
            # Only use if similarity is above threshold (0.3)
            if similarity >= 0.3 and matched_role:
                semantic_match = df[
                    (df["Title"].str.lower() == matched_role.lower()) &
                    (df["ExperienceLevel"].str.lower().str.contains(level_lower, na=False))
                ]
                if not semantic_match.empty:
                    return semantic_match
                
                # Try without level filter if still no match
                semantic_match_any_level = df[
                    df["Title"].str.lower() == matched_role.lower()
                ]
                if not semantic_match_any_level.empty:
                    return semantic_match_any_level
                    
        except Exception:
            pass  # Fallback to no match if semantic matching fails
    
    return None


def load_role_skills(role: str, level: str) -> tuple[Set[str], Optional[Dict]]:
    """
    Load required skills for a given role & experience level.
    Uses aggregated dataset where skills are already combined per role+level.
    
    Now includes semantic role matching as fallback when exact role not found.
    
    Returns:
        Tuple of (skills_set, role_match_info)
        - skills_set: Set of required skills
        - role_match_info: Dict with matching details (if semantic match was used)
    """
    df = pd.read_csv(DATASET_PATH)
    
    role_match_info = None
    
    # Try to find matching role
    filtered = find_matching_role(role, level, df)
    
    # If semantic matching was used, capture the info
    if filtered is not None and is_semantic_matching_available():
        matched_title = filtered.iloc[0]["Title"]
        if matched_title.lower() != role.lower():
            from app.ml.role_matcher import smart_role_match
            role_match_info = smart_role_match(role, level)

    if filtered is None or filtered.empty:
        return set(), role_match_info

    # Aggregated dataset uses "All_Skills" column with combined skills
    skills_text = filtered.iloc[0]["All_Skills"]

    # Skills in CSV are separated by semicolons (;)
    skills = {
        normalize_skill(s)
        for s in skills_text.split(";")
        if s.strip()
    }

    return skills, role_match_info


def compute_skill_match(
    resume_skills: List[str],
    desired_role: str,
    experience_level: str
) -> Dict:
    """
    Compute skill match percentage and gaps.
    
    Now includes semantic role matching:
    - If "Fraud Analyst" is requested but not in dataset,
      system finds similar role (e.g., "Data Analyst") and uses its skills.
    - Response includes info about which role was actually matched.
    """

    resume_skill_set = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    required_skills, role_match_info = load_role_skills(desired_role, experience_level)

    if not required_skills:
        return {
            "skill_match_percent": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "role_matching": role_match_info,
            "note": "No role data found - try a different role or check spelling"
        }

    matched = resume_skill_set.intersection(required_skills)
    missing = required_skills.difference(resume_skill_set)

    match_percent = (len(matched) / len(required_skills)) * 100

    result = {
        "skill_match_percent": round(match_percent, 2),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }
    
    # Include role matching info if semantic matching was used
    if role_match_info:
        result["role_matching"] = {
            "requested_role": role_match_info.get("query_role"),
            "matched_to_role": role_match_info.get("matched_role"),
            "similarity": role_match_info.get("similarity"),
            "match_type": role_match_info.get("match_type"),
            "note": role_match_info.get("note")
        }
    
    return result

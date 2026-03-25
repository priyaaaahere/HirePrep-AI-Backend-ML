import pandas as pd
from typing import Dict, List, Set, Optional

from app.experience import map_experience_level

DATASET_PATH = "data/job_dataset_final.csv"

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


def load_job_data() -> pd.DataFrame:
    """
    Load and preprocess job dataset
    """
    df = pd.read_csv(DATASET_PATH, encoding="latin1")
    
    # Drop rows with missing Title
    df = df.dropna(subset=["Title"])
    
    return df


def parse_skills(skills_text: str) -> Set[str]:
    """
    Parse skills string (semicolon separated) into a normalized set
    """
    if pd.isna(skills_text) or not skills_text:
        return set()
    
    return {
        normalize_skill(s)
        for s in str(skills_text).split(";")
        if s.strip()
    }


def calculate_skill_similarity(resume_skills: Set[str], job_skills: Set[str]) -> float:
    """
    Calculate Jaccard-like similarity between resume skills and job skills
    Returns a score between 0 and 100
    """
    if not job_skills:
        return 0.0
    
    matched = resume_skills.intersection(job_skills)
    
    # Weight by how many required skills are matched
    coverage = len(matched) / len(job_skills) if job_skills else 0
    
    return coverage * 100


def experience_level_match(user_level: str, job_level: str) -> float:
    """
    Calculate experience level compatibility score (0-100)
    """
    user_level = user_level.lower() if user_level else "fresher"
    job_level = job_level.lower() if job_level else ""
    
    # Define level hierarchy
    levels = {
        "fresher": 0, "entry-level": 0, "entry level": 0, "intern": 0, "trainee": 0,
        "junior": 1, "associate": 1,
        "mid": 2, "mid-level": 2,
        "senior": 3, "experienced": 3, "lead": 3, "principal": 4, "architect": 4
    }
    
    user_rank = levels.get(user_level, 1)
    
    # Find job rank from job level string
    job_rank = 1  # default
    for key, rank in levels.items():
        if key in job_level:
            job_rank = rank
            break
    
    # Calculate match score - penalize if user is underqualified
    diff = abs(user_rank - job_rank)
    
    if diff == 0:
        return 100
    elif diff == 1:
        return 70  # Close match
    elif diff == 2:
        return 40  # Stretch role
    else:
        return 20  # Significant mismatch


def recommend_roles(
    resume_skills: List[str],
    experience_years: int = 0,
    education_degree: str = "",
    top_n: int = 10
) -> List[Dict]:
    """
    Recommend job roles based on resume profile
    
    Args:
        resume_skills: List of skills from resume
        experience_years: Years of work experience
        education_degree: Degree type (Bachelor, Master, etc.)
        top_n: Number of top recommendations to return
    
    Returns:
        List of recommended roles with match scores
    """
    df = load_job_data()
    
    # Normalize resume skills
    resume_skill_set = {normalize_skill(s) for s in resume_skills}
    
    # Get user experience level
    user_exp_level = map_experience_level(experience_years)
    
    recommendations = []
    
    # Group by unique job title to avoid duplicates
    job_groups = df.groupby("Title")
    
    for title, group in job_groups:
        # Get the first row for this title (representative)
        row = group.iloc[0]
        
        job_skills = parse_skills(row.get("All_Skills", ""))
        job_level = str(row.get("ExperienceLevel_Category", ""))
        
        # Calculate skill similarity
        skill_score = calculate_skill_similarity(resume_skill_set, job_skills)
        
        # Calculate experience level match
        exp_score = experience_level_match(user_exp_level, job_level)
        
        # Final score: weighted combination
        # Skill match: 70%, Experience match: 30%
        final_score = (0.7 * skill_score) + (0.3 * exp_score)
        
        # Find matched and missing skills
        matched_skills = sorted(resume_skill_set.intersection(job_skills))
        missing_skills = sorted(job_skills.difference(resume_skill_set))
        
        recommendations.append({
            "role": title,
            "experience_level": job_level,
            "match_score": round(final_score, 2),
            "skill_match_percent": round(skill_score, 2),
            "matched_skills": matched_skills[:10],  # Limit for readability
            "skills_to_learn": missing_skills[:10],  # Top skills to learn
            "total_required_skills": len(job_skills)
        })
    
    # Sort by match score descending
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    return recommendations[:top_n]


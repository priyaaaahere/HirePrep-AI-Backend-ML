import pandas as pd
from typing import Dict, List, Set, Optional
from collections import defaultdict

DATASET_PATH = "data/job_dataset.csv"

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
    df = pd.read_csv(DATASET_PATH)
    
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


def map_experience_level(years: int) -> str:
    """
    Map years of experience to experience level category
    """
    if years <= 1:
        return "fresher"
    elif years <= 3:
        return "junior"
    elif years <= 5:
        return "mid"
    else:
        return "senior"


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
        
        job_skills = parse_skills(row.get("Skills", ""))
        job_level = str(row.get("ExperienceLevel", ""))
        job_keywords = str(row.get("Keywords", ""))
        
        # Calculate skill similarity
        skill_score = calculate_skill_similarity(resume_skill_set, job_skills)
        
        # Calculate experience level match
        exp_score = experience_level_match(user_exp_level, job_level)
        
        # Bonus for keyword matches
        keyword_bonus = 0
        if job_keywords:
            keywords_set = {normalize_skill(k) for k in job_keywords.split(";")}
            keyword_matches = resume_skill_set.intersection(keywords_set)
            keyword_bonus = min(len(keyword_matches) * 5, 20)  # Max 20 bonus points
        
        # Final score: weighted combination
        # Skill match: 60%, Experience match: 30%, Keyword bonus: 10%
        final_score = (0.6 * skill_score) + (0.3 * exp_score) + (0.1 * keyword_bonus)
        
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


def get_role_recommendations_from_profile(profile: dict, top_n: int = 10) -> Dict:
    """
    Get role recommendations from a complete user profile.
    
    If user specifies a desired_role that's not in dataset, uses semantic
    matching to find similar roles and prioritizes them in recommendations.
    
    Args:
        profile: User profile dictionary (from resume parser)
        top_n: Number of recommendations to return
    
    Returns:
        Dictionary with recommendations and metadata
    """
    # Extract relevant data from profile
    skills = profile.get("skills", [])
    
    education = profile.get("education", {})
    degree = education.get("degree", "")
    
    experience = profile.get("experience", {})
    experience_years = experience.get("years", 0)
    
    # Check if user has a desired role specified
    career_prefs = profile.get("career_preferences", {})
    desired_role = career_prefs.get("desired_role", "")
    
    # Get recommendations
    recommendations = recommend_roles(
        resume_skills=skills,
        experience_years=experience_years,
        education_degree=degree,
        top_n=top_n
    )
    
    # If user has a desired role, use semantic matching to find related roles
    role_match_info = None
    if desired_role and is_semantic_matching_available():
        try:
            from app.ml.role_matcher import smart_role_match
            
            role_match_result = smart_role_match(desired_role, "")
            matched_role = role_match_result.get("matched_role", "")
            similarity = role_match_result.get("similarity", 0)
            
            if similarity >= 0.3 and matched_role:
                role_match_info = {
                    "requested_role": desired_role,
                    "matched_to_role": matched_role,
                    "similarity": similarity,
                    "match_type": role_match_result.get("match_type"),
                    "alternatives": role_match_result.get("alternatives", [])
                }
                
                # Boost matching roles in recommendations
                for rec in recommendations:
                    if rec["role"].lower() == matched_role.lower():
                        rec["is_desired_role_match"] = True
                        rec["match_score"] = min(100, rec["match_score"] + 10)  # Boost score
                    # Also check alternatives
                    alt_roles = [a.get("role", "").lower() if isinstance(a, dict) else "" 
                                 for a in role_match_result.get("alternatives", [])]
                    if rec["role"].lower() in alt_roles:
                        rec["is_related_to_desired"] = True
                        rec["match_score"] = min(100, rec["match_score"] + 5)  # Smaller boost
                
                # Re-sort after boosting
                recommendations.sort(key=lambda x: x["match_score"], reverse=True)
                
        except Exception:
            pass  # Continue without semantic matching if it fails
    
    # Categorize recommendations
    high_match = [r for r in recommendations if r["match_score"] >= 60]
    moderate_match = [r for r in recommendations if 40 <= r["match_score"] < 60]
    stretch_roles = [r for r in recommendations if r["match_score"] < 40]
    
    result = {
        "total_roles_analyzed": len(load_job_data()["Title"].unique()),
        "recommendations": recommendations,
        "summary": {
            "high_match_count": len(high_match),
            "moderate_match_count": len(moderate_match),
            "stretch_roles_count": len(stretch_roles),
            "top_role": recommendations[0]["role"] if recommendations else None,
            "top_score": recommendations[0]["match_score"] if recommendations else 0
        },
        "user_profile_summary": {
            "skills_count": len(skills),
            "experience_level": map_experience_level(experience_years),
            "experience_years": experience_years
        }
    }
    
    # Include role matching info if semantic matching was used
    if role_match_info:
        result["desired_role_matching"] = role_match_info
    
    return result

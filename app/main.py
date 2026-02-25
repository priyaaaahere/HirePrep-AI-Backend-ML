from fastapi import FastAPI, UploadFile, File
import tempfile

from app.pdf_to_text import extract_text_from_pdf
from app.resume_parser import parse_resume_for_frontend, parse_resume_for_analysis
from app.experience import map_experience_level
from app.ml.placement_predictor import predict_base_probability
from app.ml.skill_matcher import compute_skill_match
from app.ml.role_recommender import get_role_recommendations_from_profile
from app.Services.gemini_role_recommender import get_gemini_role_recommendations

app = FastAPI(title="HirePrep AI Backend")


@app.get("/")
def root():
    return {"status": "HirePrep AI backend running"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload resume PDF → return EDITABLE profile JSON for frontend display
    Returns: name, email, phone, github, linkedin, cgpa, projects, internships, certifications, skills, meta
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    raw_text = extract_text_from_pdf(pdf_path)
    profile_json = parse_resume_for_frontend(raw_text)

    return profile_json


@app.post("/parse-resume-for-analysis")
async def parse_resume_analysis(file: UploadFile = File(...)):
    """
    Upload resume PDF → return data optimized for ML analysis
    Returns: cgpa, project_count, internship_count, certification_count, skills
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    raw_text = extract_text_from_pdf(pdf_path)
    analysis_data = parse_resume_for_analysis(raw_text)

    return analysis_data


@app.post("/placement-probability")
async def placement_probability(profile: dict):
    meta = profile.get("meta", {})

    if meta.get("user_verified") is not True:
        return {"error": "Profile not confirmed by user"}

    # -------------------------
    # 1. Base placement probability (ML)
    # -------------------------
    base_prob = predict_base_probability(profile)

    # -------------------------
    # 2. Skill match (role-aware)
    # -------------------------
    career = profile.get("career_preferences", {})
    desired_role = career.get("desired_role", "")
    
    # Calculate experience_level from experience years (not from user input)
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)

    skills = profile.get("skills", [])

    skill_result = compute_skill_match(
        resume_skills=skills,
        desired_role=desired_role,
        experience_level=experience_level
    )

    skill_match_percent = skill_result.get("skill_match_percent", 0)

    # -------------------------
    # 3. Final placement probability
    # -------------------------
    BASE_WEIGHT = 0.6
    SKILL_WEIGHT = 0.4
    
    final_probability = round(
        BASE_WEIGHT * base_prob + SKILL_WEIGHT * skill_match_percent,
        2
    )

    return {
        "placement_analysis": {
            "base_probability": round(base_prob, 2),
            "skill_match_percent": skill_match_percent,
            "final_probability": final_probability,
            "formula": f"({BASE_WEIGHT} × {round(base_prob, 2)}) + ({SKILL_WEIGHT} × {skill_match_percent}) = {final_probability}",
            "weights": {
                "base_ml_weight": BASE_WEIGHT,
                "skill_match_weight": SKILL_WEIGHT
            }
        },
        "skill_match": skill_result,
        "final_placement_probability": final_probability,
    }


@app.post("/recommend-roles")
async def recommend_roles(profile: dict, top_n: int = 5):
    """
    Recommend suitable job roles based on user's resume/profile.
    Analyzes skills, experience, and education to find best matches.
    """
    meta = profile.get("meta", {})

    if meta.get("user_verified") is not True:
        return {"error": "Profile not confirmed by user"}

    gemini_result = get_gemini_role_recommendations(profile, top_n=top_n)
    if gemini_result:
        return {
            "skill_analysis": gemini_result.get("skill_analysis"),
            "role_recommendations": {
                "top_roles": [
                    {
                        "rank": role.get("rank"),
                        "role": role.get("role", ""),
                        "experience_level": role.get("experience_level", ""),
                        "skill_match_percent": role.get("skill_match_percent", 0),
                        "matched_skills": role.get("matched_skills", []),
                        "skills_to_learn": role.get("skills_to_learn", []),
                    }
                    for role in gemini_result.get("top_roles", [])
                ],
            },
            "source": gemini_result.get("source", "gemini_hybrid"),
        }

    return get_role_recommendations_from_profile(profile, top_n=top_n)

@app.post("/complete-analysis")
async def complete_analysis(profile: dict):
    """
    🎯 UNIFIED ENDPOINT - Returns complete analysis in one response:
    1. Placement Probability (base ML + skill-adjusted + final)
    2. Skill Gap Analysis (matched, missing, percentage)
    3. Top 5 Role Recommendations (sorted by match %)
    """
    meta = profile.get("meta", {})

    if meta.get("user_verified") is not True:
        return {"error": "Profile not confirmed by user. Please set meta.user_verified = true"}

    # =========================================================
    # 1. BASE PLACEMENT PROBABILITY (from ML Model)
    # =========================================================
    base_probability = predict_base_probability(profile)

    # =========================================================
    # 2. SKILL MATCH ANALYSIS (role-aware)
    # =========================================================
    career = profile.get("career_preferences", {})
    desired_role = career.get("desired_role", "")
    
    # Calculate experience_level from experience years (not from user input)
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)
    
    skills = profile.get("skills", [])

    skill_analysis = compute_skill_match(
        resume_skills=skills,
        desired_role=desired_role,
        experience_level=experience_level
    )

    skill_match_percent = skill_analysis.get("skill_match_percent", 0)

    # =========================================================
    # 3. FINAL PLACEMENT PROBABILITY (weighted formula)
    # =========================================================
    # Formula: 60% base (ML) + 40% skill match
    BASE_WEIGHT = 0.6
    SKILL_WEIGHT = 0.4

    final_probability = round(
        (BASE_WEIGHT * base_probability) + (SKILL_WEIGHT * skill_match_percent),
        2
    )

    # =========================================================
    # 4. ROLE RECOMMENDATIONS (Top 5)
    # =========================================================
    # gemini_roles = get_gemini_role_recommendations(profile, top_n=5)
    # if gemini_roles:
    #     top_roles = gemini_roles.get("top_roles", [])
    #     role_source = gemini_roles.get("source", "gemini")
    # else:
    #     
    role_result = get_role_recommendations_from_profile(profile, top_n=5)
    top_roles = [
            {
                "rank": idx + 1,
                "role": role["role"],
                "experience_level": role["experience_level"],
                "match_percentage": role["match_score"],
                "skill_match_percent": role["skill_match_percent"],
                "matched_skills": role["matched_skills"],
                "skills_to_learn": role["skills_to_learn"]
            }
            for idx, role in enumerate(role_result.get("recommendations", []))
        ]
    role_source = "dataset"

    # =========================================================
    # 5. BUILD COMPLETE RESPONSE
    # =========================================================
    return {
        "status": "success",

        # ---- PLACEMENT PROBABILITY BREAKDOWN ----
        "placement_analysis": {
            "base_probability_ml": round(base_probability, 2),
            "skill_match_percent": skill_match_percent,
            "final_probability": final_probability,
            "formula_used": f"({BASE_WEIGHT} × base_probability) + ({SKILL_WEIGHT} × skill_match) = ({BASE_WEIGHT} × {round(base_probability, 2)}) + ({SKILL_WEIGHT} × {skill_match_percent}) = {final_probability}%",
            "interpretation": get_probability_interpretation(final_probability)
        },

        # ---- SKILL GAP ANALYSIS ----
        "skill_analysis": {
            "desired_role": desired_role,
            "experience_level": experience_level,
            "total_skills_in_resume": len(skills),
            "skill_match_percent": skill_match_percent,
            "matched_skills": skill_analysis.get("matched_skills", []),
            "missing_skills": skill_analysis.get("missing_skills", []),
            "matched_count": len(skill_analysis.get("matched_skills", [])),
            "missing_count": len(skill_analysis.get("missing_skills", []))
        },

        # ---- TOP 5 ROLE RECOMMENDATIONS ----
        "role_recommendations": {
            "top_roles": top_roles,
            "source": role_source
        },

    }


def get_probability_interpretation(probability: float) -> str:
    """
    Returns human-readable interpretation of placement probability
    """
    if probability >= 80:
        return "Excellent! Very high chances of placement. Your profile is strong."
    elif probability >= 60:
        return "Good! Above average chances. Focus on missing skills to improve further."
    elif probability >= 40:
        return "Moderate chances. Consider adding more projects and certifications."
    elif probability >= 20:
        return "Below average. Work on skill gaps and gain more practical experience."
    else:
        return "Needs improvement. Focus on building core skills and getting internships."
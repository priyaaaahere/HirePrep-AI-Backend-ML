from fastapi import FastAPI, UploadFile, File
import tempfile

from app.pdf_to_text import extract_text_from_pdf
from app.resume_parser import build_editable_profile
from app.ml.placement_predictor import predict_base_probability
from app.ml.skill_matcher import compute_skill_match

app = FastAPI(title="HirePrep AI Backend")


@app.get("/")
def root():
    return {"status": "HirePrep AI backend running"}


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload resume PDF → return EDITABLE draft JSON
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    raw_text = extract_text_from_pdf(pdf_path)
    profile_json = build_editable_profile(raw_text)

    return profile_json


@app.post("/analyze-profile")
async def analyze_profile(profile: dict):
    """
    Accept ONLY user-confirmed profile for ML / EDA
    """

    meta = profile.get("meta", {})

    if meta.get("user_verified") is not True:
        return {
            "error": "Profile not confirmed by user. Analysis aborted."
        }

    # Future: ML, skill gap, ATS, Gemini
    return {
        "message": "Confirmed profile received. Ready for analysis.",
        "profile_used": profile
    }


@app.get("/test-resume")
def test_resume():
    """
    Local test using data/resume.pdf
    """
    pdf_path = "data/resume.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    profile_json = build_editable_profile(raw_text)
    return profile_json

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
    experience_level = career.get("experience_level", "")

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
    final_probability = round(
        0.6 * base_prob + 0.4 * skill_match_percent,
        2
    )

    return {
        "skill_match": skill_result,
        "final_placement_probability": final_probability,
    }
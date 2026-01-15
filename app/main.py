from fastapi import FastAPI, UploadFile, File
import tempfile

from app.pdf_to_text import extract_text_from_pdf
from app.resume_parser import build_editable_profile

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

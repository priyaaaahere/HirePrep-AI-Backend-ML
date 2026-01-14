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
    1. Receives resume PDF from frontend
    2. Parses it
    3. Returns EDITABLE JSON
    """

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    # PDF → text
    raw_text = extract_text_from_pdf(pdf_path)

    # text → editable JSON
    profile_json = build_editable_profile(raw_text)

    return profile_json


@app.post("/analyze-profile")
async def analyze_profile(profile: dict):
    """
    Receives user-edited JSON
    (Later: ML, EDA, ATS, Gemini)
    """
    return {
        "message": "Profile received successfully",
        "profile_used": profile
    }


@app.get("/test-resume")
def test_resume():
    """
    Test endpoint to parse the sample resume.pdf from the data folder.
    """
    pdf_path = "data/resume.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    profile_json = build_editable_profile(raw_text)
    return profile_json
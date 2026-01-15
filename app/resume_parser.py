import re
from typing import List, Dict
import spacy

nlp = spacy.load("en_core_web_sm")


# ----------------------------
# Cleaning
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[•●▪◦►]", "\n", text)
    return text.strip()


# ----------------------------
# Section extraction
# ----------------------------
def extract_section(text: str, section_name: str) -> str:
    """
    Extracts text belonging to a given section (e.g., Skills)
    """
    pattern = rf"{section_name}\s*(.*?)(?:\n[A-Z][A-Za-z\s]+|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


# ----------------------------
# Basic extractors
# ----------------------------
def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"\+?\d[\d\s\-]{8,15}", text)
    return match.group() if match else ""


def extract_linkedin(text: str) -> str:
    match = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s]+", text)
    return match.group() if match else ""


def extract_github(text: str) -> str:
    match = re.search(r"(https?://)?(www\.)?github\.com/[^\s]+", text)
    return match.group() if match else ""


def extract_name(text: str) -> str:
    doc = nlp(text[:300])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""


# ----------------------------
# UNIVERSAL skill extraction (NO ONTOLOGY)
# ----------------------------
def extract_skills_from_section(skills_text: str) -> List[str]:
    if not skills_text:
        return []

    # Remove category labels like "Languages:", "Tools:"
    skills_text = re.sub(r"[A-Za-z\s]+:", "", skills_text)

    # Split on | , or newline
    raw_skills = re.split(r"\||,|\n", skills_text)

    # Clean & normalize
    skills = {
        skill.strip()
        for skill in raw_skills
        if skill.strip() and len(skill.strip()) > 1
    }

    return sorted(skills)


# ----------------------------
# Education & experience
# ----------------------------
def extract_degree(text: str) -> str:
    t = text.lower()
    if "b.tech" in t or "bachelor" in t:
        return "Bachelor"
    if "master" in t or "m.tech" in t:
        return "Master"
    return ""


def extract_branch(text: str) -> str:
    t = text.lower()
    if "computer science" in t:
        return "Computer Science"
    if "information technology" in t:
        return "Information Technology"
    return ""


def extract_cgpa(text: str) -> str:
    match = re.search(r"(cgpa|gpa)\s*[:\-]?\s*(\d\.\d{1,2})", text.lower())
    return match.group(2) if match else ""


def extract_graduation_year(text: str) -> str:
    years = re.findall(r"(20\d{2})", text)
    return years[-1] if years else ""


def estimate_experience_years(text: str) -> int:
    match = re.search(r"(\d+)\+?\s+years?", text.lower())
    return int(match.group(1)) if match else 0


# ----------------------------
# FINAL editable profile
# ----------------------------
def build_editable_profile(resume_text: str) -> Dict:
    clean = clean_text(resume_text)

    skills_section = extract_section(clean, "Skills")
    skills = extract_skills_from_section(skills_section)

    experience_years = estimate_experience_years(clean)

    return {
        "personal_info": {
            "full_name": extract_name(clean),
            "email": extract_email(clean),
            "phone": extract_phone(clean),
            "linkedin": extract_linkedin(clean),
            "github": extract_github(clean)
        },

        "skills": skills,

        "education": {
            "degree": extract_degree(clean),
            "branch": extract_branch(clean),
            "cgpa": extract_cgpa(clean),
            "graduation_year": extract_graduation_year(clean)
        },

        "projects": [],
        "internships": [],
        "certifications": [],
        "achievements": [],

        "experience": {
            "years": experience_years,
            "level": (
                "fresher" if experience_years == 0 else
                "junior" if experience_years < 2 else
                "mid" if experience_years <= 5 else
                "senior"
            )
        },

        "meta": {
            "status": "draft",
            "editable": True,
            "user_verified": False
        }
    }

import re
from typing import List, Dict


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[•●▪◦►]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"\+?\d[\d\s\-]{8,15}", text)
    return match.group() if match else ""


def extract_skills(text: str) -> List[str]:
    skill_bank = [
        "python", "java", "sql", "pandas", "numpy",
        "matplotlib", "seaborn", "scikit-learn",
        "tensorflow", "machine learning",
        "git", "github", "mysql"
    ]
    text_lower = text.lower()
    return sorted({skill for skill in skill_bank if skill in text_lower})


def estimate_experience_years(text: str) -> int:
    match = re.search(r"(\d+)\+?\s+years?", text.lower())
    return int(match.group(1)) if match else 0


def build_editable_profile(resume_text: str) -> Dict:
    """
    Build a FRONTEND-EDITABLE resume JSON
    """
    clean = clean_text(resume_text)
    experience_years = estimate_experience_years(clean)

    return {
        "personal_info": {
            "full_name": "",
            "email": extract_email(clean),
            "phone": extract_phone(clean),
            "linkedin": "",
            "github": ""
        },

        "skills": extract_skills(clean),

        "education": {
            "degree": "",
            "branch": "",
            "cgpa": "",
            "graduation_year": ""
        },

        "projects": [],

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
            "editable": True
        }
    }

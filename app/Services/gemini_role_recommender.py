import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from app.experience import map_experience_level
from app.ml.role_recommender import load_job_data

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemma-3-4b-it"


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def _process_role(role_data: dict, resume_skills: List[str]) -> Dict:
    role_name = role_data.get("role")
    experience_level = role_data.get("experience_level")

    if not role_name:
        return None

    # Use Gemini's intelligent matching directly
    matched = role_data.get("matched_skills", [])
    skills_to_learn = role_data.get("skills_to_learn", [])

    if not isinstance(matched, list):
        matched = []
    if not isinstance(skills_to_learn, list):
        skills_to_learn = []

    total = len(matched) + len(skills_to_learn)
    skill_match_percent = round((len(matched) / total) * 100, 2) if total > 0 else 0.0

    return {
        "role": role_name,
        "experience_level": experience_level,
        "skill_match_percent": skill_match_percent,
        "matched_skills": matched,
        "skills_to_learn": skills_to_learn,
    }


def _build_skill_analysis(desired_role: str, experience_level: str, resume_skills: List[str], desired_role_data: dict) -> Dict:
    """Build skill_analysis block for the desired role using Gemini's intelligent matching."""

    # Use Gemini's intelligent matching directly
    matched = desired_role_data.get("matched_skills", [])
    missing = desired_role_data.get("missing_skills", [])

    if not isinstance(matched, list):
        matched = []
    if not isinstance(missing, list):
        missing = []

    total = len(matched) + len(missing)
    skill_match_percent = round((len(matched) / total) * 100, 2) if total > 0 else 0.0

    return {
        "desired_role": desired_role,
        "experience_level": experience_level,
        "skill_match_percent": skill_match_percent,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def _build_prompt(profile: dict, top_n: int) -> str:
    skills = profile.get("skills", [])
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)
    desired_role = profile.get("career_preferences", {}).get("desired_role", "")

    skills_list = ", ".join(skills) if skills else "None provided"

    # Load real role data filtered to candidate's experience level
    df = load_job_data()
    level_df = df[df["ExperienceLevel_Category"] == experience_level]
    role_skill_map = ""
    for _, row in level_df.iterrows():
        role_skill_map += f"- {row['Title']}: {row['All_Skills']}\n"

        return f"""You are an expert career advisor and skill analyzer. Your task is to analyze a candidate's profile and return structured JSON data.

    You will be given:
    - The candidate's current skills (from their resume)
    - Their years of experience and experience level
    - Their desired job role
    - A reference list of real job roles with their required skills
    - How many alternative roles to recommend

    Your output has TWO parts:

    ========================================
    SKILL MATCHING RULES (VERY IMPORTANT):
    ========================================

    When deciding whether a candidate "knows" a skill, apply deep domain understanding. Do NOT do simple string matching. Use these principles:

    1. A specific tool implies its parent concept:
    e.g., PostgreSQL → SQL, React → JavaScript, Git → Version Control, Docker → Containerization

    2. A library implies the capability it provides:
    e.g., Pandas → Data Manipulation, Scikit-learn → ML Fundamentals, Matplotlib → Data Visualization

    3. A broad skill implies its fundamentals:
    e.g., Machine Learning → Supervised Learning, Cloud Computing → Cloud Concepts

    4. Variant names are identical:
    e.g., "scikit-learn" = "sklearn", "javascript" = "js", "node.js" = "nodejs", "dsa" = "data structures and algorithms"

    Apply this reasoning universally across ALL domains.
    If a candidate's listed skill reasonably demonstrates competence in a required skill, count it as matched.
    Only list a skill in skills_to_learn / missing_skills if it is genuinely NEW knowledge the candidate does not have.

    ========================================
    PART 1: Desired Role Analysis
    ========================================

    Analyze the candidate's desired role at their experience level ({experience_level}).
    - Determine which of the candidate's skills are relevant to this role (matched_skills).
    - Determine which important skills they still need to learn (missing_skills).

    ========================================
    PART 2: Role Recommendations
    ========================================

    Below is a reference list of real job roles at the candidate's experience level ({experience_level}) along with their typical required skills:

    {role_skill_map}

    From this list, select the {top_n} roles where the candidate has the strongest skill alignment.
    You MAY suggest a role not in this list only if the candidate's skill set strongly warrants it, but prefer roles from this list.

    RANKING CRITERIA (in priority order):
    1. Skill overlap — higher percentage of required skills already known by the candidate.
    2. Career fit — the role is a natural match for the candidate's background and experience.
    3. Learnability — the missing skills are reasonable extensions of what the candidate already knows.
    4. Diversity — the {top_n} roles should represent distinct career directions, not minor variations of the same role.

    For each recommended role, determine which candidate skills are relevant (matched_skills) and which new skills they need (skills_to_learn).

    ========================================
    FORMAT RULES:
    ========================================

    - For matched_skills, use the EXACT skill name as written in the candidate's skill list (preserve their casing/spelling).
    - For missing_skills / skills_to_learn, use standard industry lowercase names.
    - For each role, aim for 12 to 20 total skills (matched + missing combined) to reflect realistic job requirements.
    - Do NOT include skill_match_percent — it will be calculated externally.
    - experience_level must be one of: "Entry-Level", "Mid-Level", "Senior-Level".
    - Output ONLY raw JSON. No markdown, no code fences, no explanation.
    - The response MUST start with {{ and end with }}.

    ========================================
    OUTPUT FORMAT (STRICT JSON):
    ========================================

    {{
    "desired_role": {{
        "role": "string",
        "experience_level": "string",
        "matched_skills": ["string", ...],
        "missing_skills": ["string", ...]
    }},
    "roles": [
        {{
        "role": "string",
        "experience_level": "string",
        "matched_skills": ["string", ...],
        "skills_to_learn": ["string", ...]
        }}
    ]
    }}

    ========================================
    CANDIDATE PROFILE:
    ========================================

    Skills: {skills_list}
    Experience: {experience_years} years ({experience_level})
    Desired Role: {desired_role if desired_role else "Not specified"}
    Number of role recommendations needed: {top_n}

    Now return ONLY the JSON output:"""


def get_gemini_role_recommendations(profile: dict, top_n: int = 5) -> Optional[Dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    prompt = _build_prompt(profile, top_n)
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception:
        return None

    data = _extract_json(getattr(response, "text", ""))
    if not data:
        return None

    resume_skills = profile.get("skills", [])
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)
    desired_role = profile.get("career_preferences", {}).get("desired_role", "")

    # --- Build skill_analysis for the desired role ---
    skill_analysis = None
    desired_role_data = data.get("desired_role", {})
    if desired_role_data and (desired_role_data.get("matched_skills") or desired_role_data.get("missing_skills")):
        skill_analysis = _build_skill_analysis(
            desired_role=desired_role_data.get("role", desired_role),
            experience_level=desired_role_data.get("experience_level", experience_level),
            resume_skills=resume_skills,
            desired_role_data=desired_role_data,
        )

    # --- Process recommended roles ---
    roles = data.get("roles", [])
    if not isinstance(roles, list):
        return None

    processed_roles = []
    for role_data in roles:
        processed = _process_role(role_data, resume_skills)
        if processed:
            processed_roles.append(processed)

    # Sort by skill match percent DESC
    processed_roles.sort(key=lambda x: x["skill_match_percent"], reverse=True)

    # Assign ranks
    for idx, role in enumerate(processed_roles, start=1):
        role["rank"] = idx

    return {
        "skill_analysis": skill_analysis,
        "top_roles": processed_roles[:top_n],
        "source": "gemini_hybrid",
    }
import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from app.experience import map_experience_level
from app.ml.role_recommender import recommend_roles
from app.ml.skill_matcher import compute_skill_match

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


def _process_role(role_data: dict) -> Dict:
    role_name = role_data.get("role")
    experience_level = role_data.get("experience_level")

    if not role_name:
        return None

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


def _build_skill_analysis(desired_role: str, experience_level: str, desired_role_data: dict) -> Dict:
    """Build skill_analysis block for the desired role using Gemini's intelligent matching."""

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


def _build_prompt(
    profile: dict,
    dataset_recommendations: List[Dict],
    desired_role_analysis: Optional[Dict],
    top_n: int,
) -> str:
    """
    Build a focused prompt using pre-computed dataset analysis.
    Instead of sending the whole dataset, sends only top 10 pre-filtered roles
    from the dataset layer (precision), and asks Gemini to apply reasoning (intelligence).
    """
    skills = profile.get("skills", [])
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)
    desired_role = profile.get("career_preferences", {}).get("desired_role", "")

    skills_list = ", ".join(skills) if skills else "None provided"

    # --- Format desired role analysis from dataset ---
    if desired_role and desired_role_analysis and desired_role_analysis.get("matched_skills"):
        matched = desired_role_analysis.get("matched_skills", [])
        missing = desired_role_analysis.get("missing_skills", [])
        match_pct = desired_role_analysis.get("skill_match_percent", 0)
        desired_role_section = f"""DATASET ANALYSIS FOR DESIRED ROLE: {desired_role}
- Dataset skill match: {match_pct}%
- Matched skills (exact string match): {', '.join(matched) if matched else 'None'}
- Missing skills (exact string match): {', '.join(missing) if missing else 'None'}

The dataset uses simple string matching. Re-analyze using your reasoning — the candidate
likely knows MORE of the required skills than the dataset detected."""
    elif desired_role:
        desired_role_section = f"""DESIRED ROLE: {desired_role}
Note: This role was not found in the dataset. Analyze it based on your industry knowledge."""
    else:
        desired_role_section = "DESIRED ROLE: Not specified. Skip desired_role analysis — set desired_role to null in output."

    # --- Format top 10 dataset recommendations ---
    roles_section = ""
    for i, rec in enumerate(dataset_recommendations, 1):
        matched_str = ", ".join(rec.get("matched_skills", []))
        to_learn_str = ", ".join(rec.get("skills_to_learn", []))
        roles_section += (
            f"{i}. {rec['role']} (Dataset match: {rec['match_score']}%)\n"
            f"   Matched: {matched_str}\n"
            f"   To learn: {to_learn_str}\n"
            f"   Total required: {rec['total_required_skills']}\n\n"
        )

    return f"""You are an expert career advisor. Your task is to REFINE dataset-driven role recommendations using deep industry reasoning.

========================================
ABSOLUTE RULES (VIOLATING THESE = FAILURE):
========================================

RULE 1 — ROLE-SPECIFIC MATCHING ONLY:
  matched_skills must ONLY contain skills that are ACTUALLY RELEVANT to THAT SPECIFIC ROLE.
  Example: "Pandas", "Matplotlib", "Seaborn" are relevant to Data Scientist but NOT to Backend Developer.
  Do NOT blindly list all candidate skills as matched for every role.

RULE 2 — NEVER PUT KNOWN SKILLS IN skills_to_learn / missing_skills:
  If a candidate lists "Scikit-learn" in their skills, you MUST NOT put "scikit-learn" in skills_to_learn.
  Before writing skills_to_learn, cross-check EVERY item against the candidate's skill list.
  Also check for equivalences: if candidate has "MySQL", do NOT list "sql" as missing.

RULE 3 — SKILL EQUIVALENCE REASONING:
  Apply these when deciding if a candidate's skill covers a required skill:
  - Specific tool → parent concept: PostgreSQL → SQL, React → JavaScript, Docker → Containerization
  - Library → capability: Pandas → Data Manipulation, Scikit-learn → ML Fundamentals
  - Variant names are identical: "scikit-learn" = "sklearn", "node.js" = "nodejs", "MySQL" → covers "SQL"
  - Broad skill → fundamentals: Machine Learning → Supervised Learning concepts

RULE 4 — REALISTIC SKILL COUNTS:
  Each role should have 5-15 matched skills and 5-10 skills_to_learn (genuinely new knowledge only).
  If a candidate has 21 skills, it is IMPOSSIBLE for all 21 to be relevant to Backend Developer.

RULE 5 — OUTPUT INTEGRITY:
    The roles array MUST contain unique roles selected ONLY from the provided top 10 dataset recommendations.
    Do NOT include the desired role in the role_recommendations (in top_roles). desired_role is analysis-only.

RULE 6 — STRICT ROLE EXCLUSION:
- The desired_role MUST NEVER appear in the "roles" array under any condition.
- Before selecting top roles, you MUST REMOVE the desired_role from the candidate role list.
- Even if the desired_role has the highest match score, it MUST be excluded.
- If the desired_role appears in the output roles array, the response is INVALID and must be corrected.
========================================
CANDIDATE PROFILE:
========================================

Skills: {skills_list}
Experience: {experience_years} years ({experience_level})

========================================
PART 1: DESIRED ROLE RE-ANALYSIS
========================================

{desired_role_section}

Re-analyze this role. Determine which of the candidate's skills are ACTUALLY RELEVANT to this role.
A skill like "Teamwork" or "Tkinter" is NOT relevant to a Data Science role's core requirements.

========================================
PART 2: TOP 10 DATASET RECOMMENDATIONS (select best {top_n})
========================================

{roles_section}

STEP 1: Remove the desired_role completely from the dataset role list.

STEP 2: From the remaining roles, select the BEST {top_n} roles.

VERY IMPORTANT:
- The Desired Role SHOULD NOT be considered during ranking SHOULD NOT be seen in five role recommendations of output.
- It must NOT appear in the final roles array.
For each role, list ONLY the candidate skills that are TRULY RELEVANT to that role.

RANKING CRITERIA:
1. True skill relevance (NOT just having many skills — only count skills that matter for the role)
2. Career fit for this candidate's background
3. Learnability of missing skills
4. Diversity of career directions

========================================
FORMAT RULES:
========================================

- matched_skills: use EXACT skill names from candidate's list, but ONLY if relevant to the role
- skills_to_learn: standard lowercase names, NEVER include skills the candidate already has
- Do NOT include skill_match_percent — calculated externally
- experience_level: one of "Entry-Level", "Mid-Level", "Senior-Level"
- Output ONLY raw JSON. No markdown, no explanation.
- Response MUST start with {{ and end with }}

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

Now return ONLY the JSON output:"""


def get_gemini_role_recommendations(profile: dict, top_n: int = 5) -> Optional[Dict]:
    """
    Hybrid architecture: Dataset (precision) + Gemini (intelligence).

    Flow:
    1. Dataset layer: recommend_roles() → top 10 candidates (fast, precise)
    2. Dataset layer: compute_skill_match() → desired role analysis
    3. Gemini layer: refine with LLM reasoning → top 5 (intelligent)

    This saves tokens (10 roles vs entire dataset) and produces better results
    because Gemini focuses on reasoning over skill equivalences rather than
    searching through hundreds of roles.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    skills = profile.get("skills", [])
    experience_years = profile.get("experience", {}).get("years", 0)
    experience_level = map_experience_level(experience_years)
    desired_role = profile.get("career_preferences", {}).get("desired_role", "")

    # ── Step 1: Dataset precision layer ──────────────────────
    # Get top 10 from dataset (Gemini will select best 5 from these)
    dataset_recommendations = recommend_roles(
        resume_skills=skills,
        experience_years=experience_years,
        top_n=10,
    )

    # Get desired role skill analysis from dataset
    desired_role_analysis = None
    if desired_role:
        desired_role_analysis = compute_skill_match(
            resume_skills=skills,
            desired_role=desired_role,
            experience_level=experience_level,
        )

    # ── Step 2: Gemini intelligence layer ────────────────────
    prompt = _build_prompt(profile, dataset_recommendations, desired_role_analysis, top_n)
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception:
        return None

    data = _extract_json(getattr(response, "text", ""))
    if not data:
        return None

    # ── Step 3: Process Gemini's refined output ──────────────
    # Build skill_analysis for the desired role
    skill_analysis = None
    desired_role_data = data.get("desired_role")
    if desired_role_data and isinstance(desired_role_data, dict):
        if desired_role_data.get("matched_skills") or desired_role_data.get("missing_skills"):
            skill_analysis = _build_skill_analysis(
                desired_role=desired_role_data.get("role", desired_role),
                experience_level=desired_role_data.get("experience_level", experience_level),
                desired_role_data=desired_role_data,
            )

    # Process recommended roles
    roles = data.get("roles", [])
    if not isinstance(roles, list):
        return None

    processed_roles = []
    for role_data in roles:
        processed = _process_role(role_data)
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
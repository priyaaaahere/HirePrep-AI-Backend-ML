"""
main.py — Entry point / orchestration layer for the Skill Gap Analyzer.

Flow:
  1. Receive user input (job title + skills).
  2. Call job_matcher to find the best semantic match.
  3. Call skill_gap_engine to compute matched & missing skills.
  4. Return a structured, JSON-ready result.

Currently uses dummy test input.  FastAPI integration is commented out
at the bottom and can be enabled when the backend is ready.
"""

from job_matcher import match_job_role, best_match
from skill_gap_engine import compute_skill_gap


# ============================================================
# ===================== DUMMY TEST DATA ======================
# ============================================================

if __name__ == "__main__":

    resume_json = {
        "job_title": "HR Director",
        "skills": [
            "Recruitment",
            "Employee Relations",
            "Performance Management",
            "Payroll Management",
            "Leadership",
            "Communication"
        ]
    }

    user_title = resume_json["job_title"]
    user_skills = resume_json["skills"]

    # --- Step 1: Semantic job-title matching ---
    print(f"\n🔍 Searching for jobs similar to: \"{user_title}\"\n")
    top_matches = match_job_role(user_title, top_n=5, threshold=0.40)

    if not top_matches:
        print("❌ No matching job role found.")
    else:
        print(f"📋 Top {len(top_matches)} matches:")
        for i, m in enumerate(top_matches, 1):
            print(f"   {i}. {m['job_title']}  "
                  f"[{m['category']}]  "
                  f"(similarity: {m['similarity_score']:.4f})")

        # --- Step 2: Pick the best match and compute skill gap ---
        best = top_matches[0]
        result = compute_skill_gap(best, user_skills)

        print("\n✅ Matched Job Role:", result["matched_job_title"])
        print("📂 Category:", result["category"])
        print(f"📊 Similarity Score: {result['similarity_score']:.4f}")

        print("\n📌 Required Skills:")
        for s in result["required_skills"]:
            print("   -", s)

        print("\n🧠 Your Skills:")
        for s in result["user_skills"]:
            print("   -", s)

        print("\n✅ Matched Skills:")
        if result["matched_skills"]:
            for s in result["matched_skills"]:
                print("   -", s)
        else:
            print("   (none)")

        print("\n⚠️ Missing Skills (Skill Gap):")
        if result["missing_skills"]:
            for s in result["missing_skills"]:
                print("   -", s)
        else:
            print("   None 🎉")


# ============================================================
# ===================== FASTAPI VERSION ======================
# ============================================================

"""
Uncomment this section when integrating with the backend.

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResumeInput(BaseModel):
    job_title: str
    skills: list[str]

@app.post("/skill-gap")
def skill_gap(resume: ResumeInput):
    matched = best_match(resume.job_title)

    if not matched:
        return {"error": "No matching job role found."}

    result = compute_skill_gap(matched, resume.skills)
    return result
"""

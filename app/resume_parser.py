import re
from typing import List, Dict
import spacy

nlp = spacy.load("en_core_web_sm")


# ----------------------------
# Cleaning
# ----------------------------

def normalize_text(text: str) -> str:
    # Normalize line endings
    text = re.sub(r"\r", "\n", text)

    # Normalize bullets to newline
    text = re.sub(r"[•●▪◦►]", "\n", text)

    # Fix broken wrapped lines
    text = re.sub(r"\n(?=[a-z])", " ", text)

    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

def clean_project_title(line: str) -> str:
    """
    Cleans a project title line by removing dates and trailing separators.
    Keeps hyphens that are part of the title.
    """
    # Remove date ranges like "Jul’2025 - Nov’2025"
    line = re.sub(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z’']*\s*\d{4}.*",
        "",
        line,
        flags=re.IGNORECASE
    )

    # Remove trailing separators
    line = re.sub(r"[-–|:]\s*$", "", line)

    return line.strip()

def remove_dates(text: str) -> str:
    return re.sub(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z’']*\s*\d{2,4}",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()


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
    return match.group().strip() if match else ""



def extract_linkedin(text: str) -> str:
    match = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s]+", text)
    return match.group() if match else ""


def extract_github(text: str) -> str:
    match = re.search(r"(https?://)?(www\.)?github\.com/[^\s]+", text)
    return match.group() if match else ""


def extract_name(text: str) -> str:
    for line in text.split("\n")[:5]:
        line = line.strip()
        if (
            line
            and len(line.split()) <= 4
            and line.isupper()
            and "@" not in line
        ):
            return line.title()
        
    doc = nlp(text[:200])
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

    skills = []

    # Step 1: Split into lines first (preserve structure)
    lines = [line.strip() for line in skills_text.split("\n") if line.strip()]

    for line in lines:
        # Step 2: Remove category labels ONLY at start of line
        line = re.sub(r"^[A-Za-z\s/]+:\s*", "", line)

        # Step 3: Normalize separators
        line = line.replace("/", "|")

        # Step 4: Split into individual skills
        parts = re.split(r"\||,", line)

        for skill in parts:
            skill = skill.strip()
            if skill and len(skill) > 1:
                skills.append(skill)

    # Step 5: Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for s in skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_skills.append(s)

    return unique_skills


# ----------------------------
# Certifications extraction
# ----------------------------
def extract_certifications_from_section(cert_text: str) -> List[str]:
    if not cert_text:
        return []

    # Remove labels like "Certifications:"
    cert_text = re.sub(r"[A-Za-z\s]+:", "", cert_text)

    # Normalize separators
    cert_text = cert_text.replace("|", "\n")

    # Split on new lines, bullets, commas
    raw_certs = re.split(r"\n|•|,", cert_text)

    certifications = []

    for cert in raw_certs:
        cert = cert.strip()
        if not cert or len(cert) < 4:
            continue

        # 🔹 Remove dates (THIS was missing)
        cert = remove_dates(cert)

        # Remove trailing separators
        cert = re.sub(r"[-–|:]\s*$", "", cert)

        certifications.append(cert)

    # Remove duplicates (case-insensitive)
    seen = set()
    unique_certs = []
    for c in certifications:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_certs.append(c)

    return unique_certs



# ----------------------------
# Projects extraction
# ----------------------------
def extract_projects_from_section(project_text: str) -> List[str]:
    """
    Extract project titles using structural cues.
    Works even after bullets are removed during normalization.
    """
    if not project_text:
        return []

    lines = [line.strip() for line in project_text.split("\n") if line.strip()]
    projects = []

    for i, line in enumerate(lines):
        # Skip obvious non-title lines
        if line.lower().startswith("Tech:"):
            continue

        # Must start with uppercase
        if not line[0].isupper():
            continue

        # Titles are usually short-ish
        if len(line) > 150:
            continue

        # Look ahead: is there descriptive content below?
        description_lines = 0
        for j in range(i + 1, min(i + 6, len(lines))):
            next_line = lines[j]

            # Stop if next project starts
            if next_line[0].isupper() and len(next_line) < 150:
                break

            # Heuristic: description lines are longer and detailed
            if len(next_line) > 60:
                description_lines += 1

        # If followed by enough descriptive content → project title
        if description_lines >= 2:
            title = clean_project_title(line)
            title = remove_dates(title)

            if 5 < len(title) < 150:
                projects.append(title)

    # Deduplicate while preserving order
    seen = set()
    unique_projects = []
    for p in projects:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique_projects.append(p)

    return unique_projects

# ----------------------------
# Internships extraction
# ----------------------------
def extract_internships_from_section(intern_text: str) -> List[Dict[str, str]]:
    """
    Extract internship details from the Internship/Experience section.
    Returns a list of dicts with 'company' and 'role' keys.
    """
    if not intern_text:
        return []

    internships = []
    lines = intern_text.split("\n")
    
    current_company = ""
    current_role = ""
    
    # Common role keywords
    role_keywords = [
        "intern", "developer", "engineer", "analyst", "trainee", 
        "associate", "assistant", "consultant", "designer"
    ]
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        line_lower = line.lower()
        
        # Check if line contains a role
        has_role = any(kw in line_lower for kw in role_keywords)
        
        if has_role:
            # Try to split company and role
            # Common patterns: "Company Name - Role" or "Role at Company" or "Role, Company"
            
            # Pattern: "Role at Company"
            at_match = re.search(r"(.+?)\s+at\s+(.+)", line, re.IGNORECASE)
            if at_match:
                current_role = at_match.group(1).strip()
                current_company = at_match.group(2).strip()
                internships.append({"company": current_company, "role": current_role})
                continue
            
            # Pattern: "Company - Role" or "Company | Role"
            sep_match = re.split(r"\s*[-|–]\s*", line)
            if len(sep_match) >= 2:
                # Determine which part is the role
                if any(kw in sep_match[0].lower() for kw in role_keywords):
                    current_role = sep_match[0].strip()
                    current_company = sep_match[1].strip()
                else:
                    current_company = sep_match[0].strip()
                    current_role = sep_match[1].strip()
                internships.append({"company": current_company, "role": current_role})
                continue
            
            # Just the role/title line
            current_role = line
            if current_company:
                internships.append({"company": current_company, "role": current_role})
                current_company = ""
        else:
            # Might be a company name line
            # Check if it looks like a company (has Inc, Ltd, LLC, or is capitalized)
            if re.search(r"(Inc|Ltd|LLC|Corp|Company|Technologies|Solutions|Systems|Labs|Studio)", line):
                current_company = re.split(r"[-|–,]", line)[0].strip()
            elif line[0].isupper() and len(line) < 50:
                current_company = line

    # Remove duplicates
    seen = set()
    unique_internships = []
    for intern in internships:
        key = (intern["company"].lower(), intern["role"].lower())
        if key not in seen:
            seen.add(key)
            unique_internships.append(intern)
    
    return unique_internships


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
# PARSE FOR ML/ANALYSIS
# ----------------------------
def parse_resume_for_analysis(resume_text: str) -> Dict:
    """
    Parse resume and return data structure optimized for ML analysis.
    Returns counts and skills list for placement prediction and role matching.
    """
    clean = normalize_text(resume_text)

    # Extract sections
    skills_section = extract_section(clean, "Skills")
    skills = extract_skills_from_section(skills_section)

    cert_section = extract_section(clean, "Certifications")
    certifications = extract_certifications_from_section(cert_section)

    project_section = extract_section(clean, "Projects")
    projects = extract_projects_from_section(project_section)

    intern_section = extract_section(clean, "Internship") or extract_section(clean, "Experience")
    internships = extract_internships_from_section(intern_section)

    # Extract CGPA as float for ML
    cgpa_str = extract_cgpa(clean)
    cgpa = float(cgpa_str) if cgpa_str else 0.0

    return {
        "cgpa": cgpa,
        "project_count": len(projects),
        "internship_count": len(internships),
        "certification_count": len(certifications),
        "skills": skills  # Actual skills for matching algorithms
    }


# ----------------------------
# PARSE FOR FRONTEND DISPLAY
# ----------------------------
def parse_resume_for_frontend(resume_text: str) -> Dict:
    """
    Parse resume and return detailed data structure for frontend display.
    Returns all extracted information with metadata for user editing.
    """
    clean = normalize_text(resume_text)

    # Extract sections
    skills_section = extract_section(clean, "Skills")
    skills = extract_skills_from_section(skills_section)

    cert_section = extract_section(clean, "Certifications")
    certifications = extract_certifications_from_section(cert_section)

    project_section = extract_section(clean, "Projects")
    projects = extract_projects_from_section(project_section)

    intern_section = extract_section(clean, "Internship") or extract_section(clean, "Experience")
    internships = extract_internships_from_section(intern_section)

    return {
        # Personal Info
        "name": extract_name(clean),
        "email": extract_email(clean),
        "phone": extract_phone(clean),
        "github": extract_github(clean),
        "linkedin": extract_linkedin(clean),
        
        # Academic
        "cgpa": extract_cgpa(clean),
        
        # Experience & Skills
        "projects": projects,  # List[str] - project names
        "internships": internships,  # List[{company, role}]
        "certifications": certifications,  # List[str] - cert titles
        "skills": skills,  # List[str] - skill names
        
        # Metadata
        "meta": {
            "status": "draft",
            "editable": True,
            "user_verified": False
        }
    }


# ----------------------------
# LEGACY: build_editable_profile (wrapper for backward compatibility)
# ----------------------------
def build_editable_profile(resume_text: str) -> Dict:
    """
    Legacy function - now wraps parse_resume_for_frontend.
    Use parse_resume_for_frontend() or parse_resume_for_analysis() instead.
    """
    return parse_resume_for_frontend(resume_text)

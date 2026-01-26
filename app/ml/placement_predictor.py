import joblib
import pandas as pd

MODEL_PATH = "app/ml/placement_model.pkl"
_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def validate_communication_rating(value):
    """
    Ensures communication rating is a float between 1 and 5
    """
    try:
        rating = float(value)
    except (TypeError, ValueError):
        raise ValueError("Communication rating must be a number")

    if rating < 1.0 or rating > 5.0:
        raise ValueError("Communication rating must be between 1 and 5")

    return rating


def json_to_placement_features(profile: dict) -> dict:
    education = profile.get("education", {})
    projects = profile.get("projects", [])
    certifications = profile.get("certifications", [])
    internships = profile.get("internships", [])
    skills = profile.get("skills", [])

    placement_inputs = profile.get("placement_inputs", {})

    #For validation purpose of input provided by user
    communication_rating = validate_communication_rating(
        placement_inputs.get("communication_rating", 4.0)
    )

    features = {
        "CGPA": float(education.get("cgpa") or 0),
        "Projects": len(projects),
        "Workshops/Certifications": len(certifications),
        "Skills": len(skills),
        "Communication Skill Rating": communication_rating,
        "Internship": "Yes" if len(internships) > 0 else "No",
        "Hackathon": placement_inputs.get("hackathon") or "No",
        "12th Percentage": float(placement_inputs.get("twelfth_percent") or 0),
        "10th Percentage": float(placement_inputs.get("tenth_percent") or 0),
        "backlogs": int(placement_inputs.get("backlogs") or 0),
    }

    return features


def predict_base_probability(profile: dict) -> float:
    model = load_model()
    features = json_to_placement_features(profile)

    X = pd.DataFrame([features])
    prob_placed = model.predict_proba(X)[0][1]

    return float(prob_placed) * 100  # return percentage


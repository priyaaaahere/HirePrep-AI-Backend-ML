def map_experience_level(years: int) -> str:
    """
    Map years of experience to experience level category.
    This matches the logic used in data/clean_job_dataset.py

    Entry-Level: 0-2 years
    Mid-Level: 3-4 years
    Senior-Level: 5+ years
    """
    if years <= 2:
        return "Entry-Level"
    if years <= 4:
        return "Mid-Level"
    return "Senior-Level"

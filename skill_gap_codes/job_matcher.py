"""
job_matcher.py — Semantic job-title matching using Sentence Transformers.

Loads all job titles from jobs.json, pre-computes their embeddings with the
`all-MiniLM-L6-v2` model, and exposes a function to find the top-N most
similar jobs for any free-form user input.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import load_json, normalize, cosine_similarity

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data", "JSON", "jobs.json")

# ---------- LOAD MODEL & DATA ----------
print("[job_matcher] Loading SentenceTransformer model …")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

print("[job_matcher] Loading jobs from JSON …")
JOBS = load_json(JSON_PATH)

# Pre-compute embeddings for every job title (done once at import time)
JOB_TITLES = [job["job_title"] for job in JOBS]
JOB_TITLE_NORMALIZED = [normalize(t) for t in JOB_TITLES]
JOB_EMBEDDINGS = MODEL.encode(JOB_TITLE_NORMALIZED, show_progress_bar=False)

print(f"[job_matcher] {len(JOBS)} job titles embedded and ready.")


# ---------- PUBLIC API ----------
def match_job_role(user_title: str,
                   top_n: int = 5,
                   threshold: float = 0.40):
    """
    Return the top-N most semantically similar jobs for *user_title*.

    Each result is a dict with keys:
        job_title, category, job_skill_set, similarity_score

    Only results with similarity >= *threshold* are included.
    Returns an empty list when nothing passes the threshold.
    """
    user_vec = MODEL.encode([normalize(user_title)], show_progress_bar=False)[0]

    # Compute cosine similarity against every stored embedding
    scores = [
        cosine_similarity(user_vec, job_vec)
        for job_vec in JOB_EMBEDDINGS
    ]

    # Pair each job with its score and sort descending
    ranked = sorted(
        zip(JOBS, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    # Filter by threshold and trim to top_n
    results = []
    for job, score in ranked:
        if score < threshold:
            break                       # all remaining are lower
        results.append({
            "job_title":      job["job_title"],
            "category":       job["category"],
            "job_skill_set":  job["job_skill_set"],
            "similarity_score": round(score, 4),
        })
        if len(results) >= top_n:
            break

    return results


def best_match(user_title: str, threshold: float = 0.40):
    """Convenience wrapper — return the single best match or None."""
    matches = match_job_role(user_title, top_n=1, threshold=threshold)
    return matches[0] if matches else None

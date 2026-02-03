"""
NLP-based Role Matcher
======================
Uses Sentence Transformers (BERT embeddings) for semantic role matching.
When user enters a role not in dataset (e.g., "Fraud Analyst"), 
finds the most similar role (e.g., "Data Analyst") based on semantic similarity.

Why Sentence Transformers over TF-IDF?
- Captures semantic meaning: "Fraud Analyst" ≈ "Data Analyst" even without word overlap
- Pre-computed embeddings: Fast lookup at runtime
- Lightweight: all-MiniLM-L6-v2 is only ~80MB

Usage:
    matcher = RoleMatcher()
    result = matcher.find_similar_role("Fraud Analyst", "fresher")
    # Returns: {"matched_role": "Data Analyst", "similarity": 0.85, ...}
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import os
import pickle

# Lazy import to handle if sentence-transformers not installed
_model = None
_embeddings_cache = None

DATASET_PATH = "data/job_dataset_aggregated.csv"
EMBEDDINGS_CACHE_PATH = "app/ml/models/role_embeddings.pkl"

# Similarity threshold - below this, consider it a poor match
SIMILARITY_THRESHOLD = 0.3


def get_model():
    """
    Lazy load the sentence transformer model.
    Uses all-MiniLM-L6-v2 - fast, lightweight, good quality.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def load_dataset_roles() -> pd.DataFrame:
    """
    Load unique roles from the aggregated dataset.
    """
    df = pd.read_csv(DATASET_PATH)
    return df[["Title", "ExperienceLevel"]].drop_duplicates()


def get_unique_roles() -> List[str]:
    """
    Get list of unique role titles from dataset.
    """
    df = load_dataset_roles()
    return df["Title"].unique().tolist()


def compute_role_embeddings(roles: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of role titles.
    """
    model = get_model()
    embeddings = model.encode(roles, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def save_embeddings_cache(roles: List[str], embeddings: np.ndarray):
    """
    Save pre-computed embeddings to disk for faster startup.
    """
    cache_data = {
        "roles": roles,
        "embeddings": embeddings
    }
    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)


def load_embeddings_cache() -> Optional[Dict]:
    """
    Load pre-computed embeddings from disk.
    """
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def get_role_embeddings() -> Tuple[List[str], np.ndarray]:
    """
    Get role embeddings - from cache if available, otherwise compute.
    """
    global _embeddings_cache
    
    if _embeddings_cache is not None:
        return _embeddings_cache["roles"], _embeddings_cache["embeddings"]
    
    # Try loading from disk
    cache = load_embeddings_cache()
    if cache is not None:
        current_roles = set(get_unique_roles())
        cached_roles = set(cache["roles"])
        
        # Validate cache has all current roles
        if current_roles == cached_roles:
            _embeddings_cache = cache
            return cache["roles"], cache["embeddings"]
    
    # Compute fresh embeddings
    roles = get_unique_roles()
    embeddings = compute_role_embeddings(roles)
    
    # Cache in memory and on disk
    _embeddings_cache = {"roles": roles, "embeddings": embeddings}
    save_embeddings_cache(roles, embeddings)
    
    return roles, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def find_similar_role(
    query_role: str,
    experience_level: str = "",
    top_n: int = 3
) -> Dict:
    """
    Find the most similar role(s) in the dataset using semantic similarity.
    
    Args:
        query_role: User's desired role (may not be in dataset)
        experience_level: Experience level filter (optional)
        top_n: Number of similar roles to return
    
    Returns:
        Dictionary with matched role info and similarity scores
    """
    # First, check for exact match (case-insensitive)
    df = load_dataset_roles()
    df["Title_lower"] = df["Title"].str.lower()
    query_lower = query_role.lower().strip()
    
    exact_match = df[df["Title_lower"] == query_lower]
    if not exact_match.empty:
        matched_title = exact_match.iloc[0]["Title"]
        return {
            "query_role": query_role,
            "matched_role": matched_title,
            "similarity": 1.0,
            "match_type": "exact",
            "alternatives": [],
            "note": "Exact match found in dataset"
        }
    
    # Check for partial/substring match
    partial_match = df[df["Title_lower"].str.contains(query_lower, na=False)]
    if not partial_match.empty:
        matched_title = partial_match.iloc[0]["Title"]
        return {
            "query_role": query_role,
            "matched_role": matched_title,
            "similarity": 0.95,
            "match_type": "partial",
            "alternatives": partial_match["Title"].unique().tolist()[:top_n],
            "note": "Partial match found in dataset"
        }
    
    # Semantic search using embeddings
    roles, embeddings = get_role_embeddings()
    
    # Compute embedding for query role
    model = get_model()
    query_embedding = model.encode([query_role], convert_to_numpy=True)[0]
    
    # Compute similarities
    similarities = []
    for i, role in enumerate(roles):
        sim = cosine_similarity(query_embedding, embeddings[i])
        similarities.append((role, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by experience level if provided
    if experience_level:
        exp_lower = experience_level.lower()
        level_roles = df[df["ExperienceLevel"].str.lower().str.contains(exp_lower, na=False)]["Title"].unique()
        level_roles_set = set(level_roles)
        
        # Prioritize roles that match the experience level
        filtered_similarities = [
            (role, sim) for role, sim in similarities 
            if role in level_roles_set
        ]
        
        if filtered_similarities:
            similarities = filtered_similarities
    
    # Get top matches
    top_matches = similarities[:top_n]
    best_match = top_matches[0]
    
    # Determine match quality
    if best_match[1] >= 0.7:
        match_quality = "high"
    elif best_match[1] >= SIMILARITY_THRESHOLD:
        match_quality = "moderate"
    else:
        match_quality = "low"
    
    return {
        "query_role": query_role,
        "matched_role": best_match[0],
        "similarity": round(float(best_match[1]), 4),
        "match_type": "semantic",
        "match_quality": match_quality,
        "alternatives": [
            {"role": role, "similarity": round(float(sim), 4)}
            for role, sim in top_matches[1:]
        ],
        "note": f"Semantic match: '{query_role}' → '{best_match[0]}'"
    }


def get_matched_role_for_skills(
    query_role: str,
    experience_level: str = ""
) -> str:
    """
    Convenience function: returns just the best matched role name.
    Used by skill_matcher.py for skill lookup.
    
    Args:
        query_role: User's desired role
        experience_level: Experience level (optional)
    
    Returns:
        Best matched role name from dataset
    """
    result = find_similar_role(query_role, experience_level)
    
    # Only return matched role if similarity is above threshold
    if result["similarity"] >= SIMILARITY_THRESHOLD:
        return result["matched_role"]
    
    return query_role  # Return original if no good match


def precompute_embeddings():
    """
    Utility function to pre-compute and cache all role embeddings.
    Run this once after dataset changes to speed up first request.
    
    Usage:
        python -c "from app.ml.role_matcher import precompute_embeddings; precompute_embeddings()"
    """
    print("Loading dataset roles...")
    roles = get_unique_roles()
    print(f"Found {len(roles)} unique roles")
    
    print("Computing embeddings (this may take a moment)...")
    embeddings = compute_role_embeddings(roles)
    
    print("Saving embeddings cache...")
    save_embeddings_cache(roles, embeddings)
    
    print(f"✅ Embeddings cached to: {EMBEDDINGS_CACHE_PATH}")
    return len(roles)


# Role aliases for common variations (supplements semantic matching)
ROLE_ALIASES = {
    # Fraud/Risk related → Data Analyst family
    "fraud analyst": "Data Analyst",
    "fraud detection analyst": "Data Analyst",
    "risk analyst": "Data Analyst",
    "credit risk analyst": "Data Analyst",
    "anti-money laundering analyst": "Data Analyst",
    "aml analyst": "Data Analyst",
    
    # ML variations
    "machine learning engineer": "ML Engineer",
    "ml developer": "ML Engineer",
    "ai developer": "AI Engineer",
    "artificial intelligence engineer": "AI Engineer",
    
    # Data variations
    "data science engineer": "Data Scientist",
    "analytics engineer": "Data Engineer",
    "etl developer": "Data Engineer",
    
    # Web variations  
    "web developer": "Full Stack Developer",
    "front end developer": "Frontend Developer",
    "back end developer": "Backend Developer",
    "react developer": "Frontend Developer",
    "angular developer": "Frontend Developer",
    "node developer": "Backend Developer",
    "nodejs developer": "Backend Developer",
    
    # Mobile variations
    "ios developer": "iOS Developer",
    "mobile developer": "Android Developer",
    "mobile app developer": "Android Developer",
    "flutter developer": "Mobile Developer",
    "react native developer": "Mobile Developer",
    
    # DevOps variations
    "site reliability engineer": "DevOps Engineer",
    "sre": "DevOps Engineer",
    "platform engineer": "DevOps Engineer",
    "infrastructure engineer": "Cloud Engineer",
}


def get_role_from_alias(role: str) -> Optional[str]:
    """
    Check if role matches a known alias.
    """
    role_lower = role.lower().strip()
    return ROLE_ALIASES.get(role_lower)


def smart_role_match(
    query_role: str,
    experience_level: str = ""
) -> Dict:
    """
    Smart role matching combining:
    1. Exact match
    2. Alias lookup
    3. Semantic similarity
    
    This is the main entry point for role matching.
    """
    # 1. Check alias first (fastest)
    alias_match = get_role_from_alias(query_role)
    if alias_match:
        return {
            "query_role": query_role,
            "matched_role": alias_match,
            "similarity": 0.98,
            "match_type": "alias",
            "alternatives": [],
            "note": f"Alias match: '{query_role}' → '{alias_match}'"
        }
    
    # 2. Use semantic matching (handles exact, partial, and semantic)
    return find_similar_role(query_role, experience_level)

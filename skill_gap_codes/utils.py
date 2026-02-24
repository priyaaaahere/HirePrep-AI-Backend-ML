"""
utils.py — Reusable helper functions for the Skill Gap Analyzer.

Provides:
  - Text normalization (lowercasing, punctuation removal)
  - JSON file loading
  - Cosine similarity between dense vectors
"""

import re
import json
import os
import numpy as np


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces
    return text.strip()


def load_json(path: str):
    """Load and return a JSON file; raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two numpy-compatible vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))

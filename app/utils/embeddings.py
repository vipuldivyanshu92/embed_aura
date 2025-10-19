"""Lightweight embedding generation for offline operation."""

import hashlib

import numpy as np

from app.config import get_settings


def generate_embedding(text: str) -> list[float]:
    """
    Generate a pseudo-embedding vector from text using deterministic hashing.

    This is a lightweight fallback for offline operation. In production,
    this would be replaced with a real embedding model (e.g., sentence-transformers).

    Args:
        text: Input text

    Returns:
        Embedding vector of configured dimensions
    """
    settings = get_settings()
    embed_dims = settings.embed_dims

    # Normalize text
    normalized = text.lower().strip()

    # Generate deterministic seed from text
    text_hash = hashlib.sha256(normalized.encode()).digest()
    seed = int.from_bytes(text_hash[:4], byteorder="big")

    # Generate pseudo-random vector
    rng = np.random.RandomState(seed)
    vector = rng.randn(embed_dims)

    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()

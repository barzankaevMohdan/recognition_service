"""
Embedding matching module.

Matches face embeddings against known employees using cosine similarity.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..config import Config


def match_embedding_to_employee(
    embedding: np.ndarray,
    known_embeddings: List[np.ndarray],
    known_ids: List[int],
    config: Config
) -> Tuple[Optional[int], float]:
    """
    Match face embedding to known employees.
    
    Uses cosine similarity (dot product for normalized embeddings).
    
    Args:
        embedding: Face embedding to match (normalized)
        known_embeddings: List of known employee embeddings
        known_ids: List of corresponding employee IDs
        config: Service configuration
    
    Returns:
        Tuple of (employee_id, confidence) or (None, 0.0) if no match
        
    Confidence is cosine similarity in range [0, 1]:
    - 1.0 = perfect match
    - 0.0 = completely different
    """
    if len(known_embeddings) == 0:
        return None, 0.0
    
    # Compute cosine similarities (dot product for normalized vectors)
    similarities = [
        float(np.dot(embedding, known_emb))
        for known_emb in known_embeddings
    ]
    
    # Find best match
    best_idx = int(np.argmax(similarities))
    best_similarity = similarities[best_idx]
    
    # Check threshold
    if best_similarity > config.insightface_threshold:
        return known_ids[best_idx], best_similarity
    
    return None, 0.0









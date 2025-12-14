"""
Recognition algorithms package.

Contains modules for:
- Face quality assessment
- Image preprocessing
- Face tracking
- Presence management
- Embedding matching
"""

from .quality import compute_blur_score, is_face_acceptable
from .preprocessing import preprocess_face_for_insightface
from .tracker import FaceTrack, FaceTracker, compute_iou
from .presence import PresenceManager
from .matching import match_embedding_to_employee

__all__ = [
    'compute_blur_score',
    'is_face_acceptable',
    'preprocess_face_for_insightface',
    'FaceTrack',
    'FaceTracker',
    'compute_iou',
    'PresenceManager',
    'match_embedding_to_employee',
]









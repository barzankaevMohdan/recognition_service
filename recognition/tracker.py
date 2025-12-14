"""
Face tracking module.

Tracks faces across frames using IoU (Intersection over Union) matching.
Accumulates embeddings per track for more reliable recognition.
"""

import time
import numpy as np
from typing import List, Optional, Dict
from ..config import Config
from ..logging_config import get_logger

logger = get_logger(__name__)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union for two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
    
    Returns:
        IoU value in range [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class FaceTrack:
    """
    Represents a single face track across frames.
    
    Accumulates embeddings and quality scores for reliable recognition.
    """
    
    def __init__(self, track_id: int):
        """
        Initialize face track.
        
        Args:
            track_id: Unique track identifier
        """
        self.track_id = track_id
        self.embeddings: List[np.ndarray] = []
        self.quality_scores: List[Dict] = []
        self.last_bbox: Optional[np.ndarray] = None
        self.last_update_time: float = time.time()
        self.recognized_employee_id: Optional[int] = None
        self.recognition_confidence: float = 0.0
    
    def add_embedding(
        self,
        embedding: np.ndarray,
        quality: Dict,
        bbox: np.ndarray
    ) -> None:
        """
        Add embedding to track.
        
        Args:
            embedding: Face embedding
            quality: Quality metrics dict
            bbox: Bounding box
        """
        self.embeddings.append(embedding)
        self.quality_scores.append(quality)
        self.last_bbox = bbox
        self.last_update_time = time.time()
    
    def is_ready_for_recognition(self, config: Config) -> bool:
        """
        Check if track has enough embeddings for recognition.
        
        Args:
            config: Service configuration
        
        Returns:
            True if ready
        """
        return len(self.embeddings) >= config.min_embeddings_per_track
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """
        Get average embedding across all frames.
        
        More reliable than single frame.
        
        Returns:
            Average embedding or None
        """
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)
    
    def is_alive(self, config: Config) -> bool:
        """
        Check if track is still active.
        
        Args:
            config: Service configuration
        
        Returns:
            True if recently updated
        """
        return (time.time() - self.last_update_time) < config.track_max_age_seconds


class FaceTracker:
    """
    Manages multiple face tracks across frames.
    
    Matches detected faces to existing tracks using IoU.
    """
    
    def __init__(self, config: Config):
        """
        Initialize face tracker.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 1
    
    def update(
        self,
        faces: List,
        frame: np.ndarray,
        known_embeddings: List[np.ndarray],
        known_ids: List[int]
    ) -> List[FaceTrack]:
        """
        Update tracks with newly detected faces.
        
        Args:
            faces: List of InsightFace detection results
            frame: Current frame for face cropping
            known_embeddings: Known employee embeddings
            known_ids: Known employee IDs
        
        Returns:
            List of tracks with recognized employees
        """
        from .matching import match_embedding_to_employee
        from .quality import is_face_acceptable
        from .preprocessing import preprocess_face_for_insightface
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.is_alive(self.config)]
        
        matched_track_ids = set()
        
        for face in faces:
            bbox = face.bbox
            embedding = face.normed_embedding
            
            # Crop face from frame
            x1, y1, x2, y2 = bbox.astype(int)
            face_crop = frame[y1:y2, x1:x2] if frame is not None else None
            
            if face_crop is None or face_crop.size == 0:
                continue
            
            # Quality check
            acceptable, quality = is_face_acceptable(face_crop, bbox, self.config)
            
            if not acceptable:
                # Bad quality - skip
                continue
            
            # Find matching track
            best_track = self._find_matching_track(bbox, matched_track_ids)
            
            if best_track:
                # Preprocessing (as in original code)
                preprocessed = preprocess_face_for_insightface(face_crop, self.config)
                
                # Update existing track
                best_track.add_embedding(embedding, quality, bbox)
                matched_track_ids.add(best_track.track_id)
                
                # Try recognition if ready
                if (best_track.is_ready_for_recognition(self.config) and
                    not best_track.recognized_employee_id):
                    
                    avg_embedding = best_track.get_average_embedding()
                    if avg_embedding is not None:
                        emp_id, confidence = match_embedding_to_employee(
                            avg_embedding, known_embeddings, known_ids, self.config
                        )
                        
                        if emp_id:
                            best_track.recognized_employee_id = emp_id
                            best_track.recognition_confidence = confidence
                            logger.info(
                                f'Track {best_track.track_id} → Employee {emp_id} '
                                f'(confidence: {confidence:.3f}, '
                                f'embeddings: {len(best_track.embeddings)})'
                            )
            else:
                # Preprocessing (as in original code)
                preprocessed = preprocess_face_for_insightface(face_crop, self.config)
                
                # Create new track
                new_track = FaceTrack(self.next_track_id)
                self.next_track_id += 1
                new_track.add_embedding(embedding, quality, bbox)
                self.tracks.append(new_track)
                logger.debug(f'Created new track {new_track.track_id}')
        
        # Return recognized tracks
        return [t for t in self.tracks if t.recognized_employee_id is not None]
    
    def _find_matching_track(
        self,
        bbox: np.ndarray,
        matched_track_ids: set
    ) -> Optional[FaceTrack]:
        """
        Find track matching given bbox.
        
        Args:
            bbox: Bounding box to match
            matched_track_ids: Set of already matched track IDs
        
        Returns:
            Matching track or None
        """
        best_track = None
        best_iou = 0.0
        
        for track in self.tracks:
            if track.track_id in matched_track_ids:
                continue
            if track.last_bbox is None:
                continue
            
            iou = compute_iou(bbox, track.last_bbox)
            if iou > self.config.iou_threshold and iou > best_iou:
                best_iou = iou
                best_track = track
        
        return best_track


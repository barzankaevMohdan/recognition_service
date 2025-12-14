"""
Face quality assessment module.

Evaluates face quality based on:
- Size (height and width in pixels)
- Sharpness (Laplacian variance)
- Brightness (mean pixel value)
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from ..config import Config


def compute_blur_score(gray_face: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    
    Higher values indicate sharper images.
    
    Args:
        gray_face: Grayscale face image
    
    Returns:
        Blur score (Laplacian variance)
    """
    return cv2.Laplacian(gray_face, cv2.CV_64F).var()


def is_face_acceptable(
    face_img_bgr: np.ndarray,
    bbox: np.ndarray,
    config: Config
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if face quality is acceptable for recognition.
    
    Criteria:
    - Face height >= min_face_height_pixels
    - Blur score >= min_blur_variance
    
    Args:
        face_img_bgr: Face image in BGR format
        bbox: Bounding box [x1, y1, x2, y2]
        config: Service configuration
    
    Returns:
        Tuple of (acceptable: bool, metrics: dict)
        
    Metrics dict contains:
        - height: Face height in pixels
        - width: Face width in pixels
        - blur_score: Laplacian variance
        - brightness: Mean pixel value
    """
    try:
        # Extract bbox coordinates
        x1, y1, x2, y2 = bbox.astype(int)
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Convert to grayscale for blur computation
        gray_face = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Compute metrics
        blur_score = compute_blur_score(gray_face)
        mean_brightness = float(np.mean(gray_face))
        
        metrics = {
            'height': float(face_height),
            'width': float(face_width),
            'blur_score': float(blur_score),
            'brightness': mean_brightness,
        }
        
        # Check thresholds
        if face_height < config.min_face_height_pixels:
            return False, metrics
        
        if blur_score < config.min_blur_variance:
            return False, metrics
        
        return True, metrics
        
    except Exception as e:
        # Return False on any error
        return False, {'error': str(e)}









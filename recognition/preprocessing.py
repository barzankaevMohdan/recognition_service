"""
Image preprocessing module.

Applies enhancement pipeline to improve face recognition:
1. Denoising (fastNlMeansDenoisingColored)
2. CLAHE on luminance channel (contrast enhancement)
3. Unsharp mask (sharpening)
"""

import cv2
import numpy as np
from ..config import Config


def preprocess_face_for_insightface(face_bgr: np.ndarray, config: Config) -> np.ndarray:
    """
    Preprocess face image for InsightFace recognition.
    
    Pipeline:
    1. Denoise - Remove noise while preserving details
    2. CLAHE - Enhance contrast on luminance channel only
    3. Unsharp mask - Gentle sharpening
    
    Args:
        face_bgr: Face image in BGR format
        config: Service configuration
    
    Returns:
        Preprocessed face image
    """
    if not config.enable_preprocessing:
        return face_bgr
    
    try:
        # Step 1: Denoising
        # Preserve details with moderate strength
        denoised = cv2.fastNlMeansDenoisingColored(
            face_bgr,
            None,
            h=config.denoise_strength,
            hColor=config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Step 2: CLAHE on luminance channel
        # Convert to YCrCb (better for CLAHE than HSV)
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply CLAHE to luminance channel only
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(8, 8)
        )
        y_enhanced = clahe.apply(y)
        
        # Merge back
        enhanced = cv2.cvtColor(
            cv2.merge([y_enhanced, cr, cb]),
            cv2.COLOR_YCrCb2BGR
        )
        
        # Step 3: Unsharp mask (gentle sharpening)
        # Create blurred version
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        
        # Blend: original * 1.5 - blurred * 0.5
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return sharpened
        
    except Exception as e:
        # Return original on error
        return face_bgr









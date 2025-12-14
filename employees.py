"""
Employee management module.

Handles loading employee data from backend and building face embeddings.
"""

import requests
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from .config import Config
from .logging_config import get_logger
from .utils.cache import load_cache, save_cache, get_employees_hash

logger = get_logger(__name__)


def load_employees_from_backend(
    config: Config,
    face_app: Any
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load employees from backend and build face embeddings.
    
    Args:
        config: Service configuration
        face_app: InsightFace FaceAnalysis instance
    
    Returns:
        Tuple of (embeddings list, employee IDs list)
    """
    logger.info('Loading employees from backend...')
    
    try:
        # Fetch employees from backend
        url = f'{config.backend_url}/api/employees'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        employees = response.json()
        
        logger.info(f'Fetched {len(employees)} employees from backend')
        
        # Check cache
        current_hash = get_employees_hash(employees)
        cached_encodings, cached_ids, cached_hash = load_cache(config.cache_file)
        
        if cached_encodings and cached_hash == current_hash:
            logger.info(f'✅ Using cached encodings for {len(cached_ids)} employees')
            return cached_encodings, cached_ids
        
        logger.info('Cache miss or outdated, processing photos...')
        
        # Build embeddings
        known_embeddings: List[np.ndarray] = []
        known_ids: List[int] = []
        
        for emp in employees:
            photo_url = emp.get('photoUrl')
            if not photo_url:
                logger.warning(f"Employee {emp.get('id')} has no photo, skipping")
                continue
            
            # Build full URL
            if photo_url.startswith('http'):
                full_url = photo_url
            else:
                full_url = config.backend_url + photo_url
            
            try:
                embedding = _process_employee_photo(
                    emp, full_url, face_app, config
                )
                
                if embedding is not None:
                    known_embeddings.append(embedding)
                    known_ids.append(emp['id'])
                    
            except Exception as e:
                logger.error(f"Failed to process employee {emp.get('id')}: {e}")
                continue
        
        # Save cache
        if known_embeddings:
            save_cache(known_embeddings, known_ids, current_hash, config.cache_file)
        
        logger.info(f'✅ Loaded {len(known_ids)} employees with valid photos')
        return known_embeddings, known_ids
        
    except requests.exceptions.RequestException as e:
        logger.error(f'Failed to fetch employees from backend: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading employees: {e}')
        raise


def _process_employee_photo(
    emp: Dict[str, Any],
    photo_url: str,
    face_app: Any,
    config: Config
) -> np.ndarray | None:
    """
    Process single employee photo and extract embedding.
    
    Args:
        emp: Employee data dict
        photo_url: Full URL to photo
        face_app: InsightFace instance
        config: Service configuration
    
    Returns:
        Face embedding or None if processing failed
    """
    emp_id = emp.get('id')
    emp_name = emp.get('name', 'Unknown')
    
    logger.info(f"Processing photo for {emp_name} (ID: {emp_id})...")
    
    try:
        # Download photo
        img_response = requests.get(photo_url, timeout=10)
        img_response.raise_for_status()
        
        # Decode image
        nparr = np.frombuffer(img_response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"Failed to decode image for {emp_name}")
            return None
        
        # Import here to avoid circular dependency
        from .recognition.preprocessing import preprocess_face_for_insightface
        from .recognition.quality import is_face_acceptable
        
        # Preprocess
        preprocessed = preprocess_face_for_insightface(image, config)
        
        # Detect faces
        faces = face_app.get(preprocessed)
        
        if not faces:
            logger.warning(f"No face found for {emp_name}")
            return None
        
        # Use first face
        face = faces[0]
        
        # Quality check
        acceptable, quality = is_face_acceptable(image, face.bbox, config)
        
        if not acceptable:
            logger.warning(
                f"Face quality too low for {emp_name}: "
                f"height={quality.get('height')}px, "
                f"blur={quality.get('blur_score', 0):.1f}"
            )
            return None
        
        # Extract embedding
        embedding = face.normed_embedding
        
        logger.info(
            f"✅ {emp_name} - embedding created "
            f"(quality: h={quality.get('height')}px, "
            f"blur={quality.get('blur_score', 0):.1f})"
        )
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error processing photo for {emp_name}: {e}")
        return None


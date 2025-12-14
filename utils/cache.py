"""
Embeddings cache module.

Caches face embeddings to avoid reprocessing on every restart.
"""

import os
import pickle
import hashlib
import time
from typing import List, Tuple, Optional, Dict, Any
from ..logging_config import get_logger

logger = get_logger(__name__)


def get_employees_hash(employees: List[Dict[str, Any]]) -> str:
    """
    Compute hash of employees list for cache validation.
    
    Args:
        employees: List of employee dicts
    
    Returns:
        MD5 hash string
    """
    data = ''.join([
        f"{e.get('id', '')}-{e.get('photoUrl', '')}"
        for e in employees
    ])
    return hashlib.md5(data.encode()).hexdigest()


def save_cache(
    encodings: List,
    ids: List[int],
    emp_hash: str,
    cache_file: str
) -> None:
    """
    Save embeddings cache to file.
    
    Args:
        encodings: List of face embeddings
        ids: List of employee IDs
        emp_hash: Hash of employee list
        cache_file: Path to cache file
    """
    try:
        cache_data = {
            'encodings': encodings,
            'ids': ids,
            'hash': emp_hash,
            'timestamp': time.time(),
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f'Cache saved for {len(ids)} employees')
        
    except Exception as e:
        logger.error(f'Failed to save cache: {e}')


def load_cache(
    cache_file: str
) -> Tuple[Optional[List], Optional[List[int]], Optional[str]]:
    """
    Load embeddings cache from file.
    
    Args:
        cache_file: Path to cache file
    
    Returns:
        Tuple of (encodings, ids, hash) or (None, None, None) if cache invalid
    """
    if not os.path.exists(cache_file):
        logger.debug('Cache file not found')
        return None, None, None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        age = time.time() - cache_data.get('timestamp', 0)
        logger.info(f'Cache found (age: {age:.0f} seconds)')
        
        return (
            cache_data.get('encodings'),
            cache_data.get('ids'),
            cache_data.get('hash')
        )
        
    except Exception as e:
        logger.error(f'Failed to load cache: {e}')
        return None, None, None









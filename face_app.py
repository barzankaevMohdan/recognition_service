"""
InsightFace initialization module.

Provides face detection and recognition using InsightFace models.
"""

from insightface.app import FaceAnalysis
from .config import Config
from .logging_config import get_logger

logger = get_logger(__name__)


def initialize_face_app(config: Config) -> FaceAnalysis:
    """
    Initialize InsightFace FaceAnalysis.
    
    Args:
        config: Service configuration
    
    Returns:
        Initialized FaceAnalysis instance
    """
    logger.info('Initializing InsightFace AI...')
    
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=config.insightface_det_size)
    
    logger.info(f'✅ InsightFace initialized (det_size={config.insightface_det_size})')
    
    return face_app



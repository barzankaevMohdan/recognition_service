"""
Configuration module for Recognition Service.

Loads configuration from environment variables with sensible defaults.
All settings are immutable after initialization.
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration for Recognition Service.
    
    Backend Integration:
        backend_url: Base URL of the backend API (e.g., http://backend:3000)
    
    Camera Settings:
        camera_source: Camera source - can be:
            - Integer (0, 1, 2) for local webcam
            - RTSP URL: rtsp://user:pass@ip:port/path
            - HTTP URL: http://camera-gateway:4000/streams/1.mjpg
        camera_id: Logical identifier for this camera (for logging/monitoring)
        frame_skip: Process every N-th frame (higher = faster, less accurate)
    
    Service Identity:
        service_name: Name of this service instance
        video_port: Port for Flask HTTP server
    
    Quality Thresholds:
        min_face_height_pixels: Minimum face height in pixels to process
        min_blur_variance: Minimum Laplacian variance (higher = sharper required)
    
    Preprocessing:
        enable_preprocessing: Enable image enhancement pipeline
        clahe_clip_limit: CLAHE contrast limiting (higher = more contrast)
        denoise_strength: Denoising strength (0-10, higher = more smoothing)
    
    Recognition:
        insightface_threshold: Cosine similarity threshold (lower = stricter)
        insightface_det_size: Detection size for InsightFace (width, height)
    
    Tracking:
        min_embeddings_per_track: Minimum embeddings before recognition attempt
        track_max_age_seconds: Maximum age of track without updates
        iou_threshold: IoU threshold for bbox matching
    
    Presence Logic:
        in_threshold_seconds: Stable presence time before IN event
        out_threshold_seconds: Absence time before OUT event
    
    System:
        reload_employees_interval: Seconds between employee list reloads
        cache_file: Path to embeddings cache file
        debug_mode: Enable debug logging
    """
    
    # Backend
    backend_url: str
    
    # Camera
    camera_source: str
    camera_id: str
    frame_skip: int
    
    # Service
    service_name: str
    video_port: int
    
    # Quality
    min_face_height_pixels: int
    min_blur_variance: float
    
    # Preprocessing
    enable_preprocessing: bool
    clahe_clip_limit: float
    denoise_strength: int
    
    # InsightFace
    insightface_threshold: float
    insightface_det_size: Tuple[int, int]
    
    # Tracking
    min_embeddings_per_track: int
    track_max_age_seconds: float
    iou_threshold: float
    
    # Presence
    in_threshold_seconds: float
    out_threshold_seconds: float
    
    # System
    reload_employees_interval: int
    cache_file: str
    debug_mode: bool


def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Config: Immutable configuration object
    """
    # Helper to parse camera source
    camera_source_raw = os.getenv('CAMERA_SOURCE', '0')
    
    return Config(
        # Backend
        backend_url=os.getenv('BACKEND_URL', 'http://localhost:3000'),
        
        # Camera
        camera_source=camera_source_raw,
        camera_id=os.getenv('CAMERA_ID', camera_source_raw),
        frame_skip=int(os.getenv('FRAME_SKIP', '3')),
        
        # Service
        service_name=os.getenv('SERVICE_NAME', 'recognition'),
        video_port=int(os.getenv('VIDEO_PORT', '5001')),
        
        # Quality
        min_face_height_pixels=int(os.getenv('MIN_FACE_HEIGHT', '20')),
        min_blur_variance=float(os.getenv('MIN_BLUR_VAR', '50.0')),
        
        # Preprocessing
        enable_preprocessing=os.getenv('ENABLE_PREPROCESSING', 'true').lower() == 'true',
        clahe_clip_limit=float(os.getenv('CLAHE_CLIP', '2.0')),
        denoise_strength=int(os.getenv('DENOISE_STRENGTH', '5')),
        
        # InsightFace
        insightface_threshold=float(os.getenv('INSIGHTFACE_THRESHOLD', '0.2')),
        insightface_det_size=(640, 640),
        
        # Tracking
        min_embeddings_per_track=int(os.getenv('MIN_EMBEDDINGS', '2')),
        track_max_age_seconds=float(os.getenv('TRACK_MAX_AGE', '2.0')),
        iou_threshold=0.3,
        
        # Presence
        in_threshold_seconds=float(os.getenv('IN_THRESHOLD', '1.0')),
        out_threshold_seconds=float(os.getenv('OUT_THRESHOLD', '10.0')),
        
        # System
        reload_employees_interval=int(os.getenv('RELOAD_INTERVAL', '300')),
        cache_file=os.getenv('CACHE_FILE', 'face_encodings_cache.pkl'),
        debug_mode=os.getenv('DEBUG', 'true').lower() == 'true',
    )









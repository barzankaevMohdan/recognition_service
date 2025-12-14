"""
Camera connection and management module.

Handles connection to various camera sources:
- Local webcams (index 0, 1, 2)
- RTSP streams
- HTTP streams (from camera gateway)

Includes reconnection logic and low-latency optimizations for RTSP.
"""

import time
import cv2
import numpy as np
import requests
from typing import Optional
from .config import Config
from .logging_config import get_logger

logger = get_logger(__name__)


def connect_camera(config: Config, max_retries: int = 5) -> cv2.VideoCapture:
    """
    Connect to camera with retry logic.
    
    Args:
        config: Service configuration
        max_retries: Maximum connection attempts
    
    Returns:
        Opened VideoCapture object
    
    Raises:
        RuntimeError: If connection fails after max_retries
    """
    camera_source = config.camera_source
    
    # Determine camera type
    if camera_source.isdigit():
        camera_type = 'local'
        camera_index = int(camera_source)
        source = camera_index
    elif camera_source.startswith('rtsp://') or camera_source.startswith('http://'):
        camera_type = 'stream'
        source = camera_source
    else:
        # Try as local camera index
        try:
            camera_index = int(camera_source)
            camera_type = 'local'
            source = camera_index
        except ValueError:
            camera_type = 'stream'
            source = camera_source
    
    # Attempt connection with retries
    for attempt in range(max_retries):
        logger.info(f'Connecting to {camera_type} camera (attempt {attempt + 1}/{max_retries})...')
        
        if camera_type == 'local':
            logger.info(f'Camera index: {source}')
            video_capture = cv2.VideoCapture(source)
        else:
            logger.info(f'Camera URL: {_sanitize_url(source)}')
            video_capture = _open_stream_capture(source)
            
            # Low-latency settings for RTSP
            if video_capture and camera_source.startswith('rtsp://'):
                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify connection
        if video_capture and video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret and frame is not None:
                logger.info(f'✅ Camera connected ({camera_type})')
                logger.info(f'Frame size: {frame.shape[1]}x{frame.shape[0]}')
                
                # Flush initial buffer for RTSP
                if camera_source.startswith('rtsp://'):
                    logger.info('Flushing initial buffer for low latency...')
                    for _ in range(5):
                        video_capture.grab()
                    logger.info('Low latency mode active')
                
                return video_capture
            else:
                video_capture.release()
                logger.warning('Camera opened but failed to read frame')
        else:
            logger.warning('Failed to open camera')
            if video_capture is None:
                logger.debug('VideoCapture constructor returned None (unsupported backend?)')
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logger.info(f'Retrying in {wait_time} seconds...')
            time.sleep(wait_time)
    
    # All attempts failed
    raise RuntimeError(f'Cannot connect to camera after {max_retries} attempts')


def reconnect_camera(
    video_capture: cv2.VideoCapture,
    config: Config,
    consecutive_failures: int
) -> tuple[cv2.VideoCapture, int]:
    """
    Reconnect to camera after failures.
    
    Args:
        video_capture: Current VideoCapture (will be released)
        config: Service configuration
        consecutive_failures: Number of consecutive failures
    
    Returns:
        Tuple of (new VideoCapture, reset failure count)
    """
    logger.error(f'Too many failures ({consecutive_failures}), reconnecting...')
    
    # Release old capture
    try:
        video_capture.release()
    except Exception as e:
        logger.warning(f'Error releasing camera: {e}')
    
    # Wait before reconnect
    time.sleep(2)
    
    # Reconnect
    try:
        new_capture = connect_camera(config)
        return new_capture, 0
    except RuntimeError as e:
        logger.error(f'Reconnection failed: {e}')
        raise


def is_rtsp_stream(camera_source: str) -> bool:
    """
    Check if camera source is RTSP stream.
    
    Args:
        camera_source: Camera source string
    
    Returns:
        True if RTSP stream
    """
    return camera_source.startswith('rtsp://')


def minimize_latency_for_rtsp(video_capture: cv2.VideoCapture) -> None:
    """
    Apply low-latency optimizations for RTSP streams.
    
    Args:
        video_capture: VideoCapture object
    """
    # Grab frame without decoding to skip buffered frames
    video_capture.grab()


def _open_stream_capture(source: str) -> Optional[cv2.VideoCapture]:
    """
    Try multiple OpenCV backends to open HTTP/RTSP streams.
    For HTTP MJPEG streams, use MJPEGStreamCapture for better reliability.
    """
    # For HTTP MJPEG streams, use custom reader
    if source.startswith('http://') or source.startswith('https://'):
        if '.mjpg' in source or 'mjpeg' in source.lower():
            logger.debug('Detected MJPEG stream, using HTTP reader')
            return MJPEGStreamCapture(source)
    
    # For RTSP and other streams, try OpenCV backends
    backend_candidates = []
    if hasattr(cv2, 'CAP_FFMPEG'):
        backend_candidates.append(('CAP_FFMPEG', cv2.CAP_FFMPEG))
    backend_candidates.append(('CAP_ANY', getattr(cv2, 'CAP_ANY', None)))
    backend_candidates.append(('DEFAULT', None))
    
    for backend_name, backend_flag in backend_candidates:
        try:
            capture = cv2.VideoCapture(source) if backend_flag is None else cv2.VideoCapture(source, backend_flag)
        except Exception as exc:
            logger.debug(f'Backend {backend_name} failed: {exc}')
            capture = None
        
        if capture is None:
            continue
        
        if capture.isOpened():
            logger.debug(f'Stream opened with backend {backend_name}')
            return capture
        capture.release()
    
    return None


def _sanitize_url(url: str) -> str:
    """
    Remove password from URL for logging.
    
    Args:
        url: URL with potential password
    
    Returns:
        Sanitized URL
    """
    if '://' not in url:
        return url
    
    try:
        # Split protocol and rest
        protocol, rest = url.split('://', 1)
        
        # Check for credentials
        if '@' in rest:
            # Split credentials and host
            creds, host = rest.rsplit('@', 1)
            
            # Split username and password
            if ':' in creds:
                username = creds.split(':', 1)[0]
                return f'{protocol}://{username}@{host}'
        
        return url
    except Exception:
        return url


class MJPEGStreamCapture:
    """
    Custom VideoCapture for HTTP MJPEG streams.
    Uses requests to read stream and decode frames manually.
    More reliable than cv2.VideoCapture for HTTP streams on Windows.
    """
    
    def __init__(self, url: str, timeout: int = 10):
        """
        Initialize MJPEG stream reader.
        
        Args:
            url: HTTP URL of MJPEG stream
            timeout: Request timeout in seconds
        """
        self.url = url
        self.timeout = timeout
        self._opened = False
        self._stream = None
        self._response = None
        
        # Try to open stream
        try:
            logger.debug(f'Opening MJPEG stream: {url}')
            self._response = requests.get(url, stream=True, timeout=timeout)
            if self._response.status_code == 200:
                self._stream = self._response.iter_content(chunk_size=1024)
                self._opened = True
                self._buffer = b''
                logger.debug('MJPEG stream opened successfully')
            else:
                logger.warning(f'MJPEG stream returned status {self._response.status_code}')
        except Exception as e:
            logger.warning(f'Failed to open MJPEG stream: {e}')
            self._opened = False
    
    def isOpened(self) -> bool:
        """Check if stream is open."""
        return self._opened
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from MJPEG stream.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self._opened or self._stream is None:
            return False, None
        
        try:
            # Read until we find a complete JPEG frame
            while True:
                # Read chunk
                chunk = next(self._stream, None)
                if chunk is None:
                    return False, None
                
                self._buffer += chunk
                
                # Look for JPEG markers
                start = self._buffer.find(b'\xff\xd8')  # JPEG start
                end = self._buffer.find(b'\xff\xd9')    # JPEG end
                
                if start != -1 and end != -1 and end > start:
                    # Extract JPEG frame
                    jpg = self._buffer[start:end+2]
                    self._buffer = self._buffer[end+2:]
                    
                    # Decode JPEG to frame
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
                
                # Prevent buffer overflow
                if len(self._buffer) > 10 * 1024 * 1024:  # 10MB limit
                    logger.warning('Buffer overflow, resetting')
                    self._buffer = b''
        
        except Exception as e:
            logger.warning(f'Error reading MJPEG frame: {e}')
            return False, None
    
    def release(self) -> None:
        """Release stream resources."""
        self._opened = False
        if self._response:
            try:
                self._response.close()
            except Exception:
                pass
        self._stream = None
        self._buffer = b''
    
    def set(self, prop_id: int, value: float) -> bool:
        """Compatibility method (does nothing for MJPEG streams)."""
        return True
    
    def grab(self) -> bool:
        """Compatibility method for low-latency flush."""
        return True







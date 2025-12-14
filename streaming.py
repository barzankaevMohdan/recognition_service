"""
Video streaming module.

Manages current frame state and MJPEG stream generation for Flask.
Thread-safe frame access using locks.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Generator, Dict
import numpy as np
import cv2


DEFAULT_STREAM_ID = "default"


@dataclass
class _StreamState:
    frame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_streams: Dict[str, _StreamState] = {}
_streams_lock = threading.Lock()


def _get_stream_state(stream_id: str) -> _StreamState:
    """
    Return/create stream state for given identifier.
    """
    state = _streams.get(stream_id)
    if state is None:
        with _streams_lock:
            state = _streams.get(stream_id)
            if state is None:
                state = _StreamState()
                _streams[stream_id] = state
    return state


def set_frame(frame: np.ndarray, stream_id: str = DEFAULT_STREAM_ID) -> None:
    """
    Update current frame for a specific stream (thread-safe).
    
    Args:
        frame: New frame to set
        stream_id: Identifier of the stream (camera/service)
    """
    state = _get_stream_state(stream_id)
    with state.lock:
        state.frame = frame.copy() if frame is not None else None


def get_frame_copy(stream_id: str = DEFAULT_STREAM_ID) -> Optional[np.ndarray]:
    """
    Get a copy of current frame for specific stream (thread-safe).
    
    Returns:
        Copy of current frame or None
    """
    state = _get_stream_state(stream_id)
    with state.lock:
        return state.frame.copy() if state.frame is not None else None


def is_streaming(stream_id: str = DEFAULT_STREAM_ID) -> bool:
    """
    Check if streaming is active for a stream.
    
    Returns:
        True if current frame exists
    """
    state = _get_stream_state(stream_id)
    with state.lock:
        return state.frame is not None


def generate_mjpeg_frames(stream_id: str = DEFAULT_STREAM_ID) -> Generator[bytes, None, None]:
    """
    Generate MJPEG frames for a specific stream.
    
    Yields:
        JPEG frame bytes with multipart headers
    """
    while True:
        frame = get_frame_copy(stream_id)
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if not ret:
            time.sleep(0.033)
            continue
        
        # Yield frame with multipart headers
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # ~30 FPS
        time.sleep(0.033)









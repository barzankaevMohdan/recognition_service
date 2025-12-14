"""
Main video processing loop.

Orchestrates the entire recognition pipeline:
- Camera connection
- Frame processing
- Face detection and tracking
- Presence management
- Event sending
"""

import time
import threading
import cv2
from typing import Any, List
from .config import Config
from .logging_config import get_logger
from .camera import connect_camera, reconnect_camera, is_rtsp_stream, minimize_latency_for_rtsp
from .employees import load_employees_from_backend
from .events import send_event
from .streaming import set_frame
from .recognition.tracker import FaceTracker
from .recognition.presence import PresenceManager
from .app import create_app

logger = get_logger(__name__)


def start_flask_server(config: Config) -> None:
    """
    Start Flask server in background thread.
    
    Args:
        config: Service configuration
    """
    logger.info(f'Starting video stream server on port {config.video_port}...')
    app = create_app(config)
    app.run(
        host='0.0.0.0',
        port=config.video_port,
        threaded=True,
        debug=False,
        use_reloader=False
    )


def run(face_app: Any, config: Config, stop_flag: threading.Event = None) -> None:
    """
    Main video processing loop.
    
    Args:
        face_app: InsightFace FaceAnalysis instance
        config: Service configuration
        stop_flag: Optional threading.Event to signal graceful shutdown
    """
    stream_id = config.camera_id or config.service_name or 'default'
    
    # Load employees
    known_embeddings, known_ids = load_employees_from_backend(config, face_app)
    
    if not known_ids:
        logger.error('No employees with photos found!')
        logger.error('Please add employees via backend API before starting recognition')
        return
    
    # Initialize managers
    tracker = FaceTracker(config)
    presence_manager = PresenceManager(known_ids, config)
    
    # Start Flask server in background
    flask_thread = threading.Thread(target=start_flask_server, args=(config,), daemon=True)
    flask_thread.start()
    logger.info(f'Video stream: http://localhost:{config.video_port}/video_feed')
    
    # Connect to camera
    video_capture = connect_camera(config)
    
    # Loop state
    frame_count = 0
    consecutive_failures = 0
    MAX_FAILURES = 10
    last_reload = time.time()
    
    logger.info('🎬 Starting main loop...')
    
    try:
        while True:
            # Check for stop signal
            if stop_flag and stop_flag.is_set():
                logger.info('Stop signal received, exiting gracefully...')
                break
            
            # Minimize latency for RTSP
            if is_rtsp_stream(config.camera_source) and frame_count % 2 == 0:
                minimize_latency_for_rtsp(video_capture)
            
            # Read frame
            ret, frame = video_capture.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(f'Failed to read frame ({consecutive_failures}/{MAX_FAILURES})')
                
                if consecutive_failures >= MAX_FAILURES:
                    video_capture, consecutive_failures = reconnect_camera(
                        video_capture, config, consecutive_failures
                    )
                else:
                    time.sleep(0.5)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Hot reload employees
            if time.time() - last_reload > config.reload_employees_interval:
                logger.info('Reloading employees...')
                try:
                    new_embeddings, new_ids = load_employees_from_backend(config, face_app)
                    if new_ids:
                        known_embeddings, known_ids = new_embeddings, new_ids
                        
                        # Add new employees to presence manager
                        for emp_id in new_ids:
                            presence_manager.add_employee(emp_id)
                        
                        logger.info(f'Reloaded {len(new_ids)} employees')
                    last_reload = time.time()
                except Exception as e:
                    logger.error(f'Employee reload failed: {e}')
                    last_reload = time.time()
            
            # Process only every N-th frame
            if frame_count % config.frame_skip != 0:
                set_frame(frame, stream_id=stream_id)
                continue
            
            # Detect faces
            faces = face_app.get(frame)
            
            # Update tracks (pass frame for face cropping)
            recognized_tracks = tracker.update(faces, frame, known_embeddings, known_ids)
            
            # Get recognized employee IDs
            recognized_emp_ids = [
                t.recognized_employee_id
                for t in recognized_tracks
                if t.recognized_employee_id is not None
            ]
            
            # Update presence and get events
            events = presence_manager.update(recognized_emp_ids)
            
            # Send events to backend
            for emp_id, event_type in events:
                send_event(emp_id, event_type, config)
            
            # Visualize
            display_frame = _draw_visualization(
                frame.copy(),
                tracker,
                recognized_emp_ids,
                config
            )
            
            # Update streaming frame
            set_frame(display_frame, stream_id=stream_id)
            
            # Small delay
            time.sleep(0.05)
    
    finally:
        video_capture.release()
        logger.info('Camera released')


def _draw_visualization(
    frame,
    tracker: FaceTracker,
    recognized_emp_ids: List[int],
    config: Config
):
    """
    Draw visualization on frame.
    
    Args:
        frame: Frame to draw on
        tracker: FaceTracker instance
        recognized_emp_ids: List of recognized employee IDs
        config: Service configuration
    
    Returns:
        Frame with visualization
    """
    # Status text
    preprocessing_status = "CLAHE→Sharp→" if config.enable_preprocessing else ""
    status_text = (
        f"{preprocessing_status}InsightFace | "
        f"Tracks: {len(tracker.tracks)} | "
        f"Recognized: {len(recognized_emp_ids)}"
    )
    
    # Draw status (with shadow)
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw recognized faces (green)
    for track in tracker.tracks:
        if track.recognized_employee_id and track.last_bbox is not None:
            x1, y1, x2, y2 = track.last_bbox.astype(int)
            
            # Green rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Label
            label = f"ID: {track.recognized_employee_id} ({track.recognition_confidence:.0%})"
            cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y2 - 8),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw unrecognized faces (red)
    for track in tracker.tracks:
        if not track.recognized_employee_id and track.last_bbox is not None:
            x1, y1, x2, y2 = track.last_bbox.astype(int)
            
            # Red rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Label
            label = f"Track {track.track_id} ({len(track.embeddings)}/{config.min_embeddings_per_track})"
            cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y2 - 8),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


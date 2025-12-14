"""
Multi-camera manager for handling multiple camera streams for a company.

This module manages multiple camera threads, automatically syncing with
the backend to add/remove cameras as they are configured.
"""

import threading
import time
import requests
from typing import List, Dict, Optional
from .config import Config, load_config
from .logging_config import get_logger

logger = get_logger(__name__)


class CameraThread:
    """Represents a single camera processing thread."""
    
    def __init__(self, camera_id: int, camera_data: Dict, backend_url: str, company_slug: str):
        self.camera_id = camera_id
        self.camera_data = camera_data
        self.backend_url = backend_url
        self.company_slug = company_slug
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.config: Optional[Config] = None
    
    def start(self):
        """Start the camera processing thread."""
        if self.thread and self.thread.is_alive():
            logger.debug(f"Camera {self.camera_id} already running")
            return
        
        logger.info(f"Starting camera {self.camera_id} ({self.camera_data['name']})")
        
        # Load base config from environment/defaults
        base_config = load_config()
        
        # Create config dict with overrides for this camera
        config_dict = base_config.__dict__.copy()
        config_dict['camera_source'] = self.camera_data['streamUrl']
        config_dict['camera_id'] = str(self.camera_id)
        config_dict['backend_url'] = self.backend_url
        config_dict['service_name'] = f"{self.company_slug}-camera-{self.camera_id}"
        config_dict['video_port'] = 5000 + self.camera_id  # Dynamic port per camera
        
        # Create new config instance
        self.config = Config(**config_dict)
        
        # Start thread
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"Camera-{self.camera_id}"
        )
        self.thread.start()
    
    def _run(self):
        """Run the camera processing loop."""
        try:
            # Import here to avoid circular imports
            from .video_loop import run as run_video_loop
            from .face_app import initialize_face_app
            
            # Initialize InsightFace
            logger.info(f"[camera={self.camera_id}] Initializing InsightFace AI...")
            face_app = initialize_face_app(self.config)
            
            # Run video loop
            run_video_loop(face_app, self.config, self.stop_flag)
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id} crashed: {e}", exc_info=True)
    
    def stop(self):
        """Stop the camera processing thread."""
        logger.info(f"Stopping camera {self.camera_id}")
        self.stop_flag.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
    
    def is_alive(self) -> bool:
        """Check if the thread is alive."""
        return self.thread is not None and self.thread.is_alive()


class MultiCameraManager:
    """Manages multiple camera streams for a company."""
    
    def __init__(self, company_slug: str, backend_url: str, refresh_interval: int = 60):
        """
        Initialize the multi-camera manager.
        
        Args:
            company_slug: Company slug to manage cameras for
            backend_url: Backend API URL
            refresh_interval: Interval to refresh camera list (seconds)
        """
        self.company_slug = company_slug
        self.backend_url = backend_url
        self.refresh_interval = refresh_interval
        self.camera_threads: Dict[int, CameraThread] = {}
        self.running = True
        
        logger.info(f"Initialized MultiCameraManager for company: {company_slug}")
    
    def get_company_cameras(self) -> List[Dict]:
        """
        Fetch active cameras for the company from backend.
        
        Returns:
            List of camera dictionaries with id, name, location, streamUrl
        """
        try:
            url = f"{self.backend_url}/api/cameras/public/{self.company_slug}/cameras"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            cameras = response.json()
            logger.info(f"Fetched {len(cameras)} cameras for company {self.company_slug}")
            return cameras
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch cameras: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching cameras: {e}")
            return []
    
    def start_camera(self, camera: Dict):
        """
        Start a camera thread.
        
        Args:
            camera: Camera dictionary with id, name, location, streamUrl
        """
        camera_id = camera['id']
        
        if camera_id in self.camera_threads:
            camera_thread = self.camera_threads[camera_id]
            if camera_thread.is_alive():
                logger.debug(f"Camera {camera_id} already running")
                return
            else:
                # Thread died, remove it
                logger.warning(f"Camera {camera_id} thread died, restarting")
                del self.camera_threads[camera_id]
        
        # Create and start new camera thread
        camera_thread = CameraThread(
            camera_id=camera_id,
            camera_data=camera,
            backend_url=self.backend_url,
            company_slug=self.company_slug
        )
        camera_thread.start()
        self.camera_threads[camera_id] = camera_thread
    
    def stop_camera(self, camera_id: int):
        """
        Stop a camera thread.
        
        Args:
            camera_id: Camera ID to stop
        """
        if camera_id not in self.camera_threads:
            return
        
        camera_thread = self.camera_threads[camera_id]
        camera_thread.stop()
        del self.camera_threads[camera_id]
    
    def sync_cameras(self):
        """Synchronize running cameras with backend."""
        current_cameras = self.get_company_cameras()
        current_ids = {cam['id'] for cam in current_cameras}
        running_ids = set(self.camera_threads.keys())
        
        # Stop removed cameras
        for camera_id in running_ids - current_ids:
            logger.info(f"Camera {camera_id} removed from backend, stopping")
            self.stop_camera(camera_id)
        
        # Start new cameras
        for camera in current_cameras:
            self.start_camera(camera)
        
        # Log status
        logger.info(f"Cameras running: {len(self.camera_threads)}/{len(current_cameras)}")
    
    def run(self):
        """Main loop: sync cameras periodically."""
        logger.info(f"Starting multi-camera manager for company: {self.company_slug}")
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Refresh interval: {self.refresh_interval}s")
        
        while self.running:
            try:
                self.sync_cameras()
                
                # Wait for next sync or until stopped
                for _ in range(self.refresh_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in sync loop: {e}", exc_info=True)
                time.sleep(10)
        
        # Cleanup
        logger.info("Stopping all cameras...")
        for camera_id in list(self.camera_threads.keys()):
            self.stop_camera(camera_id)
        
        logger.info("Multi-camera manager stopped")
    
    def stop(self):
        """Stop the manager and all camera threads."""
        self.running = False


"""
Logging configuration for Recognition Service.

Provides structured logging with camera ID context.
"""

import logging
import sys
from typing import Optional


class CameraContextFilter(logging.Filter):
    """Add camera context to log records."""
    
    def __init__(self, camera_id: str):
        super().__init__()
        self.camera_id = camera_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.camera_id = self.camera_id
        return True


def setup_logging(camera_id: str, debug: bool = False) -> None:
    """
    Configure logging for the service.
    
    Args:
        camera_id: Camera identifier for log context
        debug: Enable debug level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter with camera context
    formatter = logging.Formatter(
        '[%(levelname)s] [camera=%(camera_id)s] %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add camera context filter
    console_handler.addFilter(CameraContextFilter(camera_id))
    
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)









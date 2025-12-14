"""
Timing utilities.

Helper functions for time-related operations.
"""

import time
from typing import Callable, Any, TypeVar

T = TypeVar('T')


def format_uptime(seconds: float) -> str:
    """
    Format uptime in human-readable format.
    
    Args:
        seconds: Uptime in seconds
    
    Returns:
        Formatted string (e.g., "1d 2h 30m 45s")
    """
    seconds = int(seconds)
    
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if days > 0:
        parts.append(f'{days}d')
    if hours > 0:
        parts.append(f'{hours}h')
    if minutes > 0:
        parts.append(f'{minutes}m')
    parts.append(f'{secs}s')
    
    return ' '.join(parts)


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    """
    delay = initial_delay
    last_exception: Exception | None = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= backoff_factor
    
    if last_exception:
        raise last_exception
    
    raise RuntimeError('Retry failed with no exception')

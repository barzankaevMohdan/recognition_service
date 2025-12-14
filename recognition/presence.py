"""
Presence management module.

Manages employee presence state and generates IN/OUT events based on:
- Stable presence time (IN threshold)
- Absence time (OUT threshold)
"""

import time
from typing import List, Tuple, Dict
from ..config import Config
from ..logging_config import get_logger

logger = get_logger(__name__)


class PresenceManager:
    """
    Manages presence state for all employees.
    
    Tracks when employees are present/absent and generates
    IN/OUT events based on configured thresholds.
    """
    
    def __init__(self, employee_ids: List[int], config: Config):
        """
        Initialize presence manager.
        
        Args:
            employee_ids: List of known employee IDs
            config: Service configuration
        """
        self.config = config
        self.state: Dict[int, Dict] = {}
        
        for emp_id in employee_ids:
            self.state[emp_id] = {
                'present': False,
                'last_seen': 0.0,
                'last_state_change': 0.0,
            }
    
    def update(self, recognized_employee_ids: List[int]) -> List[Tuple[int, str]]:
        """
        Update presence states and generate events.
        
        Args:
            recognized_employee_ids: List of currently recognized employee IDs
        
        Returns:
            List of events as tuples (employee_id, event_type)
            where event_type is 'IN' or 'OUT'
        """
        now = time.time()
        events: List[Tuple[int, str]] = []
        
        # Update last_seen for recognized employees
        for emp_id in recognized_employee_ids:
            if emp_id in self.state:
                self.state[emp_id]['last_seen'] = now
        
        # Check each employee's state
        for emp_id, state in self.state.items():
            if emp_id in recognized_employee_ids:
                # Employee is visible
                if not state['present']:
                    # Was absent, check if should mark as present
                    time_since_change = now - state['last_state_change']
                    
                    if time_since_change > self.config.in_threshold_seconds:
                        # Stable presence - generate IN event
                        state['present'] = True
                        state['last_state_change'] = now
                        events.append((emp_id, 'IN'))
                        logger.info(
                            f'✅ Employee {emp_id} marked as IN '
                            f'(stable presence {time_since_change:.1f}s)'
                        )
            else:
                # Employee not visible
                if state['present']:
                    # Was present, check if should mark as absent
                    time_since_seen = now - state['last_seen']
                    
                    if time_since_seen > self.config.out_threshold_seconds:
                        # Long absence - generate OUT event
                        state['present'] = False
                        state['last_state_change'] = now
                        events.append((emp_id, 'OUT'))
                        logger.info(
                            f'✅ Employee {emp_id} marked as OUT '
                            f'(absent {time_since_seen:.1f}s)'
                        )
        
        return events
    
    def add_employee(self, emp_id: int) -> None:
        """
        Add new employee to tracking.
        
        Args:
            emp_id: Employee ID
        """
        if emp_id not in self.state:
            self.state[emp_id] = {
                'present': False,
                'last_seen': 0.0,
                'last_state_change': 0.0,
            }
            logger.debug(f'Added employee {emp_id} to presence tracking')









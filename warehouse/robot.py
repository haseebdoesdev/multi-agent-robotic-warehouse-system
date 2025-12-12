from typing import Tuple, List, Optional, Set
from .environment import Warehouse
from .pathfinding import a_star


class Robot:
    
    STATUS_IDLE = "idle"
    STATUS_MOVING = "moving"
    STATUS_COLLECTING = "collecting"
    STATUS_COMPLETED = "completed"
    STATUS_WAITING = "waiting"
    
    def __init__(self, robot_id: int, start_position: Tuple[int, int]):
        self.id = robot_id
        self.position = start_position
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.target_package: Optional[Tuple[int, int]] = None
        self.status = self.STATUS_IDLE
        self.packages_collected = 0
        self.total_distance_traveled = 0
        self.wait_counter = 0
        self.replan_count = 0  # Track how many times robot had to replan
    
    def plan_path(self, warehouse: Warehouse, target: Optional[Tuple[int, int]] = None,
                  occupied_cells: Optional[Set[Tuple[int, int]]] = None,
                  is_replan: bool = False) -> bool:
        """
        Plan a path to the target using A* pathfinding.
        
        Args:
            warehouse: The warehouse environment
            target: Target position (defaults to self.target_package)
            occupied_cells: Set of cells occupied by other robots to avoid
            is_replan: Whether this is a replan (for tracking metrics)
        
        Returns:
            True if a valid path was found, False otherwise
        """
        if target is None:
            target = self.target_package
        
        if target is None:
            return False
        
        path = a_star(self.position, target, warehouse, occupied_cells=occupied_cells)
        
        if path is not None:
            self.path = path
            self.path_index = 0
            self.status = self.STATUS_MOVING
            if is_replan:
                self.replan_count += 1
            return True
        else:
            # If planning with occupied cells failed, try without them as fallback
            if occupied_cells:
                path = a_star(self.position, target, warehouse)
                if path is not None:
                    self.path = path
                    self.path_index = 0
                    self.status = self.STATUS_MOVING
                    if is_replan:
                        self.replan_count += 1
                    return True
            self.status = self.STATUS_IDLE
            return False
    
    def set_target(self, target: Tuple[int, int]):
        self.target_package = target
    
    def move(self, warehouse: Warehouse, 
             occupied_cells: Optional[Set[Tuple[int, int]]] = None) -> bool:
        """
        Move the robot one step along its path.
        
        Args:
            warehouse: The warehouse environment
            occupied_cells: Set of cells occupied by other robots (for replanning)
        
        Returns:
            True if robot moved, False otherwise
        """
        if self.status == self.STATUS_WAITING:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.status = self.STATUS_MOVING
                # Validate path after waiting - next step might be blocked now
                if self.path_index + 1 < len(self.path):
                    next_pos = self.path[self.path_index + 1]
                    if not warehouse.is_valid_move(next_pos[0], next_pos[1], ignore_packages=True):
                        # Path blocked, trigger replan
                        self.plan_path(warehouse, occupied_cells=occupied_cells, is_replan=True)
            return False
        
        if not self.path or self.path_index >= len(self.path):
            self.status = self.STATUS_IDLE
            return False
        
        if self.path_index + 1 < len(self.path):
            next_position = self.path[self.path_index + 1]
            
            if warehouse.is_valid_move(next_position[0], next_position[1], ignore_packages=True):
                self.position = next_position
                self.path_index += 1
                self.total_distance_traveled += 1
                self.status = self.STATUS_MOVING
                return True
            else:
                # Dynamic replanning: path is blocked, try to find alternative
                if self.target_package is not None:
                    if self.plan_path(warehouse, occupied_cells=occupied_cells, is_replan=True):
                        # Successfully replanned, try to move on new path
                        if self.path_index + 1 < len(self.path):
                            new_next = self.path[self.path_index + 1]
                            if warehouse.is_valid_move(new_next[0], new_next[1], ignore_packages=True):
                                self.position = new_next
                                self.path_index += 1
                                self.total_distance_traveled += 1
                                return True
                self.status = self.STATUS_IDLE
                return False
        else:
            self.status = self.STATUS_IDLE
            return False
    
    def has_reached_target(self) -> bool:
        if self.target_package is None:
            return False
        
        return self.position == self.target_package
    
    def collect_package(self, warehouse: Warehouse) -> bool:
        if self.has_reached_target():
            warehouse.remove_package(self.position[0], self.position[1])
            self.packages_collected += 1
            self.target_package = None
            self.path = []
            self.path_index = 0
            self.status = self.STATUS_COMPLETED
            return True
        return False
    
    def wait(self, duration: int = 1):
        self.status = self.STATUS_WAITING
        self.wait_counter = duration
    
    def is_idle(self) -> bool:
        return self.status in [self.STATUS_IDLE, self.STATUS_COMPLETED]
    
    def is_moving(self) -> bool:
        return self.status == self.STATUS_MOVING
    
    def get_next_position(self) -> Optional[Tuple[int, int]]:
        if self.path and self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return None
    
    def get_remaining_path_length(self) -> int:
        """Get the number of steps remaining to reach target."""
        if not self.path:
            return float('inf')
        return len(self.path) - self.path_index - 1
    
    def invalidate_path(self):
        """Clear the current path, forcing a replan on next update."""
        self.path = []
        self.path_index = 0
    
    def __str__(self) -> str:
        return f"Robot {self.id} at {self.position} ({self.status})"
    
    def __repr__(self) -> str:
        return (f"Robot(id={self.id}, pos={self.position}, status={self.status}, "
                f"collected={self.packages_collected}, distance={self.total_distance_traveled})")

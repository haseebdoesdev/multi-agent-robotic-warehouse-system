from typing import Tuple, List, Optional
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
    
    # Module 4 TODO:
    # - When UNCERTAINTY is enabled, consider calling a sensor update here or
    #   during move() to feed observations into a probabilistic obstacle map.
    # - Add a light-weight hook to trigger replanning if predicted blockage risk
    #   along the current path exceeds a threshold from config.
    def plan_path(self, warehouse: Warehouse, target: Optional[Tuple[int, int]] = None) -> bool:
        if target is None:
            target = self.target_package
        
        if target is None:
            return False
        
        path = a_star(self.position, target, warehouse)
        
        if path is not None:
            self.path = path
            self.path_index = 0
            self.status = self.STATUS_MOVING
            return True
        else:
            self.status = self.STATUS_IDLE
            return False
    
    def set_target(self, target: Tuple[int, int]):
        self.target_package = target
    
    def move(self, warehouse: Warehouse) -> bool:
        if self.status == self.STATUS_WAITING:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.status = self.STATUS_MOVING
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
                # Module 4 TODO:
                # - Dynamic replanning: when the next step is blocked due to a new obstacle
                #   or committed uncertainty, initiate a replan to current target.
                # - Optionally increment a "replan counter" metric for later reporting.
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
    
    def __str__(self) -> str:
        return f"Robot {self.id} at {self.position} ({self.status})"
    
    def __repr__(self) -> str:
        return (f"Robot(id={self.id}, pos={self.position}, status={self.status}, "
                f"collected={self.packages_collected}, distance={self.total_distance_traveled})")

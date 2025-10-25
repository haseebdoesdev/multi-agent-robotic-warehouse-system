from typing import List, Dict, Tuple, Set, Optional
from .robot import Robot
from .environment import Warehouse
from .pathfinding import a_star, path_length


class CoordinationManager:
    
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.robots: List[Robot] = []
        self.time_step = 0
        self.conflict_count = 0
    
    def add_robot(self, robot: Robot):
        self.robots.append(robot)
    
    def assign_packages(self, packages: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        assignments = {}
        available_packages = packages.copy()
        idle_robots = [r for r in self.robots if r.is_idle() and r.target_package is None]
        
        for robot in idle_robots:
            if not available_packages:
                break
            
            nearest_package = None
            min_distance = float('inf')
            
            for package in available_packages:
                path = a_star(robot.position, package, self.warehouse)
                distance = path_length(path)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_package = package
            
            if nearest_package is not None:
                assignments[robot.id] = nearest_package
                robot.set_target(nearest_package)
                available_packages.remove(nearest_package)
        
        return assignments
    
    def detect_conflicts(self) -> List[Tuple[int, int, str]]:
        conflicts = []
        
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if i >= j:
                    continue
                
                next_pos1 = robot1.get_next_position()
                next_pos2 = robot2.get_next_position()
                
                if next_pos1 is None or next_pos2 is None:
                    continue
                
                if next_pos1 == next_pos2:
                    conflicts.append((robot1.id, robot2.id, "collision"))
                
                if next_pos1 == robot2.position and next_pos2 == robot1.position:
                    conflicts.append((robot1.id, robot2.id, "head-on"))
        
        return conflicts
    
    def resolve_conflicts(self) -> bool:
        conflicts = self.detect_conflicts()
        
        if not conflicts:
            return False
        
        self.conflict_count += len(conflicts)
        
        for robot1_id, robot2_id, conflict_type in conflicts:
            robot1 = next((r for r in self.robots if r.id == robot1_id), None)
            robot2 = next((r for r in self.robots if r.id == robot2_id), None)
            
            if robot1 is None or robot2 is None:
                continue
            
            if robot1.id < robot2.id:
                robot2.wait(duration=1)
            else:
                robot1.wait(duration=1)
        
        return True
    
    def update_robots(self) -> bool:
        self.time_step += 1
        any_movement = False
        
        self.resolve_conflicts()
        
        for robot in self.robots:
            if robot.move(self.warehouse):
                any_movement = True
            
            if robot.has_reached_target():
                robot.collect_package(self.warehouse)
        
        return any_movement
    
    def plan_all_paths(self) -> int:
        planned_count = 0
        
        for robot in self.robots:
            if robot.target_package is not None and not robot.path:
                if robot.plan_path(self.warehouse):
                    planned_count += 1
        
        return planned_count
    
    def get_all_robot_positions(self) -> Set[Tuple[int, int]]:
        return {robot.position for robot in self.robots}
    
    def are_all_packages_collected(self) -> bool:
        return len(self.warehouse.packages) == 0
    
    def are_all_robots_idle(self) -> bool:
        return all(robot.is_idle() for robot in self.robots)
    
    def get_statistics(self) -> Dict[str, any]:
        total_packages = sum(robot.packages_collected for robot in self.robots)
        total_distance = sum(robot.total_distance_traveled for robot in self.robots)
        
        stats = {
            'timesteps': self.time_step,
            'total_packages_collected': total_packages,
            'total_distance_traveled': total_distance,
            'conflicts_resolved': self.conflict_count,
            'robots': []
        }
        
        for robot in self.robots:
            stats['robots'].append({
                'id': robot.id,
                'position': robot.position,
                'packages_collected': robot.packages_collected,
                'distance_traveled': robot.total_distance_traveled,
                'status': robot.status
            })
        
        return stats
    
    def reset(self):
        self.time_step = 0
        self.conflict_count = 0

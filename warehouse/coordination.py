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
        self.deadlock_count = 0
        self.replan_count = 0
    
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
        """Detect immediate conflicts between robots."""
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
    
    def detect_deadlocks(self) -> List[List[int]]:
        """
        Detect circular waiting patterns (deadlocks).
        
        A deadlock occurs when robots form a cycle where each robot
        is waiting to move into a cell occupied by the next robot in the cycle.
        
        Returns:
            List of deadlock cycles, where each cycle is a list of robot IDs
        """
        # Build dependency graph: robot -> robot whose position it wants
        waiting_for: Dict[int, int] = {}
        
        for robot in self.robots:
            if robot.status == Robot.STATUS_WAITING or robot.status == Robot.STATUS_MOVING:
                next_pos = robot.get_next_position()
                if next_pos:
                    # Check if another robot is at the position this robot wants
                    blocker = next(
                        (r for r in self.robots 
                         if r.position == next_pos and r.id != robot.id), 
                        None
                    )
                    if blocker:
                        waiting_for[robot.id] = blocker.id
        
        # Find cycles using DFS
        deadlocks = []
        visited = set()
        
        for start_id in waiting_for:
            if start_id in visited:
                continue
            
            # Follow the chain to detect cycle
            path = []
            current = start_id
            path_set = set()
            
            while current is not None and current not in visited:
                if current in path_set:
                    # Found a cycle - extract it
                    cycle_start = path.index(current)
                    cycle = path[cycle_start:]
                    if len(cycle) >= 2:  # Need at least 2 robots for a deadlock
                        deadlocks.append(cycle)
                    break
                
                path.append(current)
                path_set.add(current)
                current = waiting_for.get(current)
            
            visited.update(path)
        
        return deadlocks
    
    def resolve_deadlocks(self) -> int:
        """
        Detect and resolve deadlocks by forcing one robot in each cycle to replan.
        
        Returns:
            Number of deadlocks resolved
        """
        deadlocks = self.detect_deadlocks()
        
        if not deadlocks:
            return 0
        
        self.deadlock_count += len(deadlocks)
        occupied = self.get_all_robot_positions()
        
        for cycle in deadlocks:
            # Choose the robot with longest remaining path to replan
            # (least progress made, so least cost to replan)
            robots_in_cycle = [r for r in self.robots if r.id in cycle]
            if not robots_in_cycle:
                continue
            
            # Pick robot with longest remaining path
            robot_to_replan = max(robots_in_cycle, 
                                  key=lambda r: r.get_remaining_path_length())
            
            # Force replan with other robots as obstacles
            robot_occupied = occupied - {robot_to_replan.position}
            robot_to_replan.invalidate_path()
            robot_to_replan.plan_path(self.warehouse, 
                                       occupied_cells=robot_occupied, 
                                       is_replan=True)
            self.replan_count += 1
        
        return len(deadlocks)
    
    def resolve_conflicts(self) -> bool:
        """
        Detect and resolve conflicts between robots using smart priority.
        
        Priority is based on:
        1. Remaining distance to target (closer = higher priority)
        2. Number of packages collected (more = higher priority as tiebreaker)
        
        The lower-priority robot waits and may replan.
        """
        conflicts = self.detect_conflicts()
        
        if not conflicts:
            return False
        
        self.conflict_count += len(conflicts)
        robots_to_replan = set()
        
        for robot1_id, robot2_id, conflict_type in conflicts:
            robot1 = next((r for r in self.robots if r.id == robot1_id), None)
            robot2 = next((r for r in self.robots if r.id == robot2_id), None)
            
            if robot1 is None or robot2 is None:
                continue
            
            # Calculate priority based on remaining path length
            # Shorter remaining path = higher priority (closer to goal)
            dist1 = robot1.get_remaining_path_length()
            dist2 = robot2.get_remaining_path_length()
            
            # Determine which robot should wait
            if dist1 < dist2:
                # Robot 1 is closer to goal, robot 2 should wait
                loser = robot2
                winner = robot1
            elif dist2 < dist1:
                # Robot 2 is closer to goal, robot 1 should wait
                loser = robot1
                winner = robot2
            else:
                # Tie-breaker: robot with more packages collected has priority
                if robot1.packages_collected >= robot2.packages_collected:
                    loser = robot2
                    winner = robot1
                else:
                    loser = robot1
                    winner = robot2
            
            loser.wait(duration=1)
            
            # For head-on conflicts, the waiting robot should replan
            if conflict_type == "head-on":
                robots_to_replan.add(loser.id)
        
        # Replan for robots that need new paths
        if robots_to_replan:
            occupied = self.get_all_robot_positions()
            for robot in self.robots:
                if robot.id in robots_to_replan:
                    # Remove own position from occupied set
                    robot_occupied = occupied - {robot.position}
                    robot.plan_path(self.warehouse, occupied_cells=robot_occupied, is_replan=True)
                    self.replan_count += 1
        
        return True
    
    def update_robots(self) -> bool:
        """
        Update all robots for one timestep.
        
        This includes:
        1. Conflict detection and resolution
        2. Deadlock detection and resolution
        3. Robot movement with dynamic replanning
        4. Package collection
        
        Returns:
            True if any robot moved, False otherwise
        """
        self.time_step += 1
        any_movement = False
        
        # Step 1: Resolve immediate conflicts
        self.resolve_conflicts()
        
        # Step 2: Detect and resolve deadlocks
        self.resolve_deadlocks()
        
        # Step 3: Move robots with awareness of other robot positions
        occupied = self.get_all_robot_positions()
        
        for robot in self.robots:
            # Pass occupied cells for potential dynamic replanning
            robot_occupied = occupied - {robot.position}
            if robot.move(self.warehouse, occupied_cells=robot_occupied):
                any_movement = True
                # Update occupied set after movement
                occupied.discard(robot.position)
            
            if robot.has_reached_target():
                robot.collect_package(self.warehouse)
        
        return any_movement
    
    def plan_all_paths(self) -> int:
        """
        Plan paths for all robots that need them.
        
        Uses robot-aware pathfinding to avoid other robot positions.
        
        Returns:
            Number of paths successfully planned
        """
        planned_count = 0
        occupied = self.get_all_robot_positions()
        
        for robot in self.robots:
            if robot.target_package is not None and not robot.path:
                # Exclude own position from occupied set
                robot_occupied = occupied - {robot.position}
                if robot.plan_path(self.warehouse, occupied_cells=robot_occupied):
                    planned_count += 1
        
        return planned_count
    
    def get_all_robot_positions(self) -> Set[Tuple[int, int]]:
        return {robot.position for robot in self.robots}
    
    def are_all_packages_collected(self) -> bool:
        return len(self.warehouse.packages) == 0
    
    def are_all_robots_idle(self) -> bool:
        return all(robot.is_idle() for robot in self.robots)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the simulation."""
        total_packages = sum(robot.packages_collected for robot in self.robots)
        total_distance = sum(robot.total_distance_traveled for robot in self.robots)
        total_replans = sum(robot.replan_count for robot in self.robots)
        
        stats = {
            'timesteps': self.time_step,
            'total_packages_collected': total_packages,
            'total_distance_traveled': total_distance,
            'conflicts_resolved': self.conflict_count,
            'deadlocks_resolved': self.deadlock_count,
            'total_replans': total_replans + self.replan_count,
            'robots': []
        }
        
        for robot in self.robots:
            stats['robots'].append({
                'id': robot.id,
                'position': robot.position,
                'packages_collected': robot.packages_collected,
                'distance_traveled': robot.total_distance_traveled,
                'status': robot.status,
                'replan_count': robot.replan_count
            })
        
        return stats
    
    def reset(self):
        """Reset the coordination manager for a new simulation."""
        self.time_step = 0
        self.conflict_count = 0
        self.deadlock_count = 0
        self.replan_count = 0

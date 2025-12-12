"""
Multi-Agent Q-Learning for Warehouse Robotics
----------------------------------------------
Implements coordinated multi-robot navigation using:
- Independent Q-Learning (IQL): Each robot learns independently
- Centralized Training with Decentralized Execution (CTDE)
- Collision avoidance between robots
- Deadlock detection and resolution
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set, Deque
from dataclasses import dataclass, field
from collections import deque
import pickle
import os

from .qlearning import ACTIONS, ACTION_NAMES, OPPOSITE_ACTIONS, State, QLearningAgent


@dataclass
class MultiAgentState:
    """
    State representation for multi-agent Q-learning.
    
    Each robot's state includes:
    - Its own position relative to its target
    - Local obstacle information
    - Nearby robot positions (relative)
    - Last action taken
    - Stuck level
    """
    robot_id: int
    robot_x: int
    robot_y: int
    target_x: int
    target_y: int
    # Obstacle flags
    obstacle_up: bool = False
    obstacle_down: bool = False
    obstacle_left: bool = False
    obstacle_right: bool = False
    # Other robots in adjacent cells (treated like dynamic obstacles)
    robot_up: bool = False
    robot_down: bool = False
    robot_left: bool = False
    robot_right: bool = False
    # Action history
    last_action: int = -1
    stuck_level: int = 0
    
    def get_relative_state(self) -> Tuple:
        """
        Get relative state for Q-table lookup.
        
        Includes direction to target, obstacles, and nearby robots.
        """
        dx = int(np.sign(self.target_x - self.robot_x))
        dy = int(np.sign(self.target_y - self.robot_y))
        
        # Combine obstacles and robots (both are "blocked" directions)
        blocked_up = int(self.obstacle_up or self.robot_up)
        blocked_down = int(self.obstacle_down or self.robot_down)
        blocked_left = int(self.obstacle_left or self.robot_left)
        blocked_right = int(self.obstacle_right or self.robot_right)
        
        return (
            dx, dy,
            blocked_up, blocked_down, blocked_left, blocked_right,
            # Include pure robot presence for coordination learning
            int(self.robot_up), int(self.robot_down),
            int(self.robot_left), int(self.robot_right),
            self.last_action,
            self.stuck_level
        )
    
    def to_single_agent_state(self) -> State:
        """Convert to single-agent State for compatibility."""
        return State(
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            target_x=self.target_x,
            target_y=self.target_y,
            obstacle_up=self.obstacle_up or self.robot_up,
            obstacle_down=self.obstacle_down or self.robot_down,
            obstacle_left=self.obstacle_left or self.robot_left,
            obstacle_right=self.obstacle_right or self.robot_right,
            last_action=self.last_action,
            stuck_level=self.stuck_level
        )


@dataclass
class RobotInfo:
    """Information about a single robot in multi-agent environment."""
    robot_id: int
    position: Tuple[int, int]
    target: Optional[Tuple[int, int]] = None  # Current target (can change in shared mode)
    done: bool = False
    collected: bool = False
    packages_collected: int = 0  # Total packages collected by this robot
    steps: int = 0
    total_reward: float = 0.0
    last_action: int = -1
    position_history: Deque = field(default_factory=lambda: deque(maxlen=10))
    action_history: Deque = field(default_factory=lambda: deque(maxlen=4))
    steps_since_progress: int = 0
    best_distance: int = field(default_factory=lambda: float('inf'))


class MultiAgentWarehouseEnv:
    """
    Multi-agent environment for warehouse navigation.
    
    Features:
    - Multiple robots navigating simultaneously
    - Robot-robot collision detection
    - Deadlock detection and resolution
    - Coordinated reward shaping
    - SHARED PACKAGES MODE: Any robot can collect any package
    """
    
    def __init__(self, warehouse, num_robots: int = 3, 
                 num_packages: int = None,
                 shared_packages: bool = True):
        """
        Initialize multi-agent environment.
        
        Args:
            warehouse: Warehouse instance
            num_robots: Number of robots to simulate
            num_packages: Number of packages (defaults to num_robots if not shared)
            shared_packages: If True, any robot can collect any package.
                           If False, each robot has its own target.
        """
        self.warehouse = warehouse
        self.width = warehouse.width
        self.height = warehouse.height
        self.num_robots = num_robots
        self.shared_packages = shared_packages
        
        # Number of packages (in shared mode, can be different from num_robots)
        if num_packages is None:
            self.num_packages = num_robots + 2 if shared_packages else num_robots
        else:
            self.num_packages = num_packages
        
        self.robots: List[RobotInfo] = []
        self.available_packages: Set[Tuple[int, int]] = set()  # Shared package pool
        self.collected_packages: List[Tuple[int, int]] = []    # Collected packages
        self.total_packages_collected: int = 0
        
        self.done = False
        self.global_steps = 0
        self.max_steps = self.width * self.height * 3 * max(num_robots, self.num_packages)
        
        # Rewards
        self.reward_reach_target = 100.0
        self.reward_collect_package = 120.0  # Bonus for collecting shared package
        self.reward_step = -0.5
        self.reward_collision_obstacle = -10.0
        self.reward_collision_robot = -15.0  # Higher penalty for robot collision
        self.reward_closer = 3.0
        self.reward_farther = -2.0
        self.reward_wait = -1.0  # Less penalty for waiting (useful for coordination)
        self.reward_timeout = -50.0
        self.reward_oscillation = -8.0
        self.reward_revisit = -4.0
        self.reward_progress = 1.0
        self.reward_deadlock = -20.0  # Penalty for being in a deadlock
        self.reward_all_done_bonus = 50.0  # Bonus when all packages collected
        
        # Stuck threshold
        self.stuck_threshold = 6
    
    def reset(self, 
              robot_starts: List[Tuple[int, int]] = None,
              targets: List[Tuple[int, int]] = None,
              packages: List[Tuple[int, int]] = None) -> List[MultiAgentState]:
        """
        Reset environment for new episode.
        
        Args:
            robot_starts: Starting positions for each robot
            targets: Target positions for each robot (ignored in shared mode)
            packages: Package positions (for shared mode)
        
        Returns:
            List of initial states for each robot
        """
        self.robots.clear()
        self.available_packages.clear()
        self.collected_packages.clear()
        self.total_packages_collected = 0
        self.done = False
        self.global_steps = 0
        
        # Generate valid positions
        used_positions: Set[Tuple[int, int]] = set()
        
        # Create robots
        for i in range(self.num_robots):
            # Get start position
            if robot_starts and i < len(robot_starts):
                start = robot_starts[i]
            else:
                start = self._random_valid_position(used_positions)
            used_positions.add(start)
            
            # Create robot info (target will be assigned after packages are placed)
            robot = RobotInfo(
                robot_id=i,
                position=start,
                target=None
            )
            robot.position_history.append(start)
            self.robots.append(robot)
        
        if self.shared_packages:
            # SHARED MODE: Create a pool of packages any robot can collect
            if packages:
                for pkg in packages:
                    if pkg not in used_positions:
                        self.available_packages.add(pkg)
                        used_positions.add(pkg)
            else:
                for _ in range(self.num_packages):
                    pkg = self._random_valid_position(used_positions)
                    self.available_packages.add(pkg)
                    used_positions.add(pkg)
            
            # Assign nearest package to each robot
            self._assign_nearest_packages()
        else:
            # INDIVIDUAL MODE: Each robot has its own target
            for i, robot in enumerate(self.robots):
                if targets and i < len(targets):
                    target = targets[i]
                else:
                    target = self._random_valid_position(used_positions)
                used_positions.add(target)
                robot.target = target
                robot.best_distance = self._manhattan_distance(robot.position, target)
        
        return self._get_all_states()
    
    def _assign_nearest_packages(self):
        """Assign nearest available package to each robot without a target."""
        if not self.available_packages:
            return
        
        for robot in self.robots:
            if robot.done or robot.target is not None:
                continue
            
            # Find nearest available package
            nearest_pkg = None
            min_dist = float('inf')
            
            for pkg in self.available_packages:
                # Check if another robot is already targeting this
                already_targeted = any(
                    r.target == pkg for r in self.robots 
                    if r.robot_id != robot.robot_id and not r.done
                )
                if already_targeted:
                    continue
                
                dist = self._manhattan_distance(robot.position, pkg)
                if dist < min_dist:
                    min_dist = dist
                    nearest_pkg = pkg
            
            # If all packages are targeted, just pick the nearest one anyway
            if nearest_pkg is None and self.available_packages:
                for pkg in self.available_packages:
                    dist = self._manhattan_distance(robot.position, pkg)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pkg = pkg
            
            if nearest_pkg:
                robot.target = nearest_pkg
                robot.best_distance = min_dist
                robot.steps_since_progress = 0
    
    def _random_valid_position(self, exclude: Set[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Get random valid position not in exclude set."""
        exclude = exclude or set()
        attempts = 0
        while attempts < 100:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            pos = (x, y)
            if pos not in exclude and self.warehouse.is_valid_move(x, y, ignore_packages=True):
                return pos
            attempts += 1
        # Fallback: find any valid position
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if pos not in exclude and self.warehouse.is_valid_move(x, y, ignore_packages=True):
                    return pos
        return (0, 0)
    
    def _get_robot_positions(self) -> Set[Tuple[int, int]]:
        """Get set of all current robot positions."""
        return {r.position for r in self.robots if not r.done}
    
    def _get_state_for_robot(self, robot: RobotInfo) -> MultiAgentState:
        """Get state for a specific robot."""
        rx, ry = robot.position
        
        # Get target position (use robot's own position if no target - means done)
        if robot.target is not None:
            tx, ty = robot.target
        else:
            # No target - robot is either done or waiting for assignment
            # In shared mode, find nearest available package
            if self.shared_packages and self.available_packages:
                nearest = min(self.available_packages, 
                            key=lambda p: self._manhattan_distance(robot.position, p))
                tx, ty = nearest
            else:
                tx, ty = rx, ry  # At target (done)
        
        # Get other robot positions
        other_positions = {r.position for r in self.robots 
                         if r.robot_id != robot.robot_id and not r.done}
        
        # Check obstacles
        obstacle_up = not self._is_valid_position(rx, ry - 1)
        obstacle_down = not self._is_valid_position(rx, ry + 1)
        obstacle_left = not self._is_valid_position(rx - 1, ry)
        obstacle_right = not self._is_valid_position(rx + 1, ry)
        
        # Check for other robots
        robot_up = (rx, ry - 1) in other_positions
        robot_down = (rx, ry + 1) in other_positions
        robot_left = (rx - 1, ry) in other_positions
        robot_right = (rx + 1, ry) in other_positions
        
        # Stuck level
        if robot.steps_since_progress >= self.stuck_threshold * 2:
            stuck_level = 2
        elif robot.steps_since_progress >= self.stuck_threshold:
            stuck_level = 1
        else:
            stuck_level = 0
        
        return MultiAgentState(
            robot_id=robot.robot_id,
            robot_x=rx,
            robot_y=ry,
            target_x=tx,
            target_y=ty,
            obstacle_up=obstacle_up,
            obstacle_down=obstacle_down,
            obstacle_left=obstacle_left,
            obstacle_right=obstacle_right,
            robot_up=robot_up,
            robot_down=robot_down,
            robot_left=robot_left,
            robot_right=robot_right,
            last_action=robot.last_action,
            stuck_level=stuck_level
        )
    
    def _get_all_states(self) -> List[MultiAgentState]:
        """Get states for all robots."""
        return [self._get_state_for_robot(r) for r in self.robots]
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (in bounds and not obstacle)."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.warehouse.is_valid_move(x, y, ignore_packages=True)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _detect_oscillation(self, robot: RobotInfo, action: int) -> bool:
        """Detect if robot is oscillating."""
        if len(robot.action_history) < 2:
            return False
        
        prev = robot.action_history[-1]
        prev_prev = robot.action_history[-2]
        
        if action == prev_prev and prev == OPPOSITE_ACTIONS.get(action, -1):
            return True
        
        return False
    
    def _count_recent_visits(self, robot: RobotInfo, position: Tuple[int, int]) -> int:
        """Count recent visits to a position."""
        return sum(1 for pos in robot.position_history if pos == position)
    
    def _detect_deadlock(self) -> List[int]:
        """
        Detect robots in deadlock (mutually blocking each other).
        
        Returns list of robot IDs in deadlock.
        """
        deadlocked = []
        
        for robot in self.robots:
            if robot.done:
                continue
            
            # Check if all directions are blocked
            rx, ry = robot.position
            other_positions = {r.position for r in self.robots 
                             if r.robot_id != robot.robot_id and not r.done}
            
            blocked_count = 0
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_pos = (rx + dx, ry + dy)
                if not self._is_valid_position(new_pos[0], new_pos[1]):
                    blocked_count += 1
                elif new_pos in other_positions:
                    blocked_count += 1
            
            # All 4 directions blocked + stuck for a while = deadlock
            if blocked_count >= 4 and robot.steps_since_progress >= self.stuck_threshold:
                deadlocked.append(robot.robot_id)
        
        return deadlocked
    
    def step(self, actions: List[int]) -> Tuple[List[MultiAgentState], List[float], List[bool], Dict]:
        """
        Take a step for all robots simultaneously.
        
        Args:
            actions: List of actions, one per robot
        
        Returns:
            (states, rewards, dones, info) for all robots
        """
        if self.done:
            states = self._get_all_states()
            return states, [0.0] * self.num_robots, [True] * self.num_robots, {'message': 'Done'}
        
        self.global_steps += 1
        
        rewards = [0.0] * self.num_robots
        dones = [False] * self.num_robots
        infos = [{} for _ in range(self.num_robots)]
        
        # Calculate intended positions for all robots
        intended_positions: List[Tuple[int, int]] = []
        for i, robot in enumerate(self.robots):
            if robot.done:
                intended_positions.append(robot.position)
                continue
            
            action = actions[i] if i < len(actions) else 4
            dx, dy = ACTIONS.get(action, (0, 0))
            new_x = robot.position[0] + dx
            new_y = robot.position[1] + dy
            intended_positions.append((new_x, new_y))
        
        # Detect collisions between robots
        position_claims: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(intended_positions):
            if not self.robots[i].done:
                if pos not in position_claims:
                    position_claims[pos] = []
                position_claims[pos].append(i)
        
        # Also check for swap collisions (robots swapping positions)
        swap_collisions: Set[int] = set()
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if self.robots[i].done or self.robots[j].done:
                    continue
                # Check if they're swapping
                if (intended_positions[i] == self.robots[j].position and 
                    intended_positions[j] == self.robots[i].position):
                    swap_collisions.add(i)
                    swap_collisions.add(j)
        
        # Process each robot's action
        for i, robot in enumerate(self.robots):
            if robot.done:
                dones[i] = True
                continue
            
            action = actions[i] if i < len(actions) else 4
            robot.steps += 1
            infos[i]['action'] = ACTION_NAMES.get(action, 'UNKNOWN')
            
            dx, dy = ACTIONS.get(action, (0, 0))
            
            # WAIT action
            if action == 4 or (dx == 0 and dy == 0):
                rewards[i] = self.reward_wait
                infos[i]['message'] = 'Waiting'
                robot.steps_since_progress += 1
                robot.action_history.append(action)
                robot.last_action = action
                continue
            
            # Check oscillation
            is_oscillating = self._detect_oscillation(robot, action)
            
            old_pos = robot.position
            new_x, new_y = intended_positions[i]
            
            # Calculate distance to target (handle None target)
            if robot.target:
                old_distance = self._manhattan_distance(old_pos, robot.target)
            else:
                old_distance = 0
            
            # Check for collisions
            collision_type = None
            
            # Out of bounds
            if not self.warehouse.is_within_bounds(new_x, new_y):
                collision_type = 'bounds'
                rewards[i] = self.reward_collision_obstacle
                infos[i]['message'] = 'Out of bounds'
            # Obstacle collision
            elif not self.warehouse.is_valid_move(new_x, new_y, ignore_packages=True):
                collision_type = 'obstacle'
                rewards[i] = self.reward_collision_obstacle
                infos[i]['message'] = 'Hit obstacle'
            # Swap collision
            elif i in swap_collisions:
                collision_type = 'robot_swap'
                rewards[i] = self.reward_collision_robot
                infos[i]['message'] = 'Swap collision with robot'
            # Position collision (multiple robots want same cell)
            elif len(position_claims.get((new_x, new_y), [])) > 1:
                collision_type = 'robot_collision'
                rewards[i] = self.reward_collision_robot
                infos[i]['message'] = 'Collision with robot'
            
            if collision_type:
                robot.steps_since_progress += 1
            else:
                # Valid move
                robot.position = (new_x, new_y)
                
                # SHARED PACKAGES MODE: Check if robot reached ANY available package
                collected_package = False
                if self.shared_packages and robot.position in self.available_packages:
                    # Collect this package!
                    self.available_packages.remove(robot.position)
                    self.collected_packages.append(robot.position)
                    self.total_packages_collected += 1
                    robot.packages_collected += 1
                    robot.collected = True
                    collected_package = True
                    
                    speed_bonus = max(0, (self.max_steps - self.global_steps) / self.max_steps * 20)
                    rewards[i] = self.reward_collect_package + speed_bonus
                    infos[i]['message'] = f'Collected package! (#{robot.packages_collected})'
                    infos[i]['collected'] = True
                    robot.steps_since_progress = 0
                    
                    # Assign new target if more packages available
                    robot.target = None
                    robot.best_distance = float('inf')
                    
                    if self.available_packages:
                        # Find nearest remaining package
                        nearest = min(self.available_packages,
                                     key=lambda p: self._manhattan_distance(robot.position, p))
                        robot.target = nearest
                        robot.best_distance = self._manhattan_distance(robot.position, nearest)
                    else:
                        # No more packages - robot is done
                        robot.done = True
                        dones[i] = True
                
                # Check if reached assigned target (individual mode or shared mode target)
                elif robot.target and robot.position == robot.target:
                    if not self.shared_packages:
                        # Individual mode - robot reached its specific target
                        speed_bonus = max(0, (self.max_steps - self.global_steps) / self.max_steps * 20)
                        rewards[i] = self.reward_reach_target + speed_bonus
                        robot.done = True
                        robot.collected = True
                        dones[i] = True
                        infos[i]['message'] = 'Reached target!'
                        robot.steps_since_progress = 0
                    else:
                        # Shared mode but reached target that was already collected
                        # Reassign to nearest available
                        if self.available_packages:
                            nearest = min(self.available_packages,
                                         key=lambda p: self._manhattan_distance(robot.position, p))
                            robot.target = nearest
                            robot.best_distance = self._manhattan_distance(robot.position, nearest)
                            infos[i]['message'] = 'Target collected by other, reassigning'
                        else:
                            robot.done = True
                            dones[i] = True
                            infos[i]['message'] = 'All packages collected!'
                
                elif not collected_package:
                    # Normal movement (not collecting)
                    if robot.target:
                        new_distance = self._manhattan_distance(robot.position, robot.target)
                        
                        if new_distance < old_distance:
                            rewards[i] = self.reward_step + self.reward_closer
                            infos[i]['message'] = 'Moving closer'
                            robot.steps_since_progress = 0
                            
                            if new_distance < robot.best_distance:
                                robot.best_distance = new_distance
                                rewards[i] += self.reward_progress
                        elif new_distance > old_distance:
                            rewards[i] = self.reward_step + self.reward_farther
                            infos[i]['message'] = 'Moving farther'
                            robot.steps_since_progress += 1
                        else:
                            rewards[i] = self.reward_step
                            infos[i]['message'] = 'Same distance'
                            robot.steps_since_progress += 1
                        
                        # Oscillation penalty
                        if is_oscillating:
                            rewards[i] += self.reward_oscillation
                            infos[i]['message'] += ' (oscillating)'
                        
                        # Revisit penalty
                        visit_count = self._count_recent_visits(robot, robot.position)
                        if visit_count > 0:
                            rewards[i] += self.reward_revisit * visit_count
                            infos[i]['message'] += f' (revisit x{visit_count})'
                    else:
                        # No target - just step penalty
                        rewards[i] = self.reward_step
                        infos[i]['message'] = 'No target'
            
            # Update history
            robot.position_history.append(robot.position)
            robot.action_history.append(action)
            robot.last_action = action
            robot.total_reward += rewards[i]
        
        # Check for deadlocks
        deadlocked = self._detect_deadlock()
        for robot_id in deadlocked:
            rewards[robot_id] += self.reward_deadlock
            infos[robot_id]['deadlock'] = True
        
        # Check completion condition
        if self.shared_packages:
            # SHARED MODE: Done when all packages collected
            all_done = len(self.available_packages) == 0
            if all_done:
                self.done = True
                # Mark all robots as done and give bonus
                for i, robot in enumerate(self.robots):
                    if not robot.done:
                        robot.done = True
                        dones[i] = True
                    rewards[i] += self.reward_all_done_bonus / self.num_robots
        else:
            # INDIVIDUAL MODE: Done when all robots reached their targets
            all_done = all(r.done for r in self.robots)
            if all_done:
                self.done = True
                for i in range(self.num_robots):
                    rewards[i] += self.reward_all_done_bonus / self.num_robots
        
        # Timeout check
        if self.global_steps >= self.max_steps and not self.done:
            self.done = True
            for i, robot in enumerate(self.robots):
                if not robot.done:
                    rewards[i] += self.reward_timeout
                    dones[i] = True
                    infos[i]['message'] = 'Timeout'
        
        # Include package info in return
        info_dict = {
            'robots': infos, 
            'deadlocked': deadlocked,
            'packages_remaining': len(self.available_packages),
            'packages_collected': self.total_packages_collected
        }
        
        states = self._get_all_states()
        return states, rewards, dones, info_dict
    
    def render(self) -> str:
        """Render environment as string."""
        lines = []
        robot_positions = {r.position: r.robot_id for r in self.robots}
        
        # For display: show available packages as 'P', robot targets as 'T'
        target_positions = {r.target: r.robot_id for r in self.robots 
                          if not r.done and r.target is not None}
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos in robot_positions:
                    row.append(str(robot_positions[pos]))
                elif self.shared_packages and pos in self.available_packages:
                    row.append('P')  # Available package
                elif pos in target_positions:
                    row.append('T')
                elif self.warehouse.grid[y, x] == 1:
                    row.append('#')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        
        # Add status line
        if self.shared_packages:
            lines.append(f"\nPackages: {self.total_packages_collected}/{self.num_packages} collected")
        
        return '\n'.join(lines)


class MultiAgentQLearning:
    """
    Multi-Agent Q-Learning with support for:
    - Independent Q-Learning (IQL): Separate Q-table per robot
    - Parameter Sharing: Shared Q-table across all robots
    - Hybrid: Shared + robot-specific adjustments
    """
    
    def __init__(self,
                 num_robots: int = 3,
                 alpha: float = 0.15,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 shared_qtable: bool = True):
        """
        Initialize multi-agent Q-learning.
        
        Args:
            num_robots: Number of robots
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            shared_qtable: If True, all robots share one Q-table (parameter sharing)
        """
        self.num_robots = num_robots
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.shared_qtable = shared_qtable
        self.n_actions = len(ACTIONS)
        
        if shared_qtable:
            # Shared Q-table for all robots
            self.q_table: Dict[Tuple, np.ndarray] = {}
            self.agents = None
        else:
            # Individual Q-table per robot
            self.agents: List[QLearningAgent] = [
                QLearningAgent(
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    use_relative_state=True
                )
                for _ in range(num_robots)
            ]
            self.q_table = None
        
        # Statistics
        self.episode_rewards: List[List[float]] = []  # Per robot rewards
        self.episode_successes: List[List[bool]] = []
    
    def _get_state_key(self, state: MultiAgentState) -> Tuple:
        """Get Q-table key from state."""
        return state.get_relative_state()
    
    def _get_q_values(self, state: MultiAgentState) -> np.ndarray:
        """Get Q-values for a state."""
        if self.shared_qtable:
            key = self._get_state_key(state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.n_actions)
            return self.q_table[key]
        else:
            agent = self.agents[state.robot_id]
            return agent._get_q_values(state.to_single_agent_state())
    
    def select_action(self, state: MultiAgentState, training: bool = True) -> int:
        """Select action for a robot."""
        # Stuck breaking
        if state.stuck_level == 2:
            if np.random.random() < 0.5:
                return self._smart_random_action(state)
        elif state.stuck_level == 1:
            if np.random.random() < 0.3:
                return self._smart_random_action(state)
        
        if training and np.random.random() < self.epsilon:
            return self._smart_random_action(state)
        
        q_values = self._get_q_values(state)
        
        if np.all(q_values == 0) and not training:
            return self._heuristic_action(state)
        
        # Tie-breaking
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        if len(best_actions) > 1:
            return int(np.random.choice(best_actions))
        
        return int(np.argmax(q_values))
    
    def select_actions(self, states: List[MultiAgentState], training: bool = True) -> List[int]:
        """Select actions for all robots."""
        return [self.select_action(s, training) for s in states]
    
    def _smart_random_action(self, state: MultiAgentState) -> int:
        """Smart random action avoiding blocked directions."""
        rel = state.get_relative_state()
        blocked = {
            0: rel[2],  # UP
            1: rel[3],  # DOWN
            2: rel[4],  # LEFT
            3: rel[5],  # RIGHT
        }
        
        valid_actions = [a for a in range(4) if not blocked[a]]
        
        # Filter opposite of last action
        if state.last_action >= 0 and state.last_action < 4:
            opposite = OPPOSITE_ACTIONS.get(state.last_action, -1)
            filtered = [a for a in valid_actions if a != opposite]
            if filtered:
                valid_actions = filtered
        
        if not valid_actions:
            return 4  # WAIT
        
        return int(np.random.choice(valid_actions))
    
    def _heuristic_action(self, state: MultiAgentState) -> int:
        """Heuristic action toward target."""
        dx = int(np.sign(state.target_x - state.robot_x))
        dy = int(np.sign(state.target_y - state.robot_y))
        rel = state.get_relative_state()
        
        blocked = {
            0: rel[2],  # UP
            1: rel[3],  # DOWN
            2: rel[4],  # LEFT
            3: rel[5],  # RIGHT
        }
        
        preferred = []
        if dx > 0 and not blocked[3]:
            preferred.append(3)  # RIGHT
        elif dx < 0 and not blocked[2]:
            preferred.append(2)  # LEFT
        if dy > 0 and not blocked[1]:
            preferred.append(1)  # DOWN
        elif dy < 0 and not blocked[0]:
            preferred.append(0)  # UP
        
        if preferred:
            return int(np.random.choice(preferred))
        
        # Any unblocked
        valid = [a for a in range(4) if not blocked[a]]
        if valid:
            return int(np.random.choice(valid))
        
        return 4
    
    def update(self, state: MultiAgentState, action: int, reward: float,
               next_state: MultiAgentState, done: bool):
        """Update Q-values for a single robot."""
        if self.shared_qtable:
            current_q = self._get_q_values(state)
            next_q = self._get_q_values(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(next_q)
            
            key = self._get_state_key(state)
            self.q_table[key][action] += self.alpha * (target - current_q[action])
        else:
            agent = self.agents[state.robot_id]
            agent.update(state.to_single_agent_state(), action, reward,
                        next_state.to_single_agent_state(), done)
    
    def update_all(self, states: List[MultiAgentState], actions: List[int],
                   rewards: List[float], next_states: List[MultiAgentState],
                   dones: List[bool]):
        """Update Q-values for all robots."""
        for i in range(len(states)):
            self.update(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if not self.shared_qtable:
            for agent in self.agents:
                agent.decay_epsilon()
    
    def get_policy_actions(self, states: List[MultiAgentState]) -> List[int]:
        """Get greedy policy actions for all robots."""
        return [self.select_action(s, training=False) for s in states]
    
    def save(self, filepath: str):
        """Save the multi-agent Q-learning model."""
        data = {
            'num_robots': self.num_robots,
            'shared_qtable': self.shared_qtable,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }
        
        if self.shared_qtable:
            data['q_table'] = self.q_table
        else:
            data['agent_q_tables'] = [a.q_table for a in self.agents]
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load multi-agent Q-learning model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.num_robots = data.get('num_robots', self.num_robots)
        self.shared_qtable = data.get('shared_qtable', True)
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.episode_rewards = data.get('episode_rewards', [])
        
        if self.shared_qtable:
            self.q_table = data.get('q_table', {})
        else:
            q_tables = data.get('agent_q_tables', [])
            for i, qt in enumerate(q_tables):
                if i < len(self.agents):
                    self.agents[i].q_table = qt
    
    def get_qtable_size(self) -> int:
        """Get total Q-table size."""
        if self.shared_qtable:
            return len(self.q_table)
        else:
            return sum(len(a.q_table) for a in self.agents)


def train_multi_agent(warehouse,
                      num_robots: int = 3,
                      episodes: int = 1000,
                      max_steps_per_episode: int = 500,
                      shared_qtable: bool = True,
                      verbose: bool = True,
                      save_path: str = None) -> Tuple[MultiAgentQLearning, Dict]:
    """
    Train multi-agent Q-learning.
    
    Args:
        warehouse: Warehouse instance
        num_robots: Number of robots
        episodes: Training episodes
        max_steps_per_episode: Max steps per episode
        shared_qtable: Whether to share Q-table
        verbose: Print progress
        save_path: Path to save model
    
    Returns:
        (trained agent, training stats)
    """
    env = MultiAgentWarehouseEnv(warehouse, num_robots)
    agent = MultiAgentQLearning(
        num_robots=num_robots,
        shared_qtable=shared_qtable
    )
    
    stats = {
        'episode_rewards': [],
        'success_rates': [],
        'avg_steps': [],
    }
    
    for episode in range(episodes):
        states = env.reset()
        episode_rewards = [0.0] * num_robots
        steps = 0
        
        while not env.done and steps < max_steps_per_episode:
            actions = agent.select_actions(states, training=True)
            next_states, rewards, dones, info = env.step(actions)
            
            agent.update_all(states, actions, rewards, next_states, dones)
            
            for i in range(num_robots):
                episode_rewards[i] += rewards[i]
            
            states = next_states
            steps += 1
        
        agent.decay_epsilon()
        
        # Track stats
        successes = [r.collected for r in env.robots]
        stats['episode_rewards'].append(sum(episode_rewards))
        stats['success_rates'].append(sum(successes) / num_robots)
        stats['avg_steps'].append(steps)
        
        if verbose and (episode + 1) % 100 == 0:
            recent_rewards = stats['episode_rewards'][-100:]
            recent_success = stats['success_rates'][-100:]
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Avg Reward = {np.mean(recent_rewards):.1f}, "
                  f"Success Rate = {np.mean(recent_success) * 100:.1f}%, "
                  f"Epsilon = {agent.epsilon:.3f}, "
                  f"Q-table = {agent.get_qtable_size()}")
    
    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    return agent, stats


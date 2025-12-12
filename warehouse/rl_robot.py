"""
RL-Enhanced Robot Module
------------------------
A Robot class that uses a trained Q-Learning agent for navigation.
Falls back to A* pathfinding when RL model is unavailable.
"""

from typing import Tuple, List, Optional
import os

from .robot import Robot
from .environment import Warehouse
from .pathfinding import a_star


class RLRobot(Robot):
    """
    Robot that uses trained RL agent for intelligent navigation.
    
    Key features:
    - Uses learned Q-values to navigate around obstacles
    - Falls back to A* when stuck or for initial path planning
    - Works on unseen grids (uses relative state representation)
    - Inherits all functionality from base Robot class
    """
    
    # Class-level shared RL agent (loaded once, shared by all robots)
    _rl_agent = None
    _rl_loaded = False
    _rl_available = False
    
    def __init__(self, robot_id: int, start_position: Tuple[int, int], 
                 warehouse: Warehouse = None):
        """
        Initialize RL-enhanced robot.
        
        Args:
            robot_id: Unique robot identifier
            start_position: Starting (x, y) position
            warehouse: Warehouse instance for obstacle checking
        """
        super().__init__(robot_id, start_position)
        self.warehouse = warehouse
        self.use_rl = True  # Can toggle RL on/off
        self.rl_steps = 0   # Count of steps taken using RL
        self.astar_steps = 0  # Count of steps taken using A*
        self._last_positions: List[Tuple[int, int]] = []  # Track recent positions
        self._stuck_counter = 0
    
    @classmethod
    def load_rl_agent(cls, model_paths: List[str] = None):
        """
        Load the RL agent (class method - loads once for all robots).
        
        Args:
            model_paths: List of paths to try loading (in order)
        """
        if cls._rl_loaded:
            return cls._rl_available
        
        cls._rl_loaded = True
        
        # Default paths to try
        if model_paths is None:
            model_paths = [
                "rl_agent_gui.pkl",
                "rl_agent.pkl",
                "multi_rl_agent.pkl"
            ]
        
        try:
            from rl import QLearningAgent
            
            for path in model_paths:
                if os.path.exists(path):
                    cls._rl_agent = QLearningAgent(use_relative_state=True)
                    cls._rl_agent.load(path)
                    cls._rl_agent.epsilon = 0.05  # Low exploration for deployment
                    cls._rl_available = True
                    print(f"[RL] Loaded trained agent from: {path}")
                    print(f"[RL] Q-table size: {len(cls._rl_agent.q_table)} states")
                    return True
            
            print("[RL] No trained model found. Using A* only.")
            return False
            
        except Exception as e:
            print(f"[RL] Failed to load agent: {e}")
            return False
    
    @classmethod
    def is_rl_available(cls) -> bool:
        """Check if RL agent is available."""
        return cls._rl_available
    
    def set_warehouse(self, warehouse: Warehouse):
        """Set the warehouse reference."""
        self.warehouse = warehouse
    
    def toggle_rl(self, enabled: bool = None):
        """Toggle RL navigation on/off."""
        if enabled is not None:
            self.use_rl = enabled
        else:
            self.use_rl = not self.use_rl
        return self.use_rl
    
    def _get_rl_state(self):
        """Get current state for RL agent."""
        if not self.target_package or not self.warehouse:
            return None
        
        try:
            from rl import State
            
            rx, ry = self.position
            tx, ty = self.target_package
            
            # Check obstacles in each direction
            obs_up = not self._is_valid_move(rx, ry - 1)
            obs_down = not self._is_valid_move(rx, ry + 1)
            obs_left = not self._is_valid_move(rx - 1, ry)
            obs_right = not self._is_valid_move(rx + 1, ry)
            
            # Determine stuck level
            if self._stuck_counter >= 12:
                stuck_level = 2
            elif self._stuck_counter >= 6:
                stuck_level = 1
            else:
                stuck_level = 0
            
            return State(
                robot_x=rx, robot_y=ry,
                target_x=tx, target_y=ty,
                obstacle_up=obs_up,
                obstacle_down=obs_down,
                obstacle_left=obs_left,
                obstacle_right=obs_right,
                last_action=-1,  # We don't track this in deployment
                stuck_level=stuck_level
            )
        except Exception:
            return None
    
    def _is_valid_move(self, x: int, y: int) -> bool:
        """Check if a position is valid."""
        if not self.warehouse:
            return False
        if x < 0 or x >= self.warehouse.width or y < 0 or y >= self.warehouse.height:
            return False
        return self.warehouse.is_valid_move(x, y, ignore_packages=True)
    
    def get_rl_next_position(self) -> Optional[Tuple[int, int]]:
        """
        Get next position using RL policy.
        
        Returns:
            Next position if RL suggests a valid move, None otherwise
        """
        if not self._rl_available or not self._rl_agent:
            return None
        
        state = self._get_rl_state()
        if state is None:
            return None
        
        try:
            from rl import ACTIONS
            
            # Get action from RL policy
            action = self._rl_agent.get_policy(state)
            
            if action == 4:  # WAIT action
                return None
            
            dx, dy = ACTIONS.get(action, (0, 0))
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            # Validate the move
            if self._is_valid_move(new_x, new_y):
                return (new_x, new_y)
            
            return None
            
        except Exception:
            return None
    
    def plan_path(self, warehouse: Warehouse, target: Optional[Tuple[int, int]] = None) -> bool:
        """
        Plan path using A* (for initial planning and fallback).
        RL is used during move() for dynamic adjustments.
        """
        self.warehouse = warehouse
        return super().plan_path(warehouse, target)
    
    def move(self, warehouse: Warehouse) -> bool:
        """
        Move robot, using RL for navigation when available.
        
        The RL agent provides step-by-step guidance, while A* is used
        as a fallback when RL doesn't have a good action.
        """
        self.warehouse = warehouse
        
        # Track position history for stuck detection
        self._last_positions.append(self.position)
        if len(self._last_positions) > 10:
            self._last_positions.pop(0)
        
        # Check if we're stuck (oscillating)
        if len(self._last_positions) >= 4:
            recent = self._last_positions[-4:]
            if len(set(recent)) <= 2:  # Only 2 unique positions in last 4
                self._stuck_counter += 1
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)
        
        # If waiting, delegate to parent
        if self.status == self.STATUS_WAITING:
            return super().move(warehouse)
        
        # If no target or no path, delegate to parent
        if not self.target_package:
            return super().move(warehouse)
        
        # Try RL navigation if enabled and available
        if self.use_rl and self._rl_available and self.target_package:
            rl_next = self.get_rl_next_position()
            
            if rl_next and rl_next != self.position:
                # Check for collisions with other robots (via path)
                # For now, just use RL suggestion directly
                old_pos = self.position
                self.position = rl_next
                self.total_distance_traveled += 1
                self.rl_steps += 1
                self.status = self.STATUS_MOVING
                
                # Update path if we have one (for visualization)
                if self.path and self.path_index < len(self.path):
                    # Check if RL move matches path
                    if self.path_index + 1 < len(self.path):
                        if self.path[self.path_index + 1] == rl_next:
                            self.path_index += 1
                        else:
                            # RL deviated from A* path - replan for visualization
                            self.plan_path(warehouse)
                
                return True
        
        # Fallback to A* movement
        result = super().move(warehouse)
        if result:
            self.astar_steps += 1
        return result
    
    def get_navigation_stats(self) -> dict:
        """Get navigation statistics."""
        total = self.rl_steps + self.astar_steps
        return {
            'rl_steps': self.rl_steps,
            'astar_steps': self.astar_steps,
            'total_steps': total,
            'rl_percentage': (self.rl_steps / total * 100) if total > 0 else 0,
            'stuck_counter': self._stuck_counter
        }
    
    def __repr__(self) -> str:
        return (f"RLRobot(id={self.id}, pos={self.position}, status={self.status}, "
                f"rl_steps={self.rl_steps}, astar_steps={self.astar_steps})")



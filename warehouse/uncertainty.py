"""
Uncertainty and dynamic adaptation module (Module 4)
----------------------------------------------------
This module implements probabilistic obstacles, sensor simulation,
and predictive rerouting for dynamic path replanning.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from .environment import Warehouse


@dataclass
class SensorObservation:
    """Represents a single sensor observation of a cell."""
    x: int
    y: int
    detected_obstacle: bool
    confidence: float


class ProbabilisticObstacleMap:
    """
    Maintains a probability grid for potential obstacles.
    Values range from 0.0 (definitely empty) to 1.0 (definitely obstacle).
    """
    
    def __init__(self, width: int, height: int, default_prob: float = 0.0):
        self.width = width
        self.height = height
        self.prob_grid = np.full((height, width), default_prob, dtype=float)
        self.last_update_time = np.zeros((height, width), dtype=int)
        self.current_time = 0
    
    def initialize_from_warehouse(self, warehouse: Warehouse):
        """Initialize probabilities from known warehouse obstacles."""
        for y in range(self.height):
            for x in range(self.width):
                if warehouse.grid[y, x] == Warehouse.OBSTACLE:
                    self.prob_grid[y, x] = 1.0
                else:
                    self.prob_grid[y, x] = 0.0
    
    def update_from_sensor(self, observations: List[SensorObservation], 
                           learning_rate: float = 0.3):
        """
        Update probabilities based on sensor observations.
        Uses Bayesian-like update with learning rate for smoothing.
        """
        for obs in observations:
            if 0 <= obs.x < self.width and 0 <= obs.y < self.height:
                current_prob = self.prob_grid[obs.y, obs.x]
                
                if obs.detected_obstacle:
                    # Increase probability towards 1.0
                    target = obs.confidence
                else:
                    # Decrease probability towards 0.0
                    target = 1.0 - obs.confidence
                
                # Smooth update
                new_prob = current_prob + learning_rate * (target - current_prob)
                self.prob_grid[obs.y, obs.x] = np.clip(new_prob, 0.0, 1.0)
                self.last_update_time[obs.y, obs.x] = self.current_time
    
    def decay_uncertainty(self, decay_rate: float = 0.01):
        """
        Decay probabilities towards neutral (0.5) over time for cells
        that haven't been observed recently.
        """
        self.current_time += 1
        
        # Only decay cells that are not definitely known (0 or 1)
        uncertain_mask = (self.prob_grid > 0.01) & (self.prob_grid < 0.99)
        
        for y in range(self.height):
            for x in range(self.width):
                if uncertain_mask[y, x]:
                    time_since_update = self.current_time - self.last_update_time[y, x]
                    if time_since_update > 5:  # Only decay if not recently observed
                        current = self.prob_grid[y, x]
                        # Decay towards 0.5 (maximum uncertainty)
                        if current > 0.5:
                            self.prob_grid[y, x] = max(0.5, current - decay_rate)
                        elif current < 0.5:
                            self.prob_grid[y, x] = min(0.5, current + decay_rate)
    
    def commit_to_environment(self, warehouse: Warehouse, 
                               obstacle_threshold: float = 0.8,
                               clear_threshold: float = 0.2):
        """
        Commit high-confidence predictions to the warehouse environment.
        - Cells with prob >= obstacle_threshold become obstacles
        - Cells with prob <= clear_threshold become empty (if was obstacle)
        """
        committed_changes = []
        
        for y in range(self.height):
            for x in range(self.width):
                prob = self.prob_grid[y, x]
                current_cell = warehouse.grid[y, x]
                
                if prob >= obstacle_threshold and current_cell == Warehouse.EMPTY:
                    warehouse.grid[y, x] = Warehouse.OBSTACLE
                    committed_changes.append((x, y, 'obstacle'))
                elif prob <= clear_threshold and current_cell == Warehouse.OBSTACLE:
                    warehouse.grid[y, x] = Warehouse.EMPTY
                    committed_changes.append((x, y, 'cleared'))
        
        return committed_changes
    
    def sample_random_changes(self, rate: float, rng: np.random.Generator = None):
        """
        Randomly flip some cell probabilities to simulate dynamic environment.
        Returns list of cells that were modified.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        modified = []
        num_cells = int(self.width * self.height * rate)
        
        for _ in range(num_cells):
            x = rng.integers(0, self.width)
            y = rng.integers(0, self.height)
            
            # Flip probability
            current = self.prob_grid[y, x]
            if current < 0.3:
                self.prob_grid[y, x] = rng.uniform(0.6, 0.9)
            elif current > 0.7:
                self.prob_grid[y, x] = rng.uniform(0.1, 0.4)
            
            modified.append((x, y))
        
        return modified
    
    def get_probability(self, x: int, y: int) -> float:
        """Get obstacle probability at a specific cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.prob_grid[y, x]
        return 1.0  # Out of bounds treated as obstacle
    
    def get_risk_map(self) -> np.ndarray:
        """Return copy of the probability grid for visualization."""
        return self.prob_grid.copy()


class SensorModel:
    """
    Simulates noisy sensor detections around a robot position.
    Models both false positives and false negatives.
    """
    
    def __init__(self, radius: int = 2, 
                 false_positive_rate: float = 0.05,
                 false_negative_rate: float = 0.10,
                 base_confidence: float = 0.8):
        self.radius = radius
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.base_confidence = base_confidence
        self.rng = np.random.default_rng()
    
    def sense_local(self, warehouse: Warehouse, 
                    robot_position: Tuple[int, int]) -> List[SensorObservation]:
        """
        Sense cells within radius of the robot position.
        Returns noisy observations with confidence values.
        """
        observations = []
        rx, ry = robot_position
        
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip robot's own position
                
                x, y = rx + dx, ry + dy
                
                if not warehouse.is_within_bounds(x, y):
                    continue
                
                distance = abs(dx) + abs(dy)  # Manhattan distance
                
                # Confidence decreases with distance
                distance_factor = 1.0 - (distance / (self.radius + 1)) * 0.3
                confidence = self.base_confidence * distance_factor
                
                actual_obstacle = warehouse.grid[y, x] == Warehouse.OBSTACLE
                
                # Apply sensor noise
                detected = actual_obstacle
                if actual_obstacle:
                    # False negative: fail to detect obstacle
                    if self.rng.random() < self.false_negative_rate:
                        detected = False
                        confidence *= 0.7
                else:
                    # False positive: detect obstacle when none exists
                    if self.rng.random() < self.false_positive_rate:
                        detected = True
                        confidence *= 0.5
                
                # Add some noise to confidence
                confidence += self.rng.uniform(-0.1, 0.1)
                confidence = np.clip(confidence, 0.1, 1.0)
                
                observations.append(SensorObservation(
                    x=x, y=y,
                    detected_obstacle=detected,
                    confidence=confidence
                ))
        
        return observations
    
    def sense_along_path(self, warehouse: Warehouse,
                         path: List[Tuple[int, int]],
                         look_ahead: int = 3) -> List[SensorObservation]:
        """
        Sense cells along a planned path (limited look-ahead).
        Useful for early detection of path blockages.
        """
        if not path:
            return []
        
        observations = []
        cells_to_sense: Set[Tuple[int, int]] = set()
        
        # Collect cells along path and their neighbors
        for i, (px, py) in enumerate(path[:look_ahead]):
            cells_to_sense.add((px, py))
            # Also sense immediate neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if warehouse.is_within_bounds(nx, ny):
                    cells_to_sense.add((nx, ny))
        
        for x, y in cells_to_sense:
            actual_obstacle = warehouse.grid[y, x] == Warehouse.OBSTACLE
            
            detected = actual_obstacle
            confidence = self.base_confidence
            
            if actual_obstacle and self.rng.random() < self.false_negative_rate:
                detected = False
                confidence *= 0.6
            elif not actual_obstacle and self.rng.random() < self.false_positive_rate:
                detected = True
                confidence *= 0.4
            
            observations.append(SensorObservation(
                x=x, y=y,
                detected_obstacle=detected,
                confidence=confidence
            ))
        
        return observations


class PredictiveRerouting:
    """
    Estimates blockage risk along a path and decides when to replan.
    """
    
    def __init__(self, risk_threshold: float = 0.5):
        self.risk_threshold = risk_threshold
        self.replan_count = 0
        self.replan_reasons: List[Dict] = []
    
    def compute_blockage_risk(self, prob_map: ProbabilisticObstacleMap,
                               path: List[Tuple[int, int]]) -> float:
        """
        Compute overall blockage risk for a path.
        Returns a value between 0 (safe) and 1 (definitely blocked).
        """
        if not path:
            return 0.0
        
        risks = self.path_risk_profile(prob_map, path)
        if not risks:
            return 0.0
        
        # Use max risk along path as the overall risk
        # (one blocked cell blocks the whole path)
        return max(risks)
    
    def path_risk_profile(self, prob_map: ProbabilisticObstacleMap,
                          path: List[Tuple[int, int]]) -> List[float]:
        """
        Compute per-cell risk along the path.
        """
        risks = []
        for x, y in path:
            risk = prob_map.get_probability(x, y)
            risks.append(risk)
        return risks
    
    def should_replan(self, prob_map: ProbabilisticObstacleMap,
                      path: List[Tuple[int, int]],
                      current_index: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Determine if replanning is needed based on path risk.
        Returns (should_replan, reason).
        """
        if not path or current_index >= len(path):
            return False, None
        
        # Only check remaining path
        remaining_path = path[current_index:]
        
        risk = self.compute_blockage_risk(prob_map, remaining_path)
        
        if risk >= self.risk_threshold:
            reason = f"High blockage risk ({risk:.2f}) detected along path"
            return True, reason
        
        # Also check if immediate next step is risky
        if remaining_path:
            next_x, next_y = remaining_path[0]
            next_risk = prob_map.get_probability(next_x, next_y)
            if next_risk >= 0.7:  # Higher threshold for immediate next step
                reason = f"Immediate path blocked (risk={next_risk:.2f})"
                return True, reason
        
        return False, None
    
    def record_replan(self, robot_id: int, reason: str, 
                      old_path: List[Tuple[int, int]],
                      new_path: Optional[List[Tuple[int, int]]]):
        """Record a replanning event for metrics."""
        self.replan_count += 1
        self.replan_reasons.append({
            'robot_id': robot_id,
            'reason': reason,
            'old_path_length': len(old_path) if old_path else 0,
            'new_path_length': len(new_path) if new_path else 0,
            'success': new_path is not None
        })
    
    def get_replan_statistics(self) -> Dict:
        """Get statistics about replanning events."""
        return {
            'total_replans': self.replan_count,
            'successful_replans': sum(1 for r in self.replan_reasons if r['success']),
            'failed_replans': sum(1 for r in self.replan_reasons if not r['success']),
            'reasons': self.replan_reasons
        }
    
    def reset(self):
        """Reset statistics."""
        self.replan_count = 0
        self.replan_reasons = []


class UncertaintyManager:
    """
    Facade class that coordinates all uncertainty-related components.
    """
    
    def __init__(self, warehouse: Warehouse, config: Dict = None):
        if config is None:
            config = {}
        
        self.warehouse = warehouse
        self.enabled = config.get('enabled', True)
        
        # Initialize components
        self.prob_map = ProbabilisticObstacleMap(
            warehouse.width, 
            warehouse.height,
            default_prob=0.0
        )
        self.prob_map.initialize_from_warehouse(warehouse)
        
        self.sensor = SensorModel(
            radius=config.get('sensor_radius', 2),
            false_positive_rate=config.get('false_positive_rate', 0.05),
            false_negative_rate=config.get('false_negative_rate', 0.10),
            base_confidence=config.get('base_confidence', 0.8)
        )
        
        self.rerouter = PredictiveRerouting(
            risk_threshold=config.get('replan_risk_threshold', 0.5)
        )
        
        self.decay_rate = config.get('decay_rate', 0.01)
        self.commit_threshold = config.get('commit_threshold', 0.8)
        self.random_change_rate = config.get('random_change_rate', 0.0)
    
    def update(self, robot_positions: List[Tuple[int, int]]):
        """
        Perform a complete uncertainty update cycle:
        1. Gather sensor observations from all robot positions
        2. Update probability map
        3. Decay old observations
        4. Optionally commit high-confidence predictions
        5. Optionally introduce random changes
        """
        if not self.enabled:
            return
        
        # Gather and process sensor observations
        all_observations = []
        for pos in robot_positions:
            observations = self.sensor.sense_local(self.warehouse, pos)
            all_observations.extend(observations)
        
        # Update probability map
        self.prob_map.update_from_sensor(all_observations)
        
        # Decay uncertainty
        self.prob_map.decay_uncertainty(self.decay_rate)
        
        # Commit high-confidence changes
        if self.commit_threshold < 1.0:
            self.prob_map.commit_to_environment(
                self.warehouse, 
                obstacle_threshold=self.commit_threshold
            )
        
        # Random environment changes (for testing dynamic scenarios)
        if self.random_change_rate > 0:
            self.prob_map.sample_random_changes(self.random_change_rate)
    
    def check_path_risk(self, path: List[Tuple[int, int]], 
                        current_index: int = 0) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a path needs replanning due to risk.
        Returns (needs_replan, risk_value, reason).
        """
        if not self.enabled or not path:
            return False, 0.0, None
        
        risk = self.rerouter.compute_blockage_risk(self.prob_map, path[current_index:])
        needs_replan, reason = self.rerouter.should_replan(self.prob_map, path, current_index)
        
        return needs_replan, risk, reason
    
    def get_risk_augmented_heuristic(self, x: int, y: int, 
                                      base_heuristic: float,
                                      risk_weight: float = 2.0) -> float:
        """
        Augment pathfinding heuristic with risk penalty.
        """
        if not self.enabled:
            return base_heuristic
        
        risk = self.prob_map.get_probability(x, y)
        return base_heuristic + risk_weight * risk
    
    def get_statistics(self) -> Dict:
        """Get uncertainty-related statistics."""
        return {
            'replan_stats': self.rerouter.get_replan_statistics(),
            'avg_uncertainty': float(np.mean(self.prob_map.prob_grid)),
            'max_uncertainty': float(np.max(self.prob_map.prob_grid)),
            'uncertain_cells': int(np.sum((self.prob_map.prob_grid > 0.2) & 
                                          (self.prob_map.prob_grid < 0.8)))
        }

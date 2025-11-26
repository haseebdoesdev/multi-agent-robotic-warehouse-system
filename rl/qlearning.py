"""
Q-Learning RL Integration Module (Module 6)
-------------------------------------------
Implements Q-learning for robot navigation in the warehouse environment.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import pickle
import os


# Action definitions
ACTIONS = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN  
    2: (-1, 0),   # LEFT
    3: (1, 0),    # RIGHT
    4: (0, 0),    # WAIT/STAY
}

ACTION_NAMES = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT', 
    3: 'RIGHT',
    4: 'WAIT'
}


@dataclass
class State:
    """
    State representation for Q-learning with obstacle awareness.
    
    Includes:
    - Robot position
    - Target position
    - Local obstacle information (what's blocked in each direction)
    """
    robot_x: int
    robot_y: int
    target_x: int
    target_y: int
    # Obstacle flags: True if obstacle/wall blocks that direction
    obstacle_up: bool = False
    obstacle_down: bool = False
    obstacle_left: bool = False
    obstacle_right: bool = False
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for Q-table indexing."""
        return (self.robot_x, self.robot_y, self.target_x, self.target_y,
                self.obstacle_up, self.obstacle_down, self.obstacle_left, self.obstacle_right)
    
    @classmethod
    def from_tuple(cls, t: Tuple) -> 'State':
        return cls(
            robot_x=t[0], robot_y=t[1], target_x=t[2], target_y=t[3],
            obstacle_up=t[4] if len(t) > 4 else False,
            obstacle_down=t[5] if len(t) > 5 else False,
            obstacle_left=t[6] if len(t) > 6 else False,
            obstacle_right=t[7] if len(t) > 7 else False
        )
    
    def get_relative_state(self) -> Tuple:
        """
        Get relative state with direction to target AND obstacle info.
        
        Returns: (dir_x, dir_y, obs_up, obs_down, obs_left, obs_right)
        - dir_x: -1 (target left), 0 (same column), 1 (target right)
        - dir_y: -1 (target above), 0 (same row), 1 (target below)
        - obs_*: 1 if blocked, 0 if clear
        
        This gives 9 directions × 16 obstacle combinations = 144 states
        """
        dx = int(np.sign(self.target_x - self.robot_x))
        dy = int(np.sign(self.target_y - self.robot_y))
        return (
            dx, dy,
            int(self.obstacle_up),
            int(self.obstacle_down),
            int(self.obstacle_left),
            int(self.obstacle_right)
        )
    
    def get_obstacle_tuple(self) -> Tuple[int, int, int, int]:
        """Get just the obstacle information as a tuple."""
        return (
            int(self.obstacle_up),
            int(self.obstacle_down),
            int(self.obstacle_left),
            int(self.obstacle_right)
        )


class WarehouseEnv:
    """
    Gymnasium-like environment for warehouse navigation.
    Single-agent environment for training Q-learning.
    """
    
    def __init__(self, warehouse, target_position: Tuple[int, int] = None):
        """
        Initialize the environment.
        
        Args:
            warehouse: Warehouse instance
            target_position: Goal position (package location)
        """
        self.warehouse = warehouse
        self.width = warehouse.width
        self.height = warehouse.height
        
        self.target_position = target_position
        self.robot_position = (0, 0)
        self.done = False
        self.steps = 0
        self.max_steps = self.width * self.height * 2
        
        # Rewards - tuned to discourage waiting and encourage quick paths
        self.reward_reach_target = 100.0
        self.reward_step = -1.0          # Higher step penalty to encourage shorter paths
        self.reward_collision = -10.0
        self.reward_invalid = -5.0
        self.reward_closer = 2.0         # Strong reward for moving closer
        self.reward_farther = -2.0       # Strong penalty for moving away
        self.reward_wait = -5.0          # Heavy penalty for waiting
        self.reward_timeout = -50.0      # Penalty for not reaching target
    
    def reset(self, robot_start: Tuple[int, int] = None,
              target: Tuple[int, int] = None) -> State:
        """
        Reset the environment for a new episode.
        
        Args:
            robot_start: Starting position for robot
            target: Target/goal position
        
        Returns:
            Initial state
        """
        self.done = False
        self.steps = 0
        
        # Set robot start position
        if robot_start is not None:
            self.robot_position = robot_start
        else:
            # Random valid starting position
            self.robot_position = self._random_valid_position()
        
        # Set target position
        if target is not None:
            self.target_position = target
        elif self.warehouse.packages:
            self.target_position = self.warehouse.packages[0]
        else:
            self.target_position = self._random_valid_position()
        
        return self._get_state()
    
    def _random_valid_position(self) -> Tuple[int, int]:
        """Get a random valid (non-obstacle) position."""
        attempts = 0
        while attempts < 100:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.warehouse.is_valid_move(x, y, ignore_packages=True):
                return (x, y)
            attempts += 1
        return (0, 0)
    
    def _get_state(self) -> State:
        """Get current state including obstacle awareness."""
        rx, ry = self.robot_position
        
        # Check what's blocked in each direction
        # UP means y-1, DOWN means y+1, LEFT means x-1, RIGHT means x+1
        obstacle_up = not self._is_valid_position(rx, ry - 1)
        obstacle_down = not self._is_valid_position(rx, ry + 1)
        obstacle_left = not self._is_valid_position(rx - 1, ry)
        obstacle_right = not self._is_valid_position(rx + 1, ry)
        
        return State(
            robot_x=rx,
            robot_y=ry,
            target_x=self.target_position[0],
            target_y=self.target_position[1],
            obstacle_up=obstacle_up,
            obstacle_down=obstacle_down,
            obstacle_left=obstacle_left,
            obstacle_right=obstacle_right
        )
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if a position is valid (in bounds and not obstacle)."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.warehouse.is_valid_move(x, y, ignore_packages=True)
    
    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: Action index (0-4)
        
        Returns:
            (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {'message': 'Episode already done'}
        
        self.steps += 1
        info = {'action': ACTION_NAMES.get(action, 'UNKNOWN')}
        
        # Get action delta
        if action not in ACTIONS:
            return self._get_state(), self.reward_invalid, False, {'message': 'Invalid action'}
        
        dx, dy = ACTIONS[action]
        
        # Check for WAIT action (action 4) - heavily penalize
        if action == 4 or (dx == 0 and dy == 0):
            reward = self.reward_wait
            info['message'] = 'Waiting (penalized)'
            
            # Check max steps with timeout penalty
            if self.steps >= self.max_steps:
                self.done = True
                reward += self.reward_timeout
                info['message'] = 'Timeout - failed to reach target'
            
            return self._get_state(), reward, self.done, info
        
        old_position = self.robot_position
        new_x = self.robot_position[0] + dx
        new_y = self.robot_position[1] + dy
        
        # Calculate distance before move
        old_distance = self._manhattan_distance(old_position, self.target_position)
        
        # Check if move is valid
        if not self.warehouse.is_within_bounds(new_x, new_y):
            reward = self.reward_invalid
            info['message'] = 'Out of bounds'
        elif not self.warehouse.is_valid_move(new_x, new_y, ignore_packages=True):
            reward = self.reward_collision
            info['message'] = 'Collision with obstacle'
        else:
            # Valid move
            self.robot_position = (new_x, new_y)
            
            # Check if reached target
            if self.robot_position == self.target_position:
                # Bonus for reaching faster (fewer steps = higher bonus)
                speed_bonus = max(0, (self.max_steps - self.steps) / self.max_steps * 20)
                reward = self.reward_reach_target + speed_bonus
                self.done = True
                info['message'] = 'Reached target!'
            else:
                # Reward based on distance change
                new_distance = self._manhattan_distance(self.robot_position, self.target_position)
                if new_distance < old_distance:
                    reward = self.reward_step + self.reward_closer
                    info['message'] = 'Moving closer'
                elif new_distance > old_distance:
                    reward = self.reward_step + self.reward_farther
                    info['message'] = 'Moving farther'
                else:
                    reward = self.reward_step
                    info['message'] = 'Same distance'
        
        # Check max steps - apply timeout penalty
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            reward += self.reward_timeout
            info['message'] = 'Timeout - failed to reach target'
        
        return self._get_state(), reward, self.done, info
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self) -> str:
        """Render the environment as a string."""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == self.robot_position:
                    row.append('R')
                elif (x, y) == self.target_position:
                    row.append('T')
                elif self.warehouse.grid[y, x] == 1:  # OBSTACLE
                    row.append('#')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        return '\n'.join(lines)


class QLearningAgent:
    """
    Q-Learning agent for warehouse navigation.
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 use_relative_state: bool = True):
        """
        Initialize Q-learning agent.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            use_relative_state: Use relative state representation (smaller Q-table)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_relative_state = use_relative_state
        
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.n_actions = len(ACTIONS)
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def _get_state_key(self, state: State) -> Tuple:
        """Get the key for Q-table lookup."""
        if self.use_relative_state:
            return state.get_relative_state()
        return state.to_tuple()
    
    def _get_q_values(self, state: State) -> np.ndarray:
        """Get Q-values for a state, initializing if necessary."""
        key = self._get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]
    
    def select_action(self, state: State, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action
            q_values = self._get_q_values(state)
            
            # If all Q-values are zero (unseen state), use heuristic
            if np.all(q_values == 0) and not training:
                return self._heuristic_action(state)
            
            return int(np.argmax(q_values))
    
    def _heuristic_action(self, state: State) -> int:
        """
        Fallback heuristic for unseen states.
        Moves toward target while avoiding obstacles.
        """
        rel = state.get_relative_state()
        dir_x, dir_y = rel[0], rel[1]
        obs_up, obs_down, obs_left, obs_right = rel[2], rel[3], rel[4], rel[5]
        
        # Priority: move in the direction of the target if not blocked
        preferred_actions = []
        
        # Horizontal preference
        if dir_x > 0 and not obs_right:
            preferred_actions.append(3)  # RIGHT
        elif dir_x < 0 and not obs_left:
            preferred_actions.append(2)  # LEFT
        
        # Vertical preference
        if dir_y > 0 and not obs_down:
            preferred_actions.append(1)  # DOWN
        elif dir_y < 0 and not obs_up:
            preferred_actions.append(0)  # UP
        
        # If we have preferred actions, pick one
        if preferred_actions:
            return preferred_actions[0]
        
        # Otherwise, try any unblocked direction
        if not obs_up:
            return 0
        if not obs_down:
            return 1
        if not obs_left:
            return 2
        if not obs_right:
            return 3
        
        # All blocked - wait (shouldn't happen)
        return 4
    
    def update(self, state: State, action: int, reward: float, 
               next_state: State, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
        current_q = self._get_q_values(state)
        next_q = self._get_q_values(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q)
        
        # Q-learning update
        key = self._get_state_key(state)
        self.q_table[key][action] += self.alpha * (target - current_q[action])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, state: State) -> int:
        """Get the best action according to learned policy."""
        return self.select_action(state, training=False)
    
    def save(self, filepath: str):
        """Save the Q-table and parameters to file."""
        data = {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])


def train_agent(warehouse, 
                episodes: int = 1000,
                max_steps_per_episode: int = 200,
                alpha: float = 0.1,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_min: float = 0.01,
                epsilon_decay: float = 0.995,
                verbose: bool = True,
                save_path: str = None) -> QLearningAgent:
    """
    Train a Q-learning agent on the warehouse environment.
    
    Args:
        warehouse: Warehouse instance
        episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration decay rate
        verbose: Print training progress
        save_path: Path to save trained agent
    
    Returns:
        Trained QLearningAgent
    """
    env = WarehouseEnv(warehouse)
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    for episode in range(episodes):
        # Random start and target for diverse training
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.done and steps < max_steps_per_episode:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        agent.decay_epsilon()
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Avg Reward = {avg_reward:.2f}, "
                  f"Avg Length = {avg_length:.1f}, "
                  f"Epsilon = {agent.epsilon:.3f}, "
                  f"Q-table size = {len(agent.q_table)}")
    
    if save_path:
        agent.save(save_path)
        print(f"Agent saved to {save_path}")
    
    return agent


def evaluate_agent(agent: QLearningAgent,
                   warehouse,
                   episodes: int = 100,
                   max_steps: int = 200,
                   verbose: bool = False) -> Dict:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained QLearningAgent
        warehouse: Warehouse instance
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print episode details
    
    Returns:
        Dictionary of evaluation metrics
    """
    env = WarehouseEnv(warehouse)
    
    successes = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while not env.done and steps < max_steps:
            action = agent.get_policy(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done and reward > 0:
                successes += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if verbose:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    return {
        'success_rate': successes / episodes * 100,
        'average_reward': np.mean(total_rewards),
        'average_steps': np.mean(total_steps),
        'min_steps': np.min(total_steps),
        'max_steps': np.max(total_steps),
        'std_reward': np.std(total_rewards)
    }


class RLGuidedPathfinder:
    """
    Hybrid pathfinder that combines A* with RL policy guidance.
    Uses RL to bias action selection when A* path is blocked or uncertain.
    """
    
    def __init__(self, agent: QLearningAgent, warehouse):
        self.agent = agent
        self.warehouse = warehouse
    
    def get_next_action(self, robot_position: Tuple[int, int],
                        target_position: Tuple[int, int],
                        a_star_suggestion: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Get the next action, combining A* and RL.
        
        Args:
            robot_position: Current robot position
            target_position: Goal position
            a_star_suggestion: Suggested next position from A*
        
        Returns:
            Next position to move to
        """
        state = State(
            robot_x=robot_position[0],
            robot_y=robot_position[1],
            target_x=target_position[0],
            target_y=target_position[1]
        )
        
        # If A* has a suggestion and it's valid, prefer it
        if a_star_suggestion is not None:
            if self.warehouse.is_valid_move(a_star_suggestion[0], a_star_suggestion[1], 
                                            ignore_packages=True):
                return a_star_suggestion
        
        # Otherwise use RL policy
        action = self.agent.get_policy(state)
        dx, dy = ACTIONS[action]
        new_pos = (robot_position[0] + dx, robot_position[1] + dy)
        
        # Validate RL suggestion
        if self.warehouse.is_valid_move(new_pos[0], new_pos[1], ignore_packages=True):
            return new_pos
        
        # Fallback: stay in place
        return robot_position


# Convenience function for quick training
def quick_train(warehouse, episodes: int = 500, save_path: str = "rl_agent.pkl") -> QLearningAgent:
    """Quick training with default parameters."""
    return train_agent(
        warehouse=warehouse,
        episodes=episodes,
        verbose=True,
        save_path=save_path
    )

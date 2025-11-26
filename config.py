"""
Configuration module for Multi-Agent Robotic Warehouse System
--------------------------------------------------------------
Contains all configurable parameters for the simulation.
"""

# ==============================================================================
# Core Simulation Settings
# ==============================================================================

GRID_WIDTH = 10
GRID_HEIGHT = 10
OBSTACLE_DENSITY = 0.15

NUM_PACKAGES = 5
NUM_ROBOTS = 2
RANDOM_SEED = 42
MAX_TIMESTEPS = 500

ROBOT_START_POSITIONS = None  # None for default, or list of (x, y) tuples

PACKAGE_LOCATIONS = None  # None for random, or list of (x, y) tuples

OBSTACLE_LOCATIONS = None  # None for random, or list of (x, y) tuples

# ==============================================================================
# Text UI Settings
# ==============================================================================

TEXT_UPDATE_DELAY = 0.5
TEXT_USE_COLORS = True
TEXT_SHOW_PATHS = True

# ==============================================================================
# Pygame UI Settings
# ==============================================================================

PYGAME_WINDOW_SIZE = 700
PYGAME_FPS = 2

# ==============================================================================
# Pathfinding Settings
# ==============================================================================

PATHFINDING_HEURISTIC = 'manhattan'  # 'manhattan' or 'euclidean'

# ==============================================================================
# Conflict Resolution Settings
# ==============================================================================

CONFLICT_WAIT_DURATION = 1

# ==============================================================================
# Output Settings
# ==============================================================================

SHOW_STATISTICS = True
VERBOSE_OUTPUT = True

# ==============================================================================
# Module 4: Uncertainty & Dynamic Adaptation
# ==============================================================================

UNCERTAINTY_ENABLED = False  # Enable probabilistic obstacles and dynamic replanning

# Sensor settings
SENSOR_RADIUS = 2
SENSOR_NOISE = {
    'false_positive_rate': 0.05,
    'false_negative_rate': 0.10,
    'base_confidence': 0.8
}

# Uncertainty map settings
UNCERTAINTY_DECAY_RATE = 0.01
OBSTACLE_COMMIT_THRESHOLD = 0.8  # Probability threshold to commit as obstacle
OBSTACLE_CLEAR_THRESHOLD = 0.2   # Probability threshold to clear obstacle

# Dynamic environment
RANDOM_OBSTACLE_CHANGE_RATE = 0.0  # Rate of random obstacle changes (0 = disabled)

# Replanning settings
REPLAN_RISK_THRESHOLD = 0.5  # Risk threshold to trigger replanning

# ==============================================================================
# Module 5: Metrics & Reporting
# ==============================================================================

METRICS_ENABLED = True  # Enable detailed metrics collection
REPORTS_OUTPUT_DIR = "reports"  # Directory for saving reports
GENERATE_PLOTS = True  # Generate matplotlib plots

# ==============================================================================
# Module 6: RL Integration
# ==============================================================================

RL_ENABLED = False  # Enable RL-guided pathfinding

# Q-Learning parameters
RL_TRAINING_EPISODES = 1000
RL_EPSILON = 0.1  # Exploration rate for trained agent
RL_EPSILON_START = 1.0  # Initial exploration rate for training
RL_EPSILON_MIN = 0.01  # Minimum exploration rate
RL_EPSILON_DECAY = 0.995  # Exploration decay rate
RL_ALPHA = 0.1  # Learning rate
RL_GAMMA = 0.99  # Discount factor

RL_MODEL_PATH = "rl_agent.pkl"  # Path to save/load trained model
RL_USE_RELATIVE_STATE = True  # Use relative state representation

# ==============================================================================
# Module 6: Scalability & Parallel Processing
# ==============================================================================

PARALLEL_COORDINATION_ENABLED = False  # Enable parallel path planning
PARALLEL_WORKERS = 4  # Number of parallel workers

# ==============================================================================
# Preset Configurations
# ==============================================================================

def config_small_test():
    """Small test configuration (5x5 grid)."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    GRID_WIDTH = 5
    GRID_HEIGHT = 5
    NUM_PACKAGES = 2
    NUM_ROBOTS = 2
    OBSTACLE_DENSITY = 0.1


def config_medium():
    """Medium configuration (10x10 grid)."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_PACKAGES = 5
    NUM_ROBOTS = 2
    OBSTACLE_DENSITY = 0.15


def config_large():
    """Large configuration (20x20 grid)."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    NUM_PACKAGES = 10
    NUM_ROBOTS = 3
    OBSTACLE_DENSITY = 0.2


def config_stress():
    """Stress test configuration (30x30 grid)."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    GRID_WIDTH = 30
    GRID_HEIGHT = 30
    NUM_PACKAGES = 15
    NUM_ROBOTS = 3
    OBSTACLE_DENSITY = 0.25


def config_predefined_test():
    """Predefined test scenario with fixed positions."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS
    global ROBOT_START_POSITIONS, PACKAGE_LOCATIONS, OBSTACLE_LOCATIONS
    
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_PACKAGES = 3
    NUM_ROBOTS = 2
    
    ROBOT_START_POSITIONS = [(0, 0), (7, 7)]
    PACKAGE_LOCATIONS = [(3, 3), (5, 2), (2, 6)]
    OBSTACLE_LOCATIONS = [(3, 4), (4, 4), (5, 4), (3, 5), (5, 5)]


def config_uncertainty_demo():
    """Configuration for demonstrating uncertainty features."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    global UNCERTAINTY_ENABLED, RANDOM_OBSTACLE_CHANGE_RATE
    
    GRID_WIDTH = 15
    GRID_HEIGHT = 15
    NUM_PACKAGES = 5
    NUM_ROBOTS = 2
    OBSTACLE_DENSITY = 0.1
    
    UNCERTAINTY_ENABLED = True
    RANDOM_OBSTACLE_CHANGE_RATE = 0.01


def config_rl_training():
    """Configuration for RL training."""
    global GRID_WIDTH, GRID_HEIGHT, NUM_PACKAGES, NUM_ROBOTS, OBSTACLE_DENSITY
    global RL_ENABLED, RL_TRAINING_EPISODES
    
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_PACKAGES = 1
    NUM_ROBOTS = 1
    OBSTACLE_DENSITY = 0.1
    
    RL_ENABLED = True
    RL_TRAINING_EPISODES = 1000


def get_uncertainty_config() -> dict:
    """Get uncertainty configuration as a dictionary."""
    return {
        'enabled': UNCERTAINTY_ENABLED,
        'sensor_radius': SENSOR_RADIUS,
        'false_positive_rate': SENSOR_NOISE['false_positive_rate'],
        'false_negative_rate': SENSOR_NOISE['false_negative_rate'],
        'base_confidence': SENSOR_NOISE['base_confidence'],
        'decay_rate': UNCERTAINTY_DECAY_RATE,
        'commit_threshold': OBSTACLE_COMMIT_THRESHOLD,
        'clear_threshold': OBSTACLE_CLEAR_THRESHOLD,
        'random_change_rate': RANDOM_OBSTACLE_CHANGE_RATE,
        'replan_risk_threshold': REPLAN_RISK_THRESHOLD
    }


def get_rl_config() -> dict:
    """Get RL configuration as a dictionary."""
    return {
        'enabled': RL_ENABLED,
        'training_episodes': RL_TRAINING_EPISODES,
        'epsilon': RL_EPSILON,
        'epsilon_start': RL_EPSILON_START,
        'epsilon_min': RL_EPSILON_MIN,
        'epsilon_decay': RL_EPSILON_DECAY,
        'alpha': RL_ALPHA,
        'gamma': RL_GAMMA,
        'model_path': RL_MODEL_PATH,
        'use_relative_state': RL_USE_RELATIVE_STATE
    }

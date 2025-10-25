# Multi-Agent Robotic Warehouse System

A grid-based multi-agent simulation system demonstrating autonomous robot coordination for package collection in a warehouse environment. This project implements various AI techniques including pathfinding, conflict resolution, and task assignment.

## Team Members

- **Abdul Haseeb** - abdlhaseeb17@gmail.com
- **Raja Ammar** - rajammarahmd@gmail.com

**Course**: AI Lab (5th Semester)  
**Status**: Mid-term Implementation

---

## Features

### Core AI Implementations

- **A* Pathfinding Algorithm**: Efficient navigation with Manhattan and Euclidean heuristics
- **Multi-Agent Coordination**: Manages multiple robots simultaneously with task distribution
- **Conflict Detection & Resolution**: Prevents collisions and handles deadlocks
- **Greedy Task Assignment**: Nearest-package-first assignment strategy
- **Dynamic Replanning**: Robots replan paths when necessary

### System Features

- Grid-based warehouse environment with obstacles
- Multiple autonomous robots operating concurrently
- Real-time package collection and tracking
- Performance statistics and metrics
- Configurable simulation parameters
- Both text-based and graphical visualization modes

---

## Project Structure

```
AI_SEEMSTER_PROJECT_2/
├── warehouse/
│   ├── environment.py      # Grid environment and warehouse logic
│   ├── robot.py            # Robot behavior and state management
│   ├── pathfinding.py      # A* algorithm implementation
│   └── coordination.py     # Multi-agent coordination manager
├── ui/
│   ├── text_ui.py          # Terminal-based visualization
│   └── pygame_ui.py        # Graphical pygame interface
├── config.py               # Configuration presets
├── main.py                 # Text simulation entry point
├── main_gui.py             # GUI simulation entry point
├── test_basic.py           # Unit tests
└── requirements.txt        # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download the project repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The project requires:
- `numpy>=1.21.0`
- `pygame>=2.0.0`

---

## Usage

### Running the Simulation

**Text-Based Mode (Terminal)**:
```bash
python main.py
```

**Graphical Mode (Pygame)**:
```bash
python main_gui.py
```

**Run Tests**:
```bash
python test_basic.py
```

### Configuration

Edit `config.py` to customize simulation parameters:

```python
GRID_WIDTH = 10              # Warehouse width
GRID_HEIGHT = 10             # Warehouse height
OBSTACLE_DENSITY = 0.15      # Percentage of obstacles
NUM_PACKAGES = 5             # Number of packages
NUM_ROBOTS = 2               # Number of robots
MAX_TIMESTEPS = 500          # Maximum simulation steps
```

**Available Presets**:
- `config_small_test()`: 5×5 grid, 2 robots, 2 packages
- `config_medium()`: 10×10 grid, 2 robots, 5 packages (default)
- `config_large()`: 20×20 grid, 3 robots, 10 packages
- `config_stress()`: 30×30 grid, 3 robots, 15 packages
- `config_predefined_test()`: Pre-defined scenario for testing

---

## Technical Details

### Pathfinding Algorithm

The system implements **A* (A-star)** pathfinding with:
- Priority queue-based open set
- Closed set for explored nodes
- Configurable heuristics (Manhattan or Euclidean distance)
- Obstacle avoidance
- Optimal path guarantees

### Multi-Agent Coordination

**Task Assignment**:
- Greedy nearest-package algorithm
- Idle robots are assigned to available packages
- Distance calculated using A* pathfinding

**Conflict Resolution**:
- Detects head-on collisions
- Detects same-position conflicts
- Priority-based waiting mechanism
- Lower ID robots have priority

### Robot States

- `idle`: No assigned task
- `moving`: Following path to target
- `collecting`: At target location
- `completed`: Task finished
- `waiting`: Conflict resolution delay

---

## Performance Metrics

The system tracks and displays:

- **Total timesteps**: Simulation duration
- **Packages collected**: Successfully retrieved packages
- **Total distance**: Combined distance traveled by all robots
- **Conflicts resolved**: Number of collision avoidances
- **Efficiency**: Average moves per package
- **Per-robot statistics**: Individual performance tracking

---

## Controls (Pygame Mode)

- **ESC**: Exit simulation
- **Close Window**: End simulation
- The simulation runs automatically once started

---

## Examples

### Text-Based Output
```
==============================================================
  Multi-Agent Robotic Warehouse - Text Simulation
==============================================================

Warehouse: 10x10
Robots: 2
Packages: 5
Obstacles: ~15%

  R0 R1 .  .  .  .  .  .  .  .
  .  .  .  ## ## .  .  P  .  .
  .  .  .  .  .  .  ## .  .  .
  P  .  ## .  .  .  .  .  P  .
  ...
```

### Statistics Display
```
Total Timesteps: 124
Packages Collected: 5
Total Distance: 67
Conflicts Resolved: 3
Efficiency: 13.40 moves per package

Robot Performance:
  Robot 0: 3 packages, 38 distance
  Robot 1: 2 packages, 29 distance
```

---

## Implementation Highlights

### Environment Module
- NumPy-based grid representation
- Dynamic obstacle and package placement
- Bounds checking and validation
- Cell state management

### Pathfinding Module
- Node class with f-cost (g + h)
- Heap-based priority queue
- Configurable heuristic functions
- Path reconstruction

### Coordination Module
- Centralized robot management
- Conflict detection between all robot pairs
- Statistical tracking
- Time-stepped updates

### Visualization
- Color-coded robots in both modes
- Path visualization (text mode)
- Real-time statistics panel
- Completion summary screen

---

## Testing

The `test_basic.py` file includes verification tests for:

- ✓ Warehouse creation and initialization
- ✓ A* pathfinding correctness
- ✓ Robot creation and state
- ✓ Robot movement mechanics
- ✓ Multi-agent coordination
- ✓ Conflict detection

Run all tests:
```bash
python test_basic.py
```

---

## Future Enhancements (Post Mid-term)

Potential improvements for final submission:
- Advanced task assignment algorithms (e.g., Hungarian algorithm)
- Dynamic obstacle handling
- Multi-goal path planning
- Communication between robots
- Machine learning-based coordination
- Priority-based package collection
- Energy/battery constraints
- Charging station integration

---

## Known Limitations

- Static obstacle configuration after initialization
- Simple priority-based conflict resolution
- No path optimization after initial planning
- Fixed robot starting positions
- Greedy task assignment (not globally optimal)

---

## Dependencies

- **NumPy**: Grid data structures and array operations
- **Pygame**: Graphical visualization and UI

---

## License

Educational project for AI Lab coursework.

---

## Acknowledgments

This project was developed as part of the 5th Semester AI Lab curriculum, implementing concepts from:
- Artificial Intelligence: A Modern Approach (Russell & Norvig)
- Multi-agent systems and coordination
- Search algorithms and heuristics
- Python game development with Pygame

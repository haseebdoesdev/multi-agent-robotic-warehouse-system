"""
Multi-Agent Warehouse - Comprehensive Test Suite
-------------------------------------------------
Tests for all modules including core, uncertainty, metrics, RL, and experiments.
"""

from warehouse import Warehouse, Robot, CoordinationManager, a_star
from warehouse.uncertainty import (
    ProbabilisticObstacleMap, 
    SensorModel, 
    PredictiveRerouting,
    UncertaintyManager,
    SensorObservation
)
from metrics import MetricsCollector, compute_metrics, SimulationTrace, StepData
from rl import QLearningAgent, WarehouseEnv, State, ACTIONS, train_agent


def test_warehouse_creation():
    print("Testing warehouse creation...")
    warehouse = Warehouse(width=10, height=10, obstacle_density=0.15)
    warehouse.place_obstacles()
    warehouse.place_packages(3)
    
    assert warehouse.width == 10
    assert warehouse.height == 10
    assert len(warehouse.packages) == 3
    print("[OK] Warehouse created successfully")


def test_pathfinding():
    print("\nTesting A* pathfinding...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    
    start = (0, 0)
    goal = (4, 4)
    path = a_star(start, goal, warehouse)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) == 9
    print(f"[OK] A* found path of length {len(path)}")


def test_robot_creation():
    print("\nTesting robot creation...")
    robot = Robot(robot_id=0, start_position=(0, 0))
    
    assert robot.id == 0
    assert robot.position == (0, 0)
    assert robot.packages_collected == 0
    print("[OK] Robot created successfully")


def test_robot_movement():
    print("\nTesting robot movement...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    robot = Robot(robot_id=0, start_position=(0, 0))
    
    robot.set_target((2, 2))
    robot.plan_path(warehouse)
    
    assert robot.path is not None
    assert len(robot.path) > 0
    
    initial_pos = robot.position
    robot.move(warehouse)
    
    assert robot.position != initial_pos
    print("[OK] Robot movement successful")


def test_coordination():
    print("\nTesting multi-agent coordination...")
    warehouse = Warehouse(width=10, height=10, obstacle_density=0.1)
    warehouse.place_obstacles()
    warehouse.place_packages(3)
    
    coordinator = CoordinationManager(warehouse)
    
    robot1 = Robot(robot_id=0, start_position=(0, 0))
    robot2 = Robot(robot_id=1, start_position=(9, 9))
    
    coordinator.add_robot(robot1)
    coordinator.add_robot(robot2)
    
    assignments = coordinator.assign_packages(warehouse.packages)
    
    assert len(assignments) <= 2
    print(f"[OK] Coordination manager assigned {len(assignments)} packages")


def test_conflict_detection():
    print("\nTesting conflict detection...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    
    coordinator = CoordinationManager(warehouse)
    
    robot1 = Robot(robot_id=0, start_position=(0, 0))
    robot2 = Robot(robot_id=1, start_position=(2, 0))
    
    robot1.path = [(0, 0), (1, 0), (2, 0)]
    robot1.path_index = 0
    robot2.path = [(2, 0), (1, 0), (0, 0)]
    robot2.path_index = 0
    
    coordinator.add_robot(robot1)
    coordinator.add_robot(robot2)
    
    conflicts = coordinator.detect_conflicts()
    
    assert len(conflicts) > 0
    print(f"[OK] Conflict detection found {len(conflicts)} conflict(s)")


# ============================================================================
# Module 4: Uncertainty Tests
# ============================================================================

def test_probabilistic_obstacle_map():
    print("\nTesting probabilistic obstacle map...")
    prob_map = ProbabilisticObstacleMap(width=10, height=10, default_prob=0.0)
    
    # Test initialization
    assert prob_map.width == 10
    assert prob_map.height == 10
    assert prob_map.get_probability(5, 5) == 0.0
    
    # Test sensor update
    observations = [
        SensorObservation(x=3, y=3, detected_obstacle=True, confidence=0.9),
        SensorObservation(x=4, y=4, detected_obstacle=False, confidence=0.8)
    ]
    prob_map.update_from_sensor(observations)
    
    assert prob_map.get_probability(3, 3) > 0.0  # Should increase
    print("[OK] Probabilistic obstacle map working correctly")


def test_sensor_model():
    print("\nTesting sensor model...")
    warehouse = Warehouse(width=10, height=10, obstacle_density=0.1)
    warehouse.place_obstacles()
    
    sensor = SensorModel(radius=2, false_positive_rate=0.05, false_negative_rate=0.10)
    
    observations = sensor.sense_local(warehouse, robot_position=(5, 5))
    
    assert len(observations) > 0
    assert all(isinstance(obs, SensorObservation) for obs in observations)
    print(f"[OK] Sensor model generated {len(observations)} observations")


def test_predictive_rerouting():
    print("\nTesting predictive rerouting...")
    prob_map = ProbabilisticObstacleMap(width=10, height=10, default_prob=0.0)
    
    # Set high probability for a cell in the path
    prob_map.prob_grid[3, 3] = 0.9
    
    rerouter = PredictiveRerouting(risk_threshold=0.5)
    
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    risk = rerouter.compute_blockage_risk(prob_map, path)
    
    assert risk >= 0.9
    
    needs_replan, reason = rerouter.should_replan(prob_map, path)
    assert needs_replan is True
    print("[OK] Predictive rerouting correctly detects risky paths")


def test_uncertainty_manager():
    print("\nTesting uncertainty manager...")
    warehouse = Warehouse(width=10, height=10, obstacle_density=0.1)
    warehouse.place_obstacles()
    
    config = {
        'enabled': True,
        'sensor_radius': 2,
        'false_positive_rate': 0.05,
        'false_negative_rate': 0.10
    }
    
    mgr = UncertaintyManager(warehouse, config)
    
    # Simulate update cycle
    robot_positions = [(3, 3), (7, 7)]
    mgr.update(robot_positions)
    
    stats = mgr.get_statistics()
    assert 'avg_uncertainty' in stats
    print("[OK] Uncertainty manager integrated correctly")


# ============================================================================
# Module 5: Metrics Tests
# ============================================================================

def test_metrics_collector():
    print("\nTesting metrics collector...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    
    robots = [
        Robot(robot_id=0, start_position=(0, 0)),
        Robot(robot_id=1, start_position=(4, 4))
    ]
    
    collector = MetricsCollector()
    
    # Record a few steps
    for step in range(5):
        collector.record_step(step, robots, warehouse, conflicts_total=step // 2)
    
    trace = collector.get_trace()
    
    assert len(trace.steps) == 5
    assert trace.steps[0].timestep == 0
    print("[OK] Metrics collector recording correctly")


def test_compute_metrics():
    print("\nTesting metrics computation...")
    # Create sample trace
    trace = SimulationTrace()
    for i in range(10):
        step = StepData(
            timestep=i,
            robot_positions=[(i, 0), (0, i)],
            robot_statuses=['moving', 'idle'],
            robot_targets=[(5, 5), (5, 5)],
            packages_remaining=5 - (i // 2),
            packages_collected_this_step=1 if i % 2 == 0 else 0,
            conflicts_this_step=1 if i % 3 == 0 else 0
        )
        trace.add_step(step)
    
    coordinator_stats = {
        'timesteps': 10,
        'total_packages_collected': 5,
        'total_distance_traveled': 20,
        'conflicts_resolved': 4,
        'robots': [
            {'id': 0, 'packages_collected': 3, 'distance_traveled': 12},
            {'id': 1, 'packages_collected': 2, 'distance_traveled': 8}
        ]
    }
    
    metrics = compute_metrics(trace, coordinator_stats)
    
    assert 'efficiency' in metrics
    assert 'timing' in metrics
    assert 'congestion' in metrics
    assert 'utilization' in metrics
    print("[OK] Metrics computation working correctly")


# ============================================================================
# Module 6: RL Tests
# ============================================================================

def test_warehouse_env():
    print("\nTesting warehouse RL environment...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    
    env = WarehouseEnv(warehouse)
    state = env.reset(robot_start=(0, 0), target=(4, 4))
    
    assert state.robot_x == 0
    assert state.robot_y == 0
    assert state.target_x == 4
    assert state.target_y == 4
    
    # Take a step
    next_state, reward, done, info = env.step(3)  # RIGHT
    
    assert next_state.robot_x == 1
    assert not done
    print("[OK] Warehouse RL environment working correctly")


def test_qlearning_agent():
    print("\nTesting Q-learning agent...")
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=0.5
    )
    
    state = State(robot_x=0, robot_y=0, target_x=4, target_y=4)
    
    # Select action
    action = agent.select_action(state, training=True)
    assert 0 <= action < len(ACTIONS)
    
    # Update Q-value
    next_state = State(robot_x=1, robot_y=0, target_x=4, target_y=4)
    agent.update(state, action, -0.1, next_state, done=False)
    
    # Q-table should have entry
    assert len(agent.q_table) > 0
    print("[OK] Q-learning agent working correctly")


def test_rl_training():
    print("\nTesting RL training (quick)...")
    warehouse = Warehouse(width=5, height=5, obstacle_density=0.0)
    
    # Quick training with few episodes
    agent = train_agent(
        warehouse,
        episodes=50,
        max_steps_per_episode=50,
        verbose=False
    )
    
    assert len(agent.episode_rewards) == 50
    print(f"[OK] RL training completed ({len(agent.q_table)} Q-table entries)")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    print("=" * 60)
    print("  Multi-Agent Warehouse - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Core tests
        print("\n--- Core Module Tests ---")
        test_warehouse_creation()
        test_pathfinding()
        test_robot_creation()
        test_robot_movement()
        test_coordination()
        test_conflict_detection()
        
        # Module 4: Uncertainty tests
        print("\n--- Module 4: Uncertainty Tests ---")
        test_probabilistic_obstacle_map()
        test_sensor_model()
        test_predictive_rerouting()
        test_uncertainty_manager()
        
        # Module 5: Metrics tests
        print("\n--- Module 5: Metrics Tests ---")
        test_metrics_collector()
        test_compute_metrics()
        
        # Module 6: RL tests
        print("\n--- Module 6: RL Tests ---")
        test_warehouse_env()
        test_qlearning_agent()
        test_rl_training()
        
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAll modules are working correctly.")
        print("You can now run:")
        print("  - python main.py (text-based simulation)")
        print("  - python main_gui.py (graphical simulation)")
        print("  - python -m experiments.scalability (scalability tests)")
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()

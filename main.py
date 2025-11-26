"""
Multi-Agent Robotic Warehouse - Text Simulation Entry Point
-----------------------------------------------------------
Run with: python main.py
"""

import time
import random
from warehouse import Warehouse, Robot, CoordinationManager
from warehouse.uncertainty import UncertaintyManager
from ui import TextUI
from metrics import MetricsCollector, compute_metrics, format_metrics_summary
from metrics.reporting import generate_report
import config


def get_valid_start_positions(warehouse, preferred_positions, num_robots):
    """Get valid starting positions for robots, avoiding obstacles."""
    valid_positions = []
    used_positions = set()
    
    for i in range(num_robots):
        preferred = preferred_positions[i % len(preferred_positions)]
        
        # Check if preferred position is valid
        if (warehouse.is_valid_move(preferred[0], preferred[1], ignore_packages=True) 
            and preferred not in used_positions):
            valid_positions.append(preferred)
            used_positions.add(preferred)
        else:
            # Find nearest valid position
            found = False
            for radius in range(1, max(warehouse.width, warehouse.height)):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = preferred[0] + dx, preferred[1] + dy
                        if (warehouse.is_valid_move(nx, ny, ignore_packages=True)
                            and (nx, ny) not in used_positions):
                            valid_positions.append((nx, ny))
                            used_positions.add((nx, ny))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # Fallback: find any valid position
                for y in range(warehouse.height):
                    for x in range(warehouse.width):
                        if (warehouse.is_valid_move(x, y, ignore_packages=True)
                            and (x, y) not in used_positions):
                            valid_positions.append((x, y))
                            used_positions.add((x, y))
                            found = True
                            break
                    if found:
                        break
    
    return valid_positions


def run_simulation():
    print("=" * 60)
    print("  Multi-Agent Robotic Warehouse - Text Simulation")
    print("=" * 60)
    print("\nInitializing simulation...\n")
    
    # Load configuration
    GRID_WIDTH = config.GRID_WIDTH
    GRID_HEIGHT = config.GRID_HEIGHT
    OBSTACLE_DENSITY = config.OBSTACLE_DENSITY
    NUM_PACKAGES = config.NUM_PACKAGES
    NUM_ROBOTS = config.NUM_ROBOTS
    UPDATE_DELAY = config.TEXT_UPDATE_DELAY
    
    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
    
    # Create warehouse
    warehouse = Warehouse(width=GRID_WIDTH, height=GRID_HEIGHT, obstacle_density=OBSTACLE_DENSITY)
    warehouse.place_obstacles(predefined=config.OBSTACLE_LOCATIONS)
    warehouse.place_packages(NUM_PACKAGES, locations=config.PACKAGE_LOCATIONS)
    
    # Create robots with validated positions
    robots = []
    if config.ROBOT_START_POSITIONS:
        preferred_positions = config.ROBOT_START_POSITIONS
    else:
        preferred_positions = [(0, 0), (GRID_WIDTH-1, GRID_HEIGHT-1), (0, GRID_HEIGHT-1)]
    
    # Get valid positions that don't overlap with obstacles
    valid_positions = get_valid_start_positions(warehouse, preferred_positions, NUM_ROBOTS)
    
    for i in range(NUM_ROBOTS):
        start_pos = valid_positions[i]
        robot = Robot(robot_id=i, start_position=start_pos)
        robots.append(robot)
    
    # Create coordinator
    coordinator = CoordinationManager(warehouse)
    for robot in robots:
        coordinator.add_robot(robot)
    
    # Initialize uncertainty manager if enabled
    uncertainty_mgr = None
    if config.UNCERTAINTY_ENABLED:
        uncertainty_mgr = UncertaintyManager(warehouse, config.get_uncertainty_config())
        print("Uncertainty module: ENABLED")
    
    # Initialize metrics collector if enabled
    metrics_collector = None
    if config.METRICS_ENABLED:
        metrics_collector = MetricsCollector()
        print("Metrics collection: ENABLED")
    
    # Create UI
    ui = TextUI(warehouse, robots, use_colors=config.TEXT_USE_COLORS)
    
    print(f"\nWarehouse: {GRID_WIDTH}x{GRID_HEIGHT}")
    print(f"Robots: {NUM_ROBOTS}")
    print(f"Packages: {NUM_PACKAGES}")
    print(f"Obstacles: ~{int(OBSTACLE_DENSITY * 100)}%")
    print("\nStarting simulation in 2 seconds...")
    time.sleep(2)
    
    # Simulation loop
    running = True
    max_steps = config.MAX_TIMESTEPS
    
    while running and coordinator.time_step < max_steps:
        # Assign packages to idle robots
        if warehouse.packages:
            coordinator.assign_packages(warehouse.packages)
        
        # Plan paths
        coordinator.plan_all_paths()
        
        # Update uncertainty if enabled
        if uncertainty_mgr:
            robot_positions = [r.position for r in robots]
            uncertainty_mgr.update(robot_positions)
            
            # Check for replanning needs
            for robot in robots:
                if robot.path:
                    needs_replan, risk, reason = uncertainty_mgr.check_path_risk(
                        robot.path, robot.path_index
                    )
                    if needs_replan:
                        robot.plan_path(warehouse)
        
        # Collect metrics
        if metrics_collector:
            metrics_collector.record_step(
                coordinator.time_step,
                robots,
                warehouse,
                coordinator.conflict_count
            )
        
        # Get current statistics
        stats = coordinator.get_statistics()
        
        # Render UI
        ui.render(coordinator.time_step, stats, show_paths=config.TEXT_SHOW_PATHS)
        
        # Check termination conditions
        if coordinator.are_all_packages_collected():
            print("\nAll packages collected!")
            running = False
            break
        
        if coordinator.are_all_robots_idle() and not warehouse.packages:
            running = False
            break
        
        # Update robots
        coordinator.update_robots()
        
        # Delay for visualization
        time.sleep(UPDATE_DELAY)
    
    # Get final statistics
    final_stats = coordinator.get_statistics()
    ui.show_completion_message(final_stats)
    
    # Compute and display detailed metrics
    if metrics_collector and config.METRICS_ENABLED:
        sim_trace = metrics_collector.get_trace()
        
        warehouse_info = {
            'width': GRID_WIDTH,
            'height': GRID_HEIGHT,
            'obstacle_density': OBSTACLE_DENSITY
        }
        
        detailed_metrics = compute_metrics(sim_trace, final_stats, warehouse_info)
        
        print("\n" + format_metrics_summary(detailed_metrics))
        
        # Generate visual reports if enabled
        if config.GENERATE_PLOTS:
            print("\nGenerating reports...")
            report_files = generate_report(
                detailed_metrics, 
                sim_trace, 
                config.REPORTS_OUTPUT_DIR
            )
            if 'error' not in report_files:
                print(f"Reports saved to: {config.REPORTS_OUTPUT_DIR}/")
                for name, path in report_files.items():
                    print(f"  - {name}: {path}")
            else:
                print(f"Note: {report_files.get('error', 'Could not generate visual reports')}")
    else:
        # Basic efficiency calculation
        if final_stats['total_packages_collected'] > 0:
            avg_efficiency = final_stats['total_distance_traveled'] / final_stats['total_packages_collected']
            print(f"\nEfficiency: {avg_efficiency:.2f} moves per package")
    
    print("\nSimulation ended.")


def run_with_rl():
    """Run simulation with RL-guided pathfinding."""
    from rl import train_agent, QLearningAgent, RLGuidedPathfinder
    import os
    
    print("=" * 60)
    print("  Multi-Agent Warehouse - RL Mode")
    print("=" * 60)
    
    # Create warehouse for training
    warehouse = Warehouse(
        width=config.GRID_WIDTH, 
        height=config.GRID_HEIGHT, 
        obstacle_density=config.OBSTACLE_DENSITY
    )
    warehouse.place_obstacles()
    
    # Check if trained model exists
    if os.path.exists(config.RL_MODEL_PATH):
        print(f"\nLoading trained agent from {config.RL_MODEL_PATH}...")
        agent = QLearningAgent()
        agent.load(config.RL_MODEL_PATH)
    else:
        print(f"\nNo trained model found. Training new agent...")
        rl_config = config.get_rl_config()
        agent = train_agent(
            warehouse,
            episodes=rl_config['training_episodes'],
            alpha=rl_config['alpha'],
            gamma=rl_config['gamma'],
            epsilon_start=rl_config['epsilon_start'],
            epsilon_min=rl_config['epsilon_min'],
            epsilon_decay=rl_config['epsilon_decay'],
            verbose=True,
            save_path=config.RL_MODEL_PATH
        )
    
    print("\nRL agent ready. Running simulation with RL guidance...")
    # The standard simulation can now use the trained agent
    run_simulation()


if __name__ == "__main__":
    try:
        # Uncomment the following line to use a different configuration:
        # config.config_large()
        # config.config_uncertainty_demo()
        
        if config.RL_ENABLED:
            run_with_rl()
        else:
            run_simulation()
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()

"""
Multi-Agent Robotic Warehouse - Interactive Pygame GUI
-------------------------------------------------------
Run with: python main_gui.py

CONTROLS:
  SPACE     - Start/Pause simulation
  1         - Obstacle mode (draw walls)
  2         - Package mode (place packages)
  3         - Robot mode (place robots)
  4         - Erase mode
  C         - Clear entire grid
  P         - Toggle path display
  +/-       - Adjust simulation speed
  H         - Show/hide help
  ESC       - Exit

MOUSE:
  Left Click  - Place item
  Right Click - Remove item
  Drag        - Draw obstacles (in obstacle mode)
"""

import random
from warehouse import Warehouse, Robot, CoordinationManager
from warehouse.uncertainty import UncertaintyManager
from ui import PygameUI
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
    """Run the interactive warehouse simulation."""
    # Load configuration
    GRID_SIZE = config.GRID_WIDTH
    OBSTACLE_DENSITY = config.OBSTACLE_DENSITY
    NUM_PACKAGES = config.NUM_PACKAGES
    NUM_ROBOTS = config.NUM_ROBOTS
    WINDOW_SIZE = config.PYGAME_WINDOW_SIZE
    FPS = config.PYGAME_FPS
    
    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
    
    # Create warehouse
    warehouse = Warehouse(width=GRID_SIZE, height=GRID_SIZE, obstacle_density=OBSTACLE_DENSITY)
    warehouse.place_obstacles(predefined=config.OBSTACLE_LOCATIONS)
    warehouse.place_packages(NUM_PACKAGES, locations=config.PACKAGE_LOCATIONS)
    
    # Create robots with validated positions
    robots = []
    if config.ROBOT_START_POSITIONS:
        preferred_positions = config.ROBOT_START_POSITIONS
    else:
        preferred_positions = [(0, 0), (GRID_SIZE-1, GRID_SIZE-1), (0, GRID_SIZE-1)]
    
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
    
    # Initialize metrics collector if enabled
    metrics_collector = None
    if config.METRICS_ENABLED:
        metrics_collector = MetricsCollector()
    
    # Create interactive UI
    ui = PygameUI(warehouse, robots, window_size=WINDOW_SIZE, fps=FPS)
    
    print("=" * 60)
    print("  Multi-Agent Robotic Warehouse - Interactive Mode")
    print("=" * 60)
    print(f"\nWarehouse: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Robots: {NUM_ROBOTS}")
    print(f"Packages: {NUM_PACKAGES}")
    print(f"Obstacles: ~{int(OBSTACLE_DENSITY * 100)}%")
    print("\n" + "-" * 60)
    print("CONTROLS:")
    print("  SPACE - Start/Pause simulation")
    print("  . (period) - Step once when paused")
    print("  1/2/3/4 - Switch modes (Obstacle/Package/Robot/Erase)")
    print("  C - Clear grid  |  P - Toggle paths  |  H - Help")
    print("  +/- - Adjust speed  |  S - Settings  |  ESC - Exit")
    print("-" * 60)
    print("\nDraw your warehouse layout, then press SPACE to start!")
    
    # Main loop
    running = True
    max_steps = config.MAX_TIMESTEPS
    simulation_complete = False
    
    while running:
        # Handle events
        if not ui.handle_events():
            break
        
        # Check if robots were added/removed via UI
        if len(ui.robots) != len(coordinator.robots):
            # Rebuild coordinator with new robot list
            coordinator = CoordinationManager(warehouse)
            for robot in ui.robots:
                coordinator.add_robot(robot)
        
        # Only run simulation logic when not paused OR step mode
        should_run = (not ui.is_paused()) or ui.should_step()
        if should_run and not simulation_complete:
            # Assign packages
            if warehouse.packages:
                coordinator.assign_packages(warehouse.packages)
            
            # Plan paths
            coordinator.plan_all_paths()
            
            # Update uncertainty if enabled
            if uncertainty_mgr:
                robot_positions = [r.position for r in ui.robots]
                uncertainty_mgr.update(robot_positions)
                
                for robot in ui.robots:
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
                    ui.robots,
                    warehouse,
                    coordinator.conflict_count
                )
            
            # Check termination
            if coordinator.are_all_packages_collected() and len(warehouse.packages) == 0:
                simulation_complete = True
            elif coordinator.are_all_robots_idle() and not warehouse.packages and coordinator.time_step > 0:
                simulation_complete = True
            elif coordinator.time_step >= max_steps:
                simulation_complete = True
            
            # Update robots
            if not simulation_complete:
                coordinator.update_robots()
        
        # Get statistics and render
        stats = coordinator.get_statistics()
        ui.render(coordinator.time_step, stats)
        
        # Show completion message
        if simulation_complete:
            final_stats = coordinator.get_statistics()
            restart = ui.show_completion_message(final_stats)
            
            if restart:
                # Reset for new simulation
                simulation_complete = False
                coordinator.reset()
                for robot in ui.robots:
                    robot.status = Robot.STATUS_IDLE
                    robot.path = []
                    robot.path_index = 0
                    robot.target_package = None
                if metrics_collector:
                    metrics_collector.reset()
            else:
                break
    
    # Display final statistics
    if coordinator.time_step > 0:
        final_stats = coordinator.get_statistics()
        
        print("\n" + "=" * 60)
        print("  FINAL STATISTICS")
        print("=" * 60)
        print(f"Total Timesteps: {final_stats['timesteps']}")
        print(f"Packages Collected: {final_stats['total_packages_collected']}")
        print(f"Total Distance: {final_stats['total_distance_traveled']}")
        print(f"Conflicts Resolved: {final_stats['conflicts_resolved']}")
        
        if final_stats['total_packages_collected'] > 0:
            avg_efficiency = final_stats['total_distance_traveled'] / final_stats['total_packages_collected']
            print(f"Efficiency: {avg_efficiency:.2f} moves per package")
        
        print("\nRobot Performance:")
        for robot_info in final_stats['robots']:
            print(f"  Robot {robot_info['id']}: "
                  f"{robot_info['packages_collected']} packages, "
                  f"{robot_info['distance_traveled']} distance")
        
        # Generate reports if metrics were collected
        if metrics_collector and config.METRICS_ENABLED:
            sim_trace = metrics_collector.get_trace()
            
            warehouse_info = {
                'width': GRID_SIZE,
                'height': GRID_SIZE,
                'obstacle_density': OBSTACLE_DENSITY
            }
            
            detailed_metrics = compute_metrics(sim_trace, final_stats, warehouse_info)
            print("\n" + format_metrics_summary(detailed_metrics))
            
            if config.GENERATE_PLOTS:
                print("\nGenerating reports...")
                report_files = generate_report(
                    detailed_metrics,
                    sim_trace,
                    config.REPORTS_OUTPUT_DIR
                )
                if 'error' not in report_files:
                    print(f"Reports saved to: {config.REPORTS_OUTPUT_DIR}/")
    
    ui.quit()
    print("\nSimulation ended.")


if __name__ == "__main__":
    try:
        run_simulation()
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()

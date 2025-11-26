"""
Multi-Agent Robotic Warehouse - Pygame GUI Simulation Entry Point
------------------------------------------------------------------
Run with: python main_gui.py
"""

import random
from warehouse import Warehouse, Robot, CoordinationManager
from warehouse.uncertainty import UncertaintyManager
from ui import PygameUI
from metrics import MetricsCollector, compute_metrics, format_metrics_summary
from metrics.reporting import generate_report
import config


def run_simulation():
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
    
    # Create robots
    robots = []
    if config.ROBOT_START_POSITIONS:
        robot_start_positions = config.ROBOT_START_POSITIONS
    else:
        robot_start_positions = [(0, 0), (GRID_SIZE-1, GRID_SIZE-1), (0, GRID_SIZE-1)]
    
    for i in range(NUM_ROBOTS):
        start_pos = robot_start_positions[i % len(robot_start_positions)]
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
    
    # Create UI
    ui = PygameUI(warehouse, robots, window_size=WINDOW_SIZE, fps=FPS)
    
    print("=" * 60)
    print("  Multi-Agent Robotic Warehouse - Pygame Simulation")
    print("=" * 60)
    print(f"\nWarehouse: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Robots: {NUM_ROBOTS}")
    print(f"Packages: {NUM_PACKAGES}")
    print(f"Obstacles: ~{int(OBSTACLE_DENSITY * 100)}%")
    if config.UNCERTAINTY_ENABLED:
        print("Uncertainty: ENABLED")
    if config.METRICS_ENABLED:
        print("Metrics: ENABLED")
    print("\nSimulation started. Close window or press ESC to exit.")
    
    # Simulation loop
    running = True
    max_steps = config.MAX_TIMESTEPS
    
    while running and coordinator.time_step < max_steps:
        # Handle pygame events
        if not ui.handle_events():
            break
        
        # Assign packages
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
        
        # Get statistics and render
        stats = coordinator.get_statistics()
        ui.render(coordinator.time_step, stats)
        
        # Check termination
        if coordinator.are_all_packages_collected():
            print("\nAll packages collected!")
            running = False
            
            final_stats = coordinator.get_statistics()
            ui.show_completion_message(final_stats)
            break
        
        if coordinator.are_all_robots_idle() and not warehouse.packages:
            running = False
            
            final_stats = coordinator.get_statistics()
            ui.show_completion_message(final_stats)
            break
        
        # Update robots
        coordinator.update_robots()
    
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
        
        # Compute and display detailed metrics
        if metrics_collector and config.METRICS_ENABLED:
            sim_trace = metrics_collector.get_trace()
            
            warehouse_info = {
                'width': GRID_SIZE,
                'height': GRID_SIZE,
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
                else:
                    print(f"Note: {report_files.get('error', 'Could not generate visual reports')}")
    
    ui.quit()
    print("\nSimulation ended.")


if __name__ == "__main__":
    try:
        # Uncomment to use different configurations:
        # config.config_large()
        # config.config_uncertainty_demo()
        
        run_simulation()
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()

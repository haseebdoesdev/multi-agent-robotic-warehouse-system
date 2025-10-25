import time
import random
from warehouse import Warehouse, Robot, CoordinationManager
from ui import TextUI
import config


def run_simulation():
    print("=" * 60)
    print("  Multi-Agent Robotic Warehouse - Text Simulation")
    print("=" * 60)
    print("\nInitializing simulation...\n")
    
    GRID_WIDTH = config.GRID_WIDTH
    GRID_HEIGHT = config.GRID_HEIGHT
    OBSTACLE_DENSITY = config.OBSTACLE_DENSITY
    NUM_PACKAGES = config.NUM_PACKAGES
    NUM_ROBOTS = config.NUM_ROBOTS
    UPDATE_DELAY = config.TEXT_UPDATE_DELAY
    
    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
    
    warehouse = Warehouse(width=GRID_WIDTH, height=GRID_HEIGHT, obstacle_density=OBSTACLE_DENSITY)
    warehouse.place_obstacles(predefined=config.OBSTACLE_LOCATIONS)
    warehouse.place_packages(NUM_PACKAGES, locations=config.PACKAGE_LOCATIONS)
    
    robots = []
    if config.ROBOT_START_POSITIONS:
        robot_start_positions = config.ROBOT_START_POSITIONS
    else:
        robot_start_positions = [(0, 0), (GRID_WIDTH-1, GRID_HEIGHT-1), (0, GRID_HEIGHT-1)]
    
    for i in range(NUM_ROBOTS):
        start_pos = robot_start_positions[i % len(robot_start_positions)]
        robot = Robot(robot_id=i, start_position=start_pos)
        robots.append(robot)
    
    coordinator = CoordinationManager(warehouse)
    for robot in robots:
        coordinator.add_robot(robot)
    
    ui = TextUI(warehouse, robots, use_colors=config.TEXT_USE_COLORS)
    
    print(f"Warehouse: {GRID_WIDTH}x{GRID_HEIGHT}")
    print(f"Robots: {NUM_ROBOTS}")
    print(f"Packages: {NUM_PACKAGES}")
    print(f"Obstacles: ~{int(OBSTACLE_DENSITY * 100)}%")
    print("\nStarting simulation in 2 seconds...")
    time.sleep(2)
    
    running = True
    max_steps = config.MAX_TIMESTEPS
    
    while running and coordinator.time_step < max_steps:
        if warehouse.packages:
            coordinator.assign_packages(warehouse.packages)
        
        coordinator.plan_all_paths()
        
        stats = coordinator.get_statistics()
        
        ui.render(coordinator.time_step, stats, show_paths=config.TEXT_SHOW_PATHS)
        
        if coordinator.are_all_packages_collected():
            print("All packages collected!")
            running = False
            break
        
        if coordinator.are_all_robots_idle() and not warehouse.packages:
            running = False
            break
        
        coordinator.update_robots()
        
        time.sleep(UPDATE_DELAY)
    
    final_stats = coordinator.get_statistics()
    ui.show_completion_message(final_stats)
    
    if final_stats['total_packages_collected'] > 0:
        avg_efficiency = final_stats['total_distance_traveled'] / final_stats['total_packages_collected']
        print(f"Efficiency: {avg_efficiency:.2f} moves per package")
    
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

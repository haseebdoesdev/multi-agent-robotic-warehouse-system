import random
from warehouse import Warehouse, Robot, CoordinationManager
from ui import PygameUI


def run_simulation():
    GRID_SIZE = 20
    OBSTACLE_DENSITY = 0.15
    NUM_PACKAGES = 5
    NUM_ROBOTS = 3
    WINDOW_SIZE = 700
    FPS = 4
    
    random.seed(42)
    
    warehouse = Warehouse(width=GRID_SIZE, height=GRID_SIZE, obstacle_density=OBSTACLE_DENSITY)
    warehouse.place_obstacles()
    warehouse.place_packages(NUM_PACKAGES)
    
    robots = []
    robot_start_positions = [(0, 0), (GRID_SIZE-1, GRID_SIZE-1), (0, GRID_SIZE-1)]
    
    for i in range(NUM_ROBOTS):
        start_pos = robot_start_positions[i % len(robot_start_positions)]
        robot = Robot(robot_id=i, start_position=start_pos)
        robots.append(robot)
    
    coordinator = CoordinationManager(warehouse)
    for robot in robots:
        coordinator.add_robot(robot)
    
    ui = PygameUI(warehouse, robots, window_size=WINDOW_SIZE, fps=FPS)
    
    print("=" * 60)
    print("  Multi-Agent Robotic Warehouse - Pygame Simulation")
    print("=" * 60)
    print(f"\nWarehouse: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Robots: {NUM_ROBOTS}")
    print(f"Packages: {NUM_PACKAGES}")
    print(f"Obstacles: ~{int(OBSTACLE_DENSITY * 100)}%")
    print("\nSimulation started. Close window or press ESC to exit.")
    
    running = True
    max_steps = 500
    
    while running and coordinator.time_step < max_steps:
        if not ui.handle_events():
            break
        
        if warehouse.packages:
            coordinator.assign_packages(warehouse.packages)
        
        coordinator.plan_all_paths()
        
        stats = coordinator.get_statistics()
        
        ui.render(coordinator.time_step, stats)
        
        if coordinator.are_all_packages_collected():
            print("All packages collected!")
            running = False
            
            final_stats = coordinator.get_statistics()
            ui.show_completion_message(final_stats)
            break
        
        if coordinator.are_all_robots_idle() and not warehouse.packages:
            running = False
            
            final_stats = coordinator.get_statistics()
            ui.show_completion_message(final_stats)
            break
        
        coordinator.update_robots()
    
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

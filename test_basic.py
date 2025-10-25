from warehouse import Warehouse, Robot, CoordinationManager, a_star


def test_warehouse_creation():
    print("Testing warehouse creation...")
    warehouse = Warehouse(width=10, height=10, obstacle_density=0.15)
    warehouse.place_obstacles()
    warehouse.place_packages(3)
    
    assert warehouse.width == 10
    assert warehouse.height == 10
    assert len(warehouse.packages) == 3
    print("✓ Warehouse created successfully")


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
    print(f"✓ A* found path of length {len(path)}")


def test_robot_creation():
    print("\nTesting robot creation...")
    robot = Robot(robot_id=0, start_position=(0, 0))
    
    assert robot.id == 0
    assert robot.position == (0, 0)
    assert robot.packages_collected == 0
    print("✓ Robot created successfully")


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
    print("✓ Robot movement successful")


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
    print(f"✓ Coordination manager assigned {len(assignments)} packages")


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
    print(f"✓ Conflict detection found {len(conflicts)} conflict(s)")


def run_all_tests():
    print("=" * 60)
    print("  Multi-Agent Warehouse - Basic Verification Tests")
    print("=" * 60)
    
    try:
        test_warehouse_creation()
        test_pathfinding()
        test_robot_creation()
        test_robot_movement()
        test_coordination()
        test_conflict_detection()
        
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe system is working correctly.")
        print("You can now run:")
        print("  - python main.py (text-based simulation)")
        print("  - python main_gui.py (graphical simulation)")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()

"""
RL Logic Verification Script
-----------------------------
Comprehensive test of all RL components.
"""

import os
import sys


def main():
    print("=" * 60)
    print("  RL LOGIC VERIFICATION")
    print("=" * 60)
    
    errors = []
    
    # Test 1: Basic imports
    print("\n[1] Testing imports...")
    try:
        from warehouse import Warehouse
        from rl import QLearningAgent, WarehouseEnv, State, ACTIONS, ACTION_NAMES, train_agent, evaluate_agent
        print("    All imports OK")
    except Exception as e:
        errors.append(f"Import error: {e}")
        print(f"    FAILED: {e}")
        return 1
    
    # Test 2: State representation
    print("\n[2] Testing State representation...")
    try:
        # State with obstacle awareness
        state = State(robot_x=2, robot_y=3, target_x=5, target_y=1,
                      obstacle_up=False, obstacle_down=True, obstacle_left=False, obstacle_right=False)
        print(f"    Full state tuple: {state.to_tuple()}")
        print(f"    Relative state (with obstacles): {state.get_relative_state()}")
        
        # Target is at (5,1), robot at (2,3)
        # dx = sign(5-2) = 1 (right)
        # dy = sign(1-3) = -1 (up)
        # obstacles = (0, 1, 0, 0) for up=False, down=True, left=False, right=False
        rel_state = state.get_relative_state()
        assert rel_state[:2] == (1, -1), f"Direction should be (1, -1), got {rel_state[:2]}"
        assert rel_state[2:] == (0, 1, 0, 0), f"Obstacles should be (0, 1, 0, 0), got {rel_state[2:]}"
        print("    State representation OK (now includes obstacle awareness!)")
    except Exception as e:
        errors.append(f"State error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 3: Environment
    print("\n[3] Testing WarehouseEnv...")
    try:
        warehouse = Warehouse(5, 5, obstacle_density=0)
        env = WarehouseEnv(warehouse)
        state = env.reset(robot_start=(0, 0), target=(4, 4))
        print(f"    Initial robot pos: ({state.robot_x}, {state.robot_y})")
        print(f"    Target pos: ({state.target_x}, {state.target_y})")
        
        # Take a step RIGHT (action 3)
        next_state, reward, done, info = env.step(3)
        print(f"    After RIGHT: pos=({next_state.robot_x}, {next_state.robot_y}), reward={reward:.2f}")
        assert next_state.robot_x == 1, f"Movement failed! Expected x=1, got x={next_state.robot_x}"
        assert reward > 0, f"Should get positive reward for moving closer, got {reward}"
        print("    Environment OK")
    except Exception as e:
        errors.append(f"Environment error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 4: Q-Learning Agent
    print("\n[4] Testing QLearningAgent...")
    try:
        agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.5)
        state = State(0, 0, 4, 4)
        
        # Select action
        action = agent.select_action(state, training=True)
        print(f"    Selected action: {ACTION_NAMES[action]}")
        
        # Update Q-value
        next_state = State(1, 0, 4, 4)
        agent.update(state, action, 0.5, next_state, done=False)
        print(f"    Q-table size after update: {len(agent.q_table)}")
        assert len(agent.q_table) >= 1, "Q-table should have entries"
        print("    Agent update OK")
    except Exception as e:
        errors.append(f"Agent error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 5: Q-Learning update formula verification
    print("\n[5] Testing Q-Learning update formula...")
    try:
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)
        state = State(0, 0, 2, 2)
        next_state = State(1, 0, 2, 2)
        
        # Initial Q-value should be 0
        q_before = agent._get_q_values(state)[0]
        print(f"    Q(s, UP) before: {q_before}")
        
        # Update: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        # Q(s,a) = 0 + 0.5 * (10 + 0.9 * 0 - 0) = 5.0
        agent.update(state, 0, 10.0, next_state, done=False)
        q_after = agent._get_q_values(state)[0]
        print(f"    Q(s, UP) after reward=10: {q_after}")
        
        expected = 5.0  # 0 + 0.5 * (10 + 0.9*0 - 0)
        assert abs(q_after - expected) < 0.01, f"Expected {expected}, got {q_after}"
        print("    Q-Learning formula OK")
    except Exception as e:
        errors.append(f"Q-formula error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 6: Epsilon-greedy policy
    print("\n[6] Testing epsilon-greedy policy...")
    try:
        agent = QLearningAgent(epsilon=1.0)  # Always explore
        state = State(0, 0, 5, 5)
        
        # With epsilon=1.0, should get variety of actions
        actions = set()
        for _ in range(50):
            actions.add(agent.select_action(state, training=True))
        print(f"    With epsilon=1.0: got {len(actions)} different actions")
        assert len(actions) > 1, "Should explore multiple actions"
        
        # With epsilon=0.0, should be greedy
        agent.epsilon = 0.0
        # First set a clear best action
        agent.q_table[state.get_relative_state()] = [0, 0, 0, 10, 0]  # RIGHT is best
        
        greedy_actions = set()
        for _ in range(20):
            greedy_actions.add(agent.select_action(state, training=True))
        print(f"    With epsilon=0.0: got {len(greedy_actions)} different actions")
        assert len(greedy_actions) == 1, "Should always pick best action"
        assert 3 in greedy_actions, "Should pick RIGHT (action 3)"
        print("    Epsilon-greedy policy OK")
    except Exception as e:
        errors.append(f"Policy error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 7: Training loop
    print("\n[7] Testing training loop (100 episodes)...")
    try:
        warehouse = Warehouse(8, 8, obstacle_density=0.1)
        warehouse.place_obstacles()
        
        trained_agent = train_agent(
            warehouse,
            episodes=100,
            max_steps_per_episode=100,
            verbose=False
        )
        print(f"    Episodes trained: {len(trained_agent.episode_rewards)}")
        print(f"    Q-table size: {len(trained_agent.q_table)}")
        print(f"    Final epsilon: {trained_agent.epsilon:.4f}")
        
        assert len(trained_agent.episode_rewards) == 100, "Should have 100 episode rewards"
        assert trained_agent.epsilon < 1.0, "Epsilon should have decayed"
        print("    Training loop OK")
    except Exception as e:
        errors.append(f"Training error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 8: Evaluation
    print("\n[8] Testing evaluation...")
    try:
        results = evaluate_agent(trained_agent, warehouse, episodes=30, verbose=False)
        print(f"    Success rate: {results['success_rate']:.1f}%")
        print(f"    Avg steps: {results['average_steps']:.1f}")
        print(f"    Avg reward: {results['average_reward']:.1f}")
        
        assert 'success_rate' in results, "Missing success_rate"
        assert 'average_steps' in results, "Missing average_steps"
        print("    Evaluation OK")
    except Exception as e:
        errors.append(f"Evaluation error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 9: Save/Load
    print("\n[9] Testing save/load...")
    try:
        trained_agent.save('test_agent_verify.pkl')
        loaded_agent = QLearningAgent()
        loaded_agent.load('test_agent_verify.pkl')
        print(f"    Loaded Q-table size: {len(loaded_agent.q_table)}")
        
        assert len(loaded_agent.q_table) == len(trained_agent.q_table), "Q-table size mismatch"
        
        # Verify Q-values match
        for key in trained_agent.q_table:
            original = trained_agent.q_table[key]
            loaded = loaded_agent.q_table[key]
            assert all(abs(o - l) < 0.001 for o, l in zip(original, loaded)), "Q-values mismatch"
        
        os.remove('test_agent_verify.pkl')
        print("    Save/Load OK")
    except Exception as e:
        errors.append(f"Save/Load error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 10: RL GUI module
    print("\n[10] Testing RL GUI module...")
    try:
        from ui.rl_gui import RLTrainingUI
        print("    RLTrainingUI imported OK")
    except Exception as e:
        errors.append(f"GUI import error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 11: Reward shaping
    print("\n[11] Testing reward shaping...")
    try:
        warehouse = Warehouse(5, 5, obstacle_density=0)
        env = WarehouseEnv(warehouse)
        env.reset(robot_start=(2, 2), target=(4, 4))
        
        # Move closer to target
        _, reward_closer, _, _ = env.step(3)  # RIGHT
        print(f"    Reward for moving closer: {reward_closer:.2f}")
        
        env.reset(robot_start=(2, 2), target=(4, 4))
        # Move away from target  
        _, reward_farther, _, _ = env.step(2)  # LEFT
        print(f"    Reward for moving away: {reward_farther:.2f}")
        
        assert reward_closer > reward_farther, "Moving closer should give higher reward"
        
        # Reach target
        env.reset(robot_start=(3, 4), target=(4, 4))
        _, reward_target, done, _ = env.step(3)  # RIGHT to target
        print(f"    Reward for reaching target: {reward_target:.2f}")
        assert reward_target > 50, "Should get large reward for target"
        assert done, "Episode should be done"
        print("    Reward shaping OK")
    except Exception as e:
        errors.append(f"Reward error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 12: Collision penalty
    print("\n[12] Testing collision penalty...")
    try:
        warehouse = Warehouse(5, 5, obstacle_density=0)
        warehouse.grid[2, 3] = Warehouse.OBSTACLE  # Place obstacle at (3, 2)
        env = WarehouseEnv(warehouse)
        env.reset(robot_start=(2, 2), target=(4, 4))
        
        # Try to move into obstacle (RIGHT)
        old_pos = env.robot_position
        _, reward_collision, _, info = env.step(3)
        print(f"    Reward for hitting obstacle: {reward_collision:.2f}")
        print(f"    Message: {info['message']}")
        
        assert reward_collision < 0, "Should get negative reward for collision"
        assert env.robot_position == old_pos, "Should not move into obstacle"
        print("    Collision penalty OK")
    except Exception as e:
        errors.append(f"Collision error: {e}")
        print(f"    FAILED: {e}")
    
    # Test 13: Obstacle awareness in state
    print("\n[13] Testing obstacle awareness in state...")
    try:
        warehouse = Warehouse(5, 5, obstacle_density=0)
        # Place obstacles around robot position (2, 2)
        warehouse.grid[1, 2] = Warehouse.OBSTACLE  # Obstacle at (2, 1) - UP from robot
        warehouse.grid[2, 3] = Warehouse.OBSTACLE  # Obstacle at (3, 2) - RIGHT from robot
        
        env = WarehouseEnv(warehouse)
        state = env.reset(robot_start=(2, 2), target=(4, 4))
        
        print(f"    Robot at ({state.robot_x}, {state.robot_y})")
        print(f"    Obstacle up: {state.obstacle_up}")
        print(f"    Obstacle down: {state.obstacle_down}")
        print(f"    Obstacle left: {state.obstacle_left}")
        print(f"    Obstacle right: {state.obstacle_right}")
        
        # UP (y-1 = 1) should be blocked
        assert state.obstacle_up == True, "UP should be blocked"
        # DOWN (y+1 = 3) should be clear
        assert state.obstacle_down == False, "DOWN should be clear"
        # LEFT (x-1 = 1) should be clear
        assert state.obstacle_left == False, "LEFT should be clear"
        # RIGHT (x+1 = 3) should be blocked
        assert state.obstacle_right == True, "RIGHT should be blocked"
        
        # Check relative state includes obstacles
        rel_state = state.get_relative_state()
        print(f"    Relative state: {rel_state}")
        assert len(rel_state) == 6, "Relative state should have 6 elements (dir_x, dir_y, obs_up, obs_down, obs_left, obs_right)"
        
        print("    Obstacle awareness OK - agent now knows what's blocked!")
    except Exception as e:
        errors.append(f"Obstacle awareness error: {e}")
        print(f"    FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  TESTS COMPLETED WITH {len(errors)} ERROR(S)")
        print("=" * 60)
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("  ALL RL TESTS PASSED!")
        print("=" * 60)
        print("\n  The RL implementation is working correctly.")
        print("  Run 'python rl_demo.py' to see it in action!")
        return 0


if __name__ == "__main__":
    sys.exit(main())




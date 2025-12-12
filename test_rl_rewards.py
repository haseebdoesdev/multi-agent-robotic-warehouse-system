"""Test improved RL reward structure."""

from warehouse import Warehouse
from rl import QLearningAgent, WarehouseEnv, State, ACTION_NAMES, train_agent, evaluate_agent
import numpy as np

print("Testing improved reward structure...")
print()

# Test 1: Verify reward values
warehouse = Warehouse(8, 8, obstacle_density=0)
env = WarehouseEnv(warehouse)
print("Reward values:")
print(f"  Reach target: +{env.reward_reach_target}")
print(f"  Move closer:  +{env.reward_closer} (+ step penalty {env.reward_step})")
print(f"  Move away:    {env.reward_farther} (+ step penalty {env.reward_step})")
print(f"  Wait:         {env.reward_wait}")
print(f"  Timeout:      {env.reward_timeout}")
print()

# Test 2: Verify WAIT is heavily penalized
state = env.reset(robot_start=(2, 2), target=(5, 5))
_, reward_wait, _, info = env.step(4)  # WAIT
print(f"WAIT action reward: {reward_wait} - {info['message']}")

# Test 3: Verify moving closer is rewarded
env.reset(robot_start=(2, 2), target=(5, 5))
_, reward_right, _, info = env.step(3)  # RIGHT (closer)
print(f"Moving closer reward: {reward_right} - {info['message']}")

# Test 4: Verify moving away is penalized
env.reset(robot_start=(2, 2), target=(5, 5))
_, reward_left, _, info = env.step(2)  # LEFT (farther)
print(f"Moving away reward: {reward_left} - {info['message']}")

# Test 5: Train and evaluate
print()
print("Training agent for 500 episodes...")
warehouse = Warehouse(8, 8, obstacle_density=0.1)
warehouse.place_obstacles()

agent = train_agent(warehouse, episodes=500, verbose=False)
agent.epsilon = 0.0  # No exploration for testing

# Check what the agent learned
print()
print("Q-table after training:")
for key, vals in agent.q_table.items():
    best_idx = np.argmax(vals)
    wait_val = vals[4]
    print(f"  State {key}: best={ACTION_NAMES[best_idx]:5s}, WAIT Q={wait_val:+.2f}, best Q={vals[best_idx]:+.2f}")

# Verify WAIT is never the best action
all_best = [np.argmax(vals) for vals in agent.q_table.values()]
wait_count = all_best.count(4)
print(f"\nWAIT is best action in {wait_count}/{len(all_best)} states")

# Evaluate
print()
print("Evaluating trained agent (50 episodes)...")
results = evaluate_agent(agent, warehouse, episodes=50, verbose=False)
print(f"  Success rate: {results['success_rate']:.1f}%")
print(f"  Avg steps: {results['average_steps']:.1f}")
print(f"  Avg reward: {results['average_reward']:.1f}")

# Watch one episode
print()
print("Watching one test episode:")
env2 = WarehouseEnv(warehouse)
state = env2.reset()
print(f"  Start: robot={env2.robot_position}, target={env2.target_position}")

for step in range(20):
    action = agent.get_policy(state)
    next_state, reward, done, info = env2.step(action)
    print(f"  Step {step+1}: {ACTION_NAMES[action]:5s} -> pos={env2.robot_position}, reward={reward:+.1f}")
    state = next_state
    if done:
        print(f"  Done! {info['message']}")
        break






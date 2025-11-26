"""
RL Training Visualization GUI
-----------------------------
A dedicated pygame interface for visualizing Q-Learning training,
watching episodes play out, and testing trained agents.
"""

import pygame
import numpy as np
import threading
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from warehouse.environment import Warehouse
from rl.qlearning import (
    QLearningAgent, WarehouseEnv, State, 
    ACTIONS, ACTION_NAMES, train_agent
)


@dataclass
class EpisodeFrame:
    """Single frame of episode visualization."""
    robot_pos: Tuple[int, int]
    target_pos: Tuple[int, int]
    action: int
    reward: float
    done: bool
    message: str


class RLTrainingUI:
    """
    Pygame UI for visualizing RL training and testing.
    """
    
    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_PANEL = (45, 45, 55)
    COLOR_GRID_BG = (60, 60, 70)
    COLOR_GRID_LINE = (80, 80, 90)
    COLOR_OBSTACLE = (100, 100, 110)
    COLOR_ROBOT = (50, 200, 100)
    COLOR_TARGET = (255, 200, 50)
    COLOR_PATH = (100, 150, 200, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DIM = (150, 150, 150)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)
    COLOR_BUTTON = (70, 130, 180)
    COLOR_BUTTON_HOVER = (90, 150, 200)
    COLOR_BUTTON_ACTIVE = (100, 200, 150)
    COLOR_GRAPH_BG = (40, 40, 50)
    COLOR_GRAPH_LINE = (100, 200, 255)
    COLOR_GRAPH_SUCCESS = (100, 255, 150)
    
    def __init__(self, warehouse: Warehouse = None, 
                 grid_size: int = 10,
                 window_width: int = 900,
                 window_height: int = 700):
        """
        Initialize the RL Training UI.
        
        Args:
            warehouse: Existing warehouse or None to create new
            grid_size: Grid size if creating new warehouse
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        # Create or use warehouse
        if warehouse is None:
            self.warehouse = Warehouse(grid_size, grid_size, obstacle_density=0.15)
            self.warehouse.place_obstacles()
        else:
            self.warehouse = warehouse
        
        # Initialize pygame
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("RL Training Visualizer")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_tiny = pygame.font.Font(None, 16)
        
        # Layout
        self.toolbar_height = 50
        self.stats_width = 250
        self.graph_height = 150
        self.grid_area_width = window_width - self.stats_width
        self.grid_area_height = window_height - self.toolbar_height - self.graph_height
        
        # Calculate cell size
        max_cells = max(self.warehouse.width, self.warehouse.height)
        self.cell_size = min(
            (self.grid_area_width - 40) // max_cells,
            (self.grid_area_height - 40) // max_cells
        )
        self.grid_width = self.cell_size * self.warehouse.width
        self.grid_height = self.cell_size * self.warehouse.height
        self.grid_offset_x = (self.grid_area_width - self.grid_width) // 2
        self.grid_offset_y = self.toolbar_height + (self.grid_area_height - self.grid_height) // 2
        
        # RL components
        self.env = WarehouseEnv(self.warehouse)
        self.agent = QLearningAgent(
            alpha=0.2,            # Slightly higher learning rate
            gamma=0.95,           # Slightly lower discount for faster learning
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.99,   # Faster decay
            use_relative_state=True
        )
        
        # Training state
        self.is_training = False
        self.is_testing = False
        self.training_thread = None
        self.current_episode = 0
        self.total_episodes = 20000  # Train for 20k episodes for thorough coverage
        self.episodes_completed = 0
        self.current_step = 0
        self.current_reward = 0.0
        self.episode_rewards: List[float] = []
        self.episode_successes: List[bool] = []
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        
        # Visualization state
        self.robot_pos: Optional[Tuple[int, int]] = None
        self.target_pos: Optional[Tuple[int, int]] = None
        self.path_history: List[Tuple[int, int]] = []
        self.last_action: Optional[int] = None
        self.last_message: str = ""
        self.episode_done = False
        
        # Speed control
        self.speed_options = [1, 2, 5, 10, 50, 100, 0]  # 0 = max speed (no visualization)
        self.speed_labels = ["1x", "2x", "5x", "10x", "50x", "100x", "Max"]
        self.current_speed_idx = 2  # Start at 5x
        self.visualization_delay = 0.2  # Base delay in seconds
        
        # UI state
        self.running = True
        self.hover_button: Optional[str] = None
        self.buttons = self._create_buttons()
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _create_buttons(self) -> Dict[str, pygame.Rect]:
        """Create toolbar buttons."""
        buttons = {}
        x = 10
        y = 10
        btn_height = 30
        spacing = 10
        
        button_defs = [
            ("train", "Train", 70),
            ("pause", "Pause", 60),
            ("test", "Test", 50),
            ("reset", "Reset", 55),
            ("save", "Save", 50),
            ("load", "Load", 50),
            ("speed_down", "−", 30),
            ("speed_label", "5x", 45),
            ("speed_up", "+", 30),
            ("new_grid", "New Grid", 75),
        ]
        
        for btn_id, label, width in button_defs:
            buttons[btn_id] = pygame.Rect(x, y, width, btn_height)
            x += width + spacing
        
        return buttons
    
    def _get_button_at(self, pos: Tuple[int, int]) -> Optional[str]:
        """Get button ID at mouse position."""
        for btn_id, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return btn_id
        return None
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.stop_training()
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.stop_training()
                    return False
                elif event.key == pygame.K_SPACE:
                    if self.is_training:
                        self.stop_training()
                    else:
                        self.start_training()
                elif event.key == pygame.K_t:
                    self.start_testing()
                elif event.key == pygame.K_r:
                    self.reset_agent()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._change_speed(1)
                elif event.key == pygame.K_MINUS:
                    self._change_speed(-1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn = self._get_button_at(event.pos)
                if btn:
                    self._handle_button_click(btn)
            
            elif event.type == pygame.MOUSEMOTION:
                self.hover_button = self._get_button_at(event.pos)
        
        return True
    
    def _handle_button_click(self, btn_id: str):
        """Handle button click."""
        if btn_id == "train":
            if self.is_training:
                self.stop_training()
            else:
                self.start_training()
        elif btn_id == "pause":
            self.stop_training()
            self.is_testing = False
        elif btn_id == "test":
            self.start_testing()
        elif btn_id == "reset":
            self.reset_agent()
        elif btn_id == "save":
            self.save_agent()
        elif btn_id == "load":
            self.load_agent()
        elif btn_id == "speed_up":
            self._change_speed(1)
        elif btn_id == "speed_down":
            self._change_speed(-1)
        elif btn_id == "new_grid":
            self.new_grid()
    
    def _change_speed(self, delta: int):
        """Change visualization speed."""
        self.current_speed_idx = max(0, min(len(self.speed_options) - 1, 
                                             self.current_speed_idx + delta))
    
    def start_training(self):
        """Start training in a background thread."""
        if self.is_training:
            return
        
        self.is_training = True
        self.is_testing = False
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training."""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=1.0)
            self.training_thread = None
    
    def _training_loop(self):
        """Background training loop."""
        while self.is_training and self.current_episode < self.total_episodes:
            # Reset environment
            state = self.env.reset()
            
            with self.lock:
                self.robot_pos = self.env.robot_position
                self.target_pos = self.env.target_position
                self.path_history = [self.robot_pos]
                self.current_step = 0
                self.current_reward = 0.0
                self.episode_done = False
            
            # Run episode
            while not self.env.done and self.is_training:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Update agent
                self.agent.update(state, action, reward, next_state, done)
                
                # Update visualization state
                with self.lock:
                    self.robot_pos = self.env.robot_position
                    self.path_history.append(self.robot_pos)
                    self.current_step += 1
                    self.current_reward += reward
                    self.last_action = action
                    self.last_message = info.get('message', '')
                    self.episode_done = done
                
                state = next_state
                
                # Delay for visualization
                speed = self.speed_options[self.current_speed_idx]
                if speed > 0:
                    time.sleep(self.visualization_delay / speed)
            
            # Episode complete
            self.agent.decay_epsilon()
            
            with self.lock:
                success = self.current_reward > 50  # Reached target
                self.episode_rewards.append(self.current_reward)
                self.episode_successes.append(success)
                self.recent_rewards.append(self.current_reward)
                self.recent_successes.append(success)
                self.current_episode += 1
                self.episodes_completed += 1
            
            # Small delay between episodes
            if self.speed_options[self.current_speed_idx] > 0:
                time.sleep(0.05)
        
        self.is_training = False
    
    def start_testing(self):
        """Start testing the trained agent."""
        if self.is_testing:
            return
        
        self.stop_training()
        self.is_testing = True
        
        # Run test episode in background
        test_thread = threading.Thread(target=self._test_loop, daemon=True)
        test_thread.start()
    
    def _test_loop(self):
        """Run a test episode with the trained agent."""
        while self.is_testing and self.running:
            # Reset environment
            state = self.env.reset()
            
            with self.lock:
                self.robot_pos = self.env.robot_position
                self.target_pos = self.env.target_position
                self.path_history = [self.robot_pos]
                self.current_step = 0
                self.current_reward = 0.0
                self.episode_done = False
            
            # Run episode with greedy policy
            while not self.env.done and self.is_testing:
                action = self.agent.get_policy(state)
                next_state, reward, done, info = self.env.step(action)
                
                with self.lock:
                    self.robot_pos = self.env.robot_position
                    self.path_history.append(self.robot_pos)
                    self.current_step += 1
                    self.current_reward += reward
                    self.last_action = action
                    self.last_message = info.get('message', '')
                    self.episode_done = done
                
                state = next_state
                
                # Slower visualization for testing
                time.sleep(0.15)
            
            # Pause between test episodes
            time.sleep(1.0)
    
    def reset_agent(self):
        """Reset the agent and training state."""
        self.stop_training()
        self.is_testing = False
        
        self.agent = QLearningAgent(
            alpha=0.1, gamma=0.99, epsilon=1.0,
            epsilon_min=0.01, epsilon_decay=0.995,
            use_relative_state=True
        )
        
        self.current_episode = 0
        self.episodes_completed = 0
        self.episode_rewards.clear()
        self.episode_successes.clear()
        self.recent_rewards.clear()
        self.recent_successes.clear()
        self.path_history.clear()
        self.robot_pos = None
        self.target_pos = None
    
    def new_grid(self):
        """Generate a new random grid - preserves Q-table for transfer learning."""
        self.stop_training()
        self.is_testing = False
        
        size = self.warehouse.width
        self.warehouse = Warehouse(size, size, obstacle_density=0.15)
        self.warehouse.place_obstacles()
        self.env = WarehouseEnv(self.warehouse)
        
        # DON'T reset agent - obstacle-aware states transfer between grids!
        # Just reset visualization state
        self.path_history.clear()
        self.robot_pos = None
        self.target_pos = None
        self.last_message = "New grid - Q-table preserved!"
    
    def save_agent(self):
        """Save the trained agent."""
        self.agent.save("rl_agent_gui.pkl")
        self.last_message = "Agent saved!"
    
    def load_agent(self):
        """Load a trained agent."""
        import os
        if os.path.exists("rl_agent_gui.pkl"):
            self.agent.load("rl_agent_gui.pkl")
            self.last_message = "Agent loaded!"
        elif os.path.exists("rl_agent.pkl"):
            self.agent.load("rl_agent.pkl")
            self.last_message = "Agent loaded!"
        else:
            self.last_message = "No saved agent found"
    
    def draw_toolbar(self):
        """Draw the toolbar."""
        toolbar_rect = pygame.Rect(0, 0, self.window_width, self.toolbar_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, toolbar_rect)
        
        for btn_id, rect in self.buttons.items():
            # Determine button color
            is_hover = (self.hover_button == btn_id)
            is_active = False
            
            if btn_id == "train" and self.is_training:
                is_active = True
            elif btn_id == "test" and self.is_testing:
                is_active = True
            
            if is_active:
                color = self.COLOR_BUTTON_ACTIVE
            elif is_hover:
                color = self.COLOR_BUTTON_HOVER
            else:
                color = self.COLOR_BUTTON
            
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            # Button label
            if btn_id == "train":
                label = "Stop" if self.is_training else "Train"
            elif btn_id == "speed_label":
                label = self.speed_labels[self.current_speed_idx]
            else:
                label = btn_id.replace("_", " ").title()
                if btn_id == "speed_up":
                    label = "+"
                elif btn_id == "speed_down":
                    label = "−"
            
            text = self.font_small.render(label, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_grid(self):
        """Draw the warehouse grid."""
        # Grid background
        grid_rect = pygame.Rect(
            self.grid_offset_x - 5,
            self.grid_offset_y - 5,
            self.grid_width + 10,
            self.grid_height + 10
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)
        
        # Draw cells
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                cell_rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Cell background
                if self.warehouse.grid[y, x] == Warehouse.OBSTACLE:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, cell_rect)
                
                # Grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, cell_rect, 1)
        
        # Draw path history
        with self.lock:
            if len(self.path_history) > 1:
                for i, pos in enumerate(self.path_history[:-1]):
                    alpha = min(255, 50 + i * 10)
                    path_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    path_surface.fill((100, 150, 200, min(100, alpha)))
                    self.screen.blit(path_surface, (
                        self.grid_offset_x + pos[0] * self.cell_size,
                        self.grid_offset_y + pos[1] * self.cell_size
                    ))
            
            # Draw target
            if self.target_pos:
                tx, ty = self.target_pos
                center_x = self.grid_offset_x + tx * self.cell_size + self.cell_size // 2
                center_y = self.grid_offset_y + ty * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 3
                pygame.draw.circle(self.screen, self.COLOR_TARGET, (center_x, center_y), radius)
                pygame.draw.circle(self.screen, (200, 150, 0), (center_x, center_y), radius, 2)
            
            # Draw robot
            if self.robot_pos:
                rx, ry = self.robot_pos
                robot_size = int(self.cell_size * 0.7)
                offset = (self.cell_size - robot_size) // 2
                robot_rect = pygame.Rect(
                    self.grid_offset_x + rx * self.cell_size + offset,
                    self.grid_offset_y + ry * self.cell_size + offset,
                    robot_size,
                    robot_size
                )
                
                color = self.COLOR_SUCCESS if self.episode_done and self.current_reward > 50 else self.COLOR_ROBOT
                pygame.draw.rect(self.screen, color, robot_rect, border_radius=5)
                pygame.draw.rect(self.screen, (30, 30, 30), robot_rect, 2, border_radius=5)
                
                # Robot direction indicator
                if self.last_action is not None and self.last_action < 4:
                    dx, dy = ACTIONS[self.last_action]
                    arrow_start = robot_rect.center
                    arrow_end = (arrow_start[0] + dx * 10, arrow_start[1] + dy * 10)
                    pygame.draw.line(self.screen, (255, 255, 255), arrow_start, arrow_end, 2)
    
    def draw_stats_panel(self):
        """Draw the statistics panel."""
        panel_x = self.window_width - self.stats_width
        panel_rect = pygame.Rect(panel_x, self.toolbar_height, 
                                  self.stats_width, self.grid_area_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        
        x = panel_x + 15
        y = self.toolbar_height + 15
        line_height = 24
        
        # Title
        title = self.font_medium.render("Training Stats", True, self.COLOR_TEXT)
        self.screen.blit(title, (x, y))
        y += line_height + 10
        
        # Status
        if self.is_training:
            status = "TRAINING"
            status_color = self.COLOR_SUCCESS
        elif self.is_testing:
            status = "TESTING"
            status_color = self.COLOR_TARGET
        else:
            status = "PAUSED"
            status_color = self.COLOR_TEXT_DIM
        
        status_text = self.font_small.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (x, y))
        y += line_height + 5
        
        # Episode info
        with self.lock:
            stats = [
                f"Episode: {self.current_episode}/{self.total_episodes}",
                f"Step: {self.current_step}",
                f"Reward: {self.current_reward:.1f}",
                "",
                f"Epsilon: {self.agent.epsilon:.3f}",
                f"Q-table: {len(self.agent.q_table)} states",
                "",
            ]
            
            if self.recent_rewards:
                avg_reward = np.mean(list(self.recent_rewards))
                stats.append(f"Avg Reward (100): {avg_reward:.1f}")
            
            if self.recent_successes:
                success_rate = np.mean(list(self.recent_successes)) * 100
                stats.append(f"Success Rate: {success_rate:.1f}%")
        
        for stat in stats:
            if stat:
                text = self.font_small.render(stat, True, self.COLOR_TEXT)
                self.screen.blit(text, (x, y))
            y += line_height
        
        # Last action/message
        y += 10
        if self.last_action is not None:
            action_text = self.font_small.render(
                f"Action: {ACTION_NAMES.get(self.last_action, '?')}", 
                True, self.COLOR_TEXT_DIM
            )
            self.screen.blit(action_text, (x, y))
            y += line_height
        
        if self.last_message:
            msg_color = self.COLOR_SUCCESS if "target" in self.last_message.lower() else self.COLOR_TEXT_DIM
            msg_text = self.font_tiny.render(self.last_message, True, msg_color)
            self.screen.blit(msg_text, (x, y))
        
        # Q-values display
        y = self.toolbar_height + self.grid_area_height - 150
        qval_title = self.font_small.render("Q-Values (relative state):", True, self.COLOR_TEXT)
        self.screen.blit(qval_title, (x, y))
        y += line_height
        
        # Show current state's Q-values
        with self.lock:
            if self.robot_pos and self.target_pos:
                state = State(
                    robot_x=self.robot_pos[0], robot_y=self.robot_pos[1],
                    target_x=self.target_pos[0], target_y=self.target_pos[1]
                )
                q_values = self.agent._get_q_values(state)
                
                for i, (action_name, q_val) in enumerate(zip(ACTION_NAMES.values(), q_values)):
                    best = (i == np.argmax(q_values))
                    color = self.COLOR_SUCCESS if best else self.COLOR_TEXT_DIM
                    text = self.font_tiny.render(f"{action_name}: {q_val:.2f}", True, color)
                    self.screen.blit(text, (x + (i % 3) * 70, y + (i // 3) * 18))
    
    def draw_graph(self):
        """Draw the training progress graph."""
        graph_y = self.window_height - self.graph_height
        graph_rect = pygame.Rect(0, graph_y, self.window_width, self.graph_height)
        pygame.draw.rect(self.screen, self.COLOR_GRAPH_BG, graph_rect)
        
        # Title
        title = self.font_small.render("Training Progress", True, self.COLOR_TEXT)
        self.screen.blit(title, (10, graph_y + 5))
        
        # Draw reward graph
        with self.lock:
            if len(self.episode_rewards) > 1:
                graph_left = 50
                graph_right = self.window_width - 20
                graph_top = graph_y + 30
                graph_bottom = self.window_height - 20
                graph_width = graph_right - graph_left
                graph_height = graph_bottom - graph_top
                
                # Normalize rewards
                rewards = list(self.episode_rewards)
                if len(rewards) > graph_width:
                    # Downsample
                    step = len(rewards) / graph_width
                    rewards = [rewards[int(i * step)] for i in range(int(graph_width))]
                
                min_r = min(rewards) if rewards else 0
                max_r = max(rewards) if rewards else 1
                range_r = max_r - min_r if max_r != min_r else 1
                
                # Draw axes
                pygame.draw.line(self.screen, self.COLOR_TEXT_DIM, 
                               (graph_left, graph_bottom), (graph_right, graph_bottom), 1)
                pygame.draw.line(self.screen, self.COLOR_TEXT_DIM,
                               (graph_left, graph_top), (graph_left, graph_bottom), 1)
                
                # Draw reward line
                points = []
                for i, r in enumerate(rewards):
                    x = graph_left + (i / max(1, len(rewards) - 1)) * graph_width
                    y = graph_bottom - ((r - min_r) / range_r) * graph_height
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.COLOR_GRAPH_LINE, False, points, 2)
                
                # Y-axis labels
                max_label = self.font_tiny.render(f"{max_r:.0f}", True, self.COLOR_TEXT_DIM)
                min_label = self.font_tiny.render(f"{min_r:.0f}", True, self.COLOR_TEXT_DIM)
                self.screen.blit(max_label, (5, graph_top))
                self.screen.blit(min_label, (5, graph_bottom - 15))
                
                # X-axis label
                ep_label = self.font_tiny.render(f"Episodes: {len(self.episode_rewards)}", 
                                                  True, self.COLOR_TEXT_DIM)
                self.screen.blit(ep_label, (graph_right - 100, graph_bottom + 3))
    
    def render(self):
        """Render the complete UI."""
        self.screen.fill(self.COLOR_BG)
        
        self.draw_toolbar()
        self.draw_grid()
        self.draw_stats_panel()
        self.draw_graph()
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def run(self):
        """Main loop."""
        print("=" * 60)
        print("  RL Training Visualizer")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE - Start/Stop training")
        print("  T     - Test trained agent")
        print("  R     - Reset agent")
        print("  +/-   - Adjust speed")
        print("  ESC   - Exit")
        print()
        
        while self.running:
            if not self.handle_events():
                break
            self.render()
        
        self.stop_training()
        pygame.quit()
    
    def quit(self):
        """Clean up."""
        self.stop_training()
        pygame.quit()



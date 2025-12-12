"""
Multi-Robot RL Training Visualization GUI
------------------------------------------
A pygame interface for visualizing multi-agent Q-Learning training,
watching multiple robots coordinate, and testing trained multi-robot systems.
"""

import pygame
import numpy as np
import threading
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from warehouse.environment import Warehouse
from rl.multi_agent_qlearning import (
    MultiAgentWarehouseEnv, MultiAgentQLearning, MultiAgentState,
    train_multi_agent, ACTIONS, ACTION_NAMES
)


# Robot colors for visualization (up to 6 robots)
ROBOT_COLORS = [
    (50, 200, 100),   # Green
    (100, 150, 255),  # Blue
    (255, 150, 100),  # Orange
    (200, 100, 255),  # Purple
    (255, 200, 50),   # Yellow
    (100, 255, 200),  # Cyan
]

TARGET_COLORS = [
    (30, 150, 70),    # Dark Green
    (70, 100, 200),   # Dark Blue
    (200, 100, 70),   # Dark Orange
    (150, 70, 200),   # Dark Purple
    (200, 150, 30),   # Dark Yellow
    (70, 200, 150),   # Dark Cyan
]


class MultiRobotRLUI:
    """
    Pygame UI for visualizing multi-robot RL training.
    """
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PANEL = (40, 40, 52)
    COLOR_TOOLBAR = (35, 35, 45)
    COLOR_GRID_BG = (55, 55, 65)
    COLOR_GRID_LINE = (75, 75, 85)
    COLOR_OBSTACLE = (95, 95, 105)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DIM = (140, 140, 150)
    COLOR_TEXT_LABEL = (110, 130, 150)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)
    COLOR_WARNING = (255, 200, 50)
    COLOR_BUTTON = (65, 120, 170)
    COLOR_BUTTON_HOVER = (85, 140, 190)
    COLOR_BUTTON_ACTIVE = (90, 190, 140)
    COLOR_BUTTON_SECONDARY = (70, 80, 100)
    COLOR_BUTTON_CONTROL = (55, 70, 90)
    COLOR_SEPARATOR = (55, 55, 65)
    COLOR_GROUP_BG = (45, 45, 58)
    COLOR_GRAPH_BG = (35, 35, 45)
    COLOR_GRAPH_LINE = (100, 200, 255)
    
    def __init__(self, 
                 warehouse: Warehouse = None,
                 grid_size: int = 12,
                 num_robots: int = 3,
                 window_width: int = 1100,
                 window_height: int = 750):
        """
        Initialize Multi-Robot RL UI.
        
        Args:
            warehouse: Existing warehouse or None to create new
            grid_size: Grid size if creating new warehouse
            num_robots: Number of robots
            window_width: Window width
            window_height: Window height
        """
        # Create warehouse
        if warehouse is None:
            self.warehouse = Warehouse(grid_size, grid_size, obstacle_density=0.12)
            self.warehouse.place_obstacles()
        else:
            self.warehouse = warehouse
        
        self.num_robots = min(num_robots, 6)  # Max 6 robots
        
        # Initialize pygame
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Multi-Robot RL Training")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_tiny = pygame.font.Font(None, 16)
        
        # Layout
        self.toolbar_height = 62
        self.stats_width = 280
        self.graph_height = 140
        self.grid_area_width = window_width - self.stats_width
        self.grid_area_height = window_height - self.toolbar_height - self.graph_height
        
        # Calculate cell size
        max_cells = max(self.warehouse.width, self.warehouse.height)
        self.cell_size = min(
            (self.grid_area_width - 60) // max_cells,
            (self.grid_area_height - 40) // max_cells
        )
        self.grid_width = self.cell_size * self.warehouse.width
        self.grid_height = self.cell_size * self.warehouse.height
        self.grid_offset_x = (self.grid_area_width - self.grid_width) // 2
        self.grid_offset_y = self.toolbar_height + (self.grid_area_height - self.grid_height) // 2
        
        # Shared packages mode (any robot can collect any package)
        self.shared_packages = True
        self.num_packages = self.num_robots + 2  # More packages than robots
        
        # RL components
        self.env = MultiAgentWarehouseEnv(
            self.warehouse, 
            self.num_robots,
            num_packages=self.num_packages,
            shared_packages=self.shared_packages
        )
        self.agent = MultiAgentQLearning(
            num_robots=self.num_robots,
            alpha=0.15,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            shared_qtable=True
        )
        
        # Training state
        self.is_training = False
        self.is_testing = False
        self.training_thread = None
        self.current_episode = 0
        self.total_episodes = 10000
        self.episodes_completed = 0
        self.current_step = 0
        self.episode_rewards: List[float] = []
        self.episode_successes: List[float] = []
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        
        # Episode presets
        self.episode_presets = [1000, 5000, 10000, 25000, 50000]
        self.current_preset_idx = 2
        
        # Robot presets
        self.robot_presets = [2, 3, 4, 5, 6]
        self.current_robot_idx = 1  # Start with 3 robots
        
        # Package presets
        self.package_presets = [3, 5, 7, 10, 15]
        self.current_package_idx = 1  # Start with 5 packages
        
        # Visualization state
        self.robot_positions: List[Tuple[int, int]] = []
        self.target_positions: List[Tuple[int, int]] = []
        self.available_packages: List[Tuple[int, int]] = []  # Shared package pool
        self.robot_dones: List[bool] = []
        self.path_histories: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_robots)]
        self.last_actions: List[int] = [-1] * self.num_robots
        self.last_message: str = ""
        self.deadlocked_robots: List[int] = []
        self.packages_collected: int = 0
        
        # Speed control
        self.speed_options = [1, 2, 5, 10, 50, 100, 0]
        self.speed_labels = ["1x", "2x", "5x", "10x", "50x", "100x", "Max"]
        self.current_speed_idx = 3  # Start at 10x
        self.visualization_delay = 0.15
        
        # UI state
        self.running = True
        self.hover_button: Optional[str] = None
        self.buttons = self._create_buttons()
        
        # Thread lock
        self.lock = threading.Lock()
    
    def _create_buttons(self) -> Dict[str, pygame.Rect]:
        """Create toolbar buttons with grouped layout."""
        buttons = {}
        btn_height = 26
        btn_y = 24  # Buttons lower to leave room for labels
        group_spacing = 16  # Space between groups
        btn_spacing = 3    # Space between buttons in same group
        
        x = 12
        
        # Group 1: Control buttons (Train, Pause, Test)
        self.group_control_x = x
        for btn_id, width in [("train", 55), ("pause", 48), ("test", 42)]:
            buttons[btn_id] = pygame.Rect(x, btn_y, width, btn_height)
            x += width + btn_spacing
        self.group_control_end = x - btn_spacing
        
        x += group_spacing
        
        # Group 2: Data buttons (Reset, Save, Load)
        self.group_data_x = x
        for btn_id, width in [("reset", 46), ("save", 40), ("load", 40)]:
            buttons[btn_id] = pygame.Rect(x, btn_y, width, btn_height)
            x += width + btn_spacing
        self.group_data_end = x - btn_spacing
        
        x += group_spacing
        
        # Group 3: Speed control
        self.group_speed_x = x
        for btn_id, width in [("speed_down", 22), ("speed_label", 38), ("speed_up", 22)]:
            buttons[btn_id] = pygame.Rect(x, btn_y, width, btn_height)
            x += width + 2
        self.group_speed_end = x - 2
        
        x += group_spacing
        
        # Group 4: Episodes control
        self.group_episodes_x = x
        for btn_id, width in [("ep_down", 22), ("ep_label", 42), ("ep_up", 22)]:
            buttons[btn_id] = pygame.Rect(x, btn_y, width, btn_height)
            x += width + 2
        self.group_episodes_end = x - 2
        
        x += group_spacing
        
        # Group 5: Robots control
        self.group_robots_x = x
        for btn_id, width in [("robot_down", 22), ("robot_label", 36), ("robot_up", 22)]:
            buttons[btn_id] = pygame.Rect(x, btn_y, width, btn_height)
            x += width + 2
        self.group_robots_end = x - 2
        
        x += group_spacing
        
        # Group 6: Grid button
        self.group_grid_x = x
        buttons["new_grid"] = pygame.Rect(x, btn_y, 50, btn_height)
        self.group_grid_end = x + 50
        
        return buttons
    
    def _get_button_at(self, pos: Tuple[int, int]) -> Optional[str]:
        """Get button at mouse position."""
        for btn_id, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return btn_id
        return None
    
    def _format_episodes(self, count: int) -> str:
        """Format episode count."""
        if count >= 1000000:
            return f"{count // 1000000}M"
        elif count >= 1000:
            return f"{count // 1000}K"
        return str(count)
    
    def _change_episodes(self, delta: int):
        """Change episode count."""
        if self.is_training:
            self.last_message = "Stop training first"
            return
        self.current_preset_idx = max(0, min(len(self.episode_presets) - 1,
                                              self.current_preset_idx + delta))
        self.total_episodes = self.episode_presets[self.current_preset_idx]
        self.last_message = f"Episodes: {self._format_episodes(self.total_episodes)}"
    
    def _change_robots(self, delta: int):
        """Change number of robots."""
        if self.is_training or self.is_testing:
            self.last_message = "Stop first to change robots"
            return
        
        self.current_robot_idx = max(0, min(len(self.robot_presets) - 1,
                                             self.current_robot_idx + delta))
        new_num = self.robot_presets[self.current_robot_idx]
        
        if new_num != self.num_robots:
            self.num_robots = new_num
            self._reinitialize_rl()
            self.last_message = f"Robots: {self.num_robots}"
    
    def _reinitialize_rl(self):
        """Reinitialize RL components with new robot/package count."""
        self.env = MultiAgentWarehouseEnv(
            self.warehouse, 
            self.num_robots,
            num_packages=self.num_packages,
            shared_packages=self.shared_packages
        )
        self.agent = MultiAgentQLearning(
            num_robots=self.num_robots,
            alpha=0.15,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            shared_qtable=True
        )
        self.path_histories = [[] for _ in range(self.num_robots)]
        self.last_actions = [-1] * self.num_robots
        self.robot_positions.clear()
        self.target_positions.clear()
        self.available_packages.clear()
        self.robot_dones.clear()
        self.packages_collected = 0
        self.current_episode = 0
        self.episodes_completed = 0
        self.episode_rewards.clear()
        self.episode_successes.clear()
        self.recent_rewards.clear()
        self.recent_successes.clear()
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
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
                elif event.key == pygame.K_LEFT:
                    self._change_episodes(-1)
                elif event.key == pygame.K_RIGHT:
                    self._change_episodes(1)
                elif event.key == pygame.K_UP:
                    self._change_robots(1)
                elif event.key == pygame.K_DOWN:
                    self._change_robots(-1)
            
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
        elif btn_id == "ep_up":
            self._change_episodes(1)
        elif btn_id == "ep_down":
            self._change_episodes(-1)
        elif btn_id == "robot_up":
            self._change_robots(1)
        elif btn_id == "robot_down":
            self._change_robots(-1)
        elif btn_id == "new_grid":
            self.new_grid()
    
    def _change_speed(self, delta: int):
        """Change visualization speed."""
        self.current_speed_idx = max(0, min(len(self.speed_options) - 1,
                                             self.current_speed_idx + delta))
    
    def start_training(self):
        """Start training."""
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
            states = self.env.reset()
            
            with self.lock:
                self.robot_positions = [self.env.robots[i].position for i in range(self.num_robots)]
                self.target_positions = [self.env.robots[i].target for i in range(self.num_robots)]
                self.available_packages = list(self.env.available_packages)
                self.robot_dones = [False] * self.num_robots
                self.path_histories = [[pos] for pos in self.robot_positions]
                self.current_step = 0
                self.deadlocked_robots = []
                self.packages_collected = 0
            
            episode_reward = 0.0
            
            # Run episode
            while not self.env.done and self.is_training:
                actions = self.agent.select_actions(states, training=True)
                next_states, rewards, dones, info = self.env.step(actions)
                
                self.agent.update_all(states, actions, rewards, next_states, dones)
                
                episode_reward += sum(rewards)
                
                with self.lock:
                    for i in range(self.num_robots):
                        self.robot_positions[i] = self.env.robots[i].position
                        self.path_histories[i].append(self.robot_positions[i])
                        self.last_actions[i] = actions[i]
                        self.robot_dones[i] = self.env.robots[i].done
                        # Update target positions for shared mode
                        self.target_positions[i] = self.env.robots[i].target
                    self.current_step += 1
                    self.deadlocked_robots = info.get('deadlocked', [])
                    self.available_packages = list(self.env.available_packages)
                    self.packages_collected = info.get('packages_collected', 0)
                    
                    # Get message from first robot
                    robot_infos = info.get('robots', [{}] * self.num_robots)
                    messages = [ri.get('message', '') for ri in robot_infos]
                    self.last_message = messages[0] if messages else ''
                
                states = next_states
                
                # Visualization delay
                speed = self.speed_options[self.current_speed_idx]
                if speed > 0:
                    time.sleep(self.visualization_delay / speed)
            
            # Episode complete
            self.agent.decay_epsilon()
            
            with self.lock:
                successes = sum(1 for r in self.env.robots if r.collected) / self.num_robots
                self.episode_rewards.append(episode_reward)
                self.episode_successes.append(successes)
                self.recent_rewards.append(episode_reward)
                self.recent_successes.append(successes)
                self.current_episode += 1
                self.episodes_completed += 1
            
            if self.speed_options[self.current_speed_idx] > 0:
                time.sleep(0.03)
        
        self.is_training = False
    
    def start_testing(self):
        """Start testing."""
        if self.is_testing:
            return
        
        self.stop_training()
        self.is_testing = True
        
        test_thread = threading.Thread(target=self._test_loop, daemon=True)
        test_thread.start()
    
    def _test_loop(self):
        """Test loop."""
        while self.is_testing and self.running:
            states = self.env.reset()
            
            with self.lock:
                self.robot_positions = [self.env.robots[i].position for i in range(self.num_robots)]
                self.target_positions = [self.env.robots[i].target for i in range(self.num_robots)]
                self.available_packages = list(self.env.available_packages)
                self.robot_dones = [False] * self.num_robots
                self.path_histories = [[pos] for pos in self.robot_positions]
                self.current_step = 0
                self.deadlocked_robots = []
                self.packages_collected = 0
            
            while not self.env.done and self.is_testing:
                actions = self.agent.get_policy_actions(states)
                next_states, rewards, dones, info = self.env.step(actions)
                
                with self.lock:
                    for i in range(self.num_robots):
                        self.robot_positions[i] = self.env.robots[i].position
                        self.path_histories[i].append(self.robot_positions[i])
                        self.last_actions[i] = actions[i]
                        self.robot_dones[i] = self.env.robots[i].done
                        self.target_positions[i] = self.env.robots[i].target
                    self.current_step += 1
                    self.deadlocked_robots = info.get('deadlocked', [])
                    self.available_packages = list(self.env.available_packages)
                    self.packages_collected = info.get('packages_collected', 0)
                
                states = next_states
                time.sleep(0.12)
            
            time.sleep(1.5)
    
    def reset_agent(self):
        """Reset agent."""
        self.stop_training()
        self.is_testing = False
        self._reinitialize_rl()
        self.last_message = "Agent reset!"
    
    def new_grid(self):
        """Generate new grid."""
        self.stop_training()
        self.is_testing = False
        
        size = self.warehouse.width
        self.warehouse = Warehouse(size, size, obstacle_density=0.12)
        self.warehouse.place_obstacles()
        self.env = MultiAgentWarehouseEnv(
            self.warehouse, 
            self.num_robots,
            num_packages=self.num_packages,
            shared_packages=self.shared_packages
        )
        
        self.path_histories = [[] for _ in range(self.num_robots)]
        self.robot_positions.clear()
        self.target_positions.clear()
        self.available_packages.clear()
        self.packages_collected = 0
        self.last_message = "New grid - Q-table preserved!"
    
    def save_agent(self):
        """Save agent."""
        self.agent.save("multi_rl_agent.pkl")
        self.last_message = "Multi-agent saved!"
    
    def load_agent(self):
        """Load agent."""
        import os
        if os.path.exists("multi_rl_agent.pkl"):
            try:
                self.agent.load("multi_rl_agent.pkl")
                self.last_message = "Multi-agent loaded!"
            except Exception as e:
                self.last_message = f"Load failed: {e}"
        else:
            self.last_message = "No saved agent found"
    
    def draw_toolbar(self):
        """Draw toolbar with grouped controls."""
        # Main toolbar background
        toolbar_rect = pygame.Rect(0, 0, self.window_width, self.toolbar_height)
        pygame.draw.rect(self.screen, self.COLOR_TOOLBAR, toolbar_rect)
        
        # Bottom border line
        pygame.draw.line(self.screen, self.COLOR_SEPARATOR,
                        (0, self.toolbar_height - 1),
                        (self.window_width, self.toolbar_height - 1), 1)
        
        # Draw group labels
        label_y = 6
        labels = [
            (self.group_control_x, self.group_control_end, "CONTROL"),
            (self.group_data_x, self.group_data_end, "DATA"),
            (self.group_speed_x, self.group_speed_end, "SPEED"),
            (self.group_episodes_x, self.group_episodes_end, "EPISODES"),
            (self.group_robots_x, self.group_robots_end, "ROBOTS"),
            (self.group_grid_x, self.group_grid_end, "GRID"),
        ]
        
        for start_x, end_x, label_text in labels:
            # Draw subtle background for group
            group_rect = pygame.Rect(start_x - 3, 18, end_x - start_x + 6, 34)
            pygame.draw.rect(self.screen, self.COLOR_GROUP_BG, group_rect, border_radius=4)
            
            # Draw label
            label = self.font_tiny.render(label_text, True, self.COLOR_TEXT_LABEL)
            label_x = start_x + (end_x - start_x - label.get_width()) // 2
            self.screen.blit(label, (label_x, label_y))
        
        # Draw buttons
        for btn_id, rect in self.buttons.items():
            is_hover = (self.hover_button == btn_id)
            is_active = False
            
            if btn_id == "train" and self.is_training:
                is_active = True
            elif btn_id == "test" and self.is_testing:
                is_active = True
            
            # Determine button color based on type
            if is_active:
                color = self.COLOR_BUTTON_ACTIVE
            elif is_hover:
                color = self.COLOR_BUTTON_HOVER
            elif btn_id in ("speed_down", "speed_up", "ep_down", "ep_up", "robot_down", "robot_up"):
                color = self.COLOR_BUTTON_CONTROL
            elif btn_id in ("speed_label", "ep_label", "robot_label"):
                color = self.COLOR_BUTTON_SECONDARY
            else:
                color = self.COLOR_BUTTON
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            
            # Button border for value displays
            if btn_id in ("speed_label", "ep_label", "robot_label"):
                pygame.draw.rect(self.screen, self.COLOR_SEPARATOR, rect, 1, border_radius=4)
            
            # Button label
            if btn_id == "train":
                label = "■ Stop" if self.is_training else "▶ Train"
            elif btn_id == "speed_label":
                label = self.speed_labels[self.current_speed_idx]
            elif btn_id == "ep_label":
                label = self._format_episodes(self.total_episodes)
            elif btn_id == "robot_label":
                label = f"{self.num_robots}R"
            elif btn_id in ("ep_up", "robot_up"):
                label = "▶"
            elif btn_id in ("ep_down", "robot_down"):
                label = "◀"
            elif btn_id == "speed_up":
                label = "+"
            elif btn_id == "speed_down":
                label = "−"
            elif btn_id == "new_grid":
                label = "New"
            elif btn_id == "test":
                label = "◉ Test" if self.is_testing else "Test"
            else:
                label = btn_id.title()
            
            text = self.font_small.render(label, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_grid(self):
        """Draw warehouse grid with all robots."""
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
                
                if self.warehouse.grid[y, x] == Warehouse.OBSTACLE:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, cell_rect)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, cell_rect, 1)
        
        with self.lock:
            # Draw path histories
            for i, path in enumerate(self.path_histories):
                if len(path) > 1:
                    color = (*ROBOT_COLORS[i % len(ROBOT_COLORS)][:3], 80)
                    for j, pos in enumerate(path[:-1]):
                        alpha = min(100, 30 + j * 5)
                        path_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        path_surface.fill((*color[:3], min(80, alpha)))
                        self.screen.blit(path_surface, (
                            self.grid_offset_x + pos[0] * self.cell_size,
                            self.grid_offset_y + pos[1] * self.cell_size
                        ))
            
            # Draw available packages (shared mode)
            if self.shared_packages:
                for pkg in self.available_packages:
                    px, py = pkg
                    center_x = self.grid_offset_x + px * self.cell_size + self.cell_size // 2
                    center_y = self.grid_offset_y + py * self.cell_size + self.cell_size // 2
                    radius = self.cell_size // 3
                    
                    # Gold color for packages
                    pkg_color = (255, 200, 50)
                    pygame.draw.circle(self.screen, pkg_color, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, (200, 150, 0), (center_x, center_y), radius, 2)
                    
                    # Package symbol
                    label = self.font_tiny.render("P", True, (50, 50, 50))
                    label_rect = label.get_rect(center=(center_x, center_y))
                    self.screen.blit(label, label_rect)
            
            # Draw robot targets (show which package each robot is heading to)
            for i, target in enumerate(self.target_positions):
                if target is None:
                    continue  # No target assigned
                if i < len(self.robot_dones) and self.robot_dones[i]:
                    continue  # Don't draw target if robot reached it
                
                tx, ty = target
                center_x = self.grid_offset_x + tx * self.cell_size + self.cell_size // 2
                center_y = self.grid_offset_y + ty * self.cell_size + self.cell_size // 2
                
                # Draw a small indicator showing which robot is targeting this
                if self.shared_packages:
                    # Draw robot color ring around the package
                    ring_radius = self.cell_size // 3 + 4
                    robot_color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                    pygame.draw.circle(self.screen, robot_color, (center_x, center_y), ring_radius, 3)
                else:
                    # Individual mode: draw target as before
                    radius = self.cell_size // 3
                    color = TARGET_COLORS[i % len(TARGET_COLORS)]
                    pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), radius, 2)
                    label = self.font_tiny.render(str(i), True, self.COLOR_TEXT)
                    label_rect = label.get_rect(center=(center_x, center_y))
                    self.screen.blit(label, label_rect)
            
            # Draw robots
            for i, pos in enumerate(self.robot_positions):
                rx, ry = pos
                robot_size = int(self.cell_size * 0.7)
                offset = (self.cell_size - robot_size) // 2
                robot_rect = pygame.Rect(
                    self.grid_offset_x + rx * self.cell_size + offset,
                    self.grid_offset_y + ry * self.cell_size + offset,
                    robot_size,
                    robot_size
                )
                
                # Color based on state
                if i < len(self.robot_dones) and self.robot_dones[i]:
                    color = self.COLOR_SUCCESS
                elif i in self.deadlocked_robots:
                    color = self.COLOR_FAIL
                else:
                    color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                
                pygame.draw.rect(self.screen, color, robot_rect, border_radius=5)
                pygame.draw.rect(self.screen, (30, 30, 30), robot_rect, 2, border_radius=5)
                
                # Robot ID
                id_text = self.font_tiny.render(str(i), True, (255, 255, 255))
                id_rect = id_text.get_rect(center=robot_rect.center)
                self.screen.blit(id_text, id_rect)
                
                # Direction indicator
                if i < len(self.last_actions) and self.last_actions[i] >= 0 and self.last_actions[i] < 4:
                    dx, dy = ACTIONS[self.last_actions[i]]
                    arrow_start = robot_rect.center
                    arrow_end = (arrow_start[0] + dx * 8, arrow_start[1] + dy * 8)
                    pygame.draw.line(self.screen, (255, 255, 255), arrow_start, arrow_end, 2)
    
    def draw_stats_panel(self):
        """Draw statistics panel."""
        panel_x = self.window_width - self.stats_width
        panel_rect = pygame.Rect(panel_x, self.toolbar_height,
                                  self.stats_width, self.grid_area_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        
        x = panel_x + 15
        y = self.toolbar_height + 15
        line_height = 22
        
        # Title
        title = self.font_medium.render("Multi-Robot RL", True, self.COLOR_TEXT)
        self.screen.blit(title, (x, y))
        y += line_height + 8
        
        # Status
        if self.is_training:
            status = "TRAINING"
            status_color = self.COLOR_SUCCESS
        elif self.is_testing:
            status = "TESTING"
            status_color = self.COLOR_WARNING
        else:
            status = "PAUSED"
            status_color = self.COLOR_TEXT_DIM
        
        status_text = self.font_small.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (x, y))
        y += line_height + 5
        
        # Stats
        with self.lock:
            stats = [
                f"Robots: {self.num_robots}",
                f"Episode: {self.current_episode}/{self.total_episodes}",
                f"Step: {self.current_step}",
            ]
            
            # Add package info for shared mode
            if self.shared_packages:
                remaining = len(self.available_packages)
                stats.append(f"Packages: {self.packages_collected}/{self.num_packages}")
                stats.append(f"Remaining: {remaining}")
            
            stats.extend([
                "",
                f"Epsilon: {self.agent.epsilon:.3f}",
                f"Q-table: {self.agent.get_qtable_size()} states",
                "",
            ])
            
            if self.recent_rewards:
                avg_reward = np.mean(list(self.recent_rewards))
                stats.append(f"Avg Reward: {avg_reward:.1f}")
            
            if self.recent_successes:
                success_rate = np.mean(list(self.recent_successes)) * 100
                stats.append(f"Success Rate: {success_rate:.1f}%")
        
        for stat in stats:
            if stat:
                text = self.font_small.render(stat, True, self.COLOR_TEXT)
                self.screen.blit(text, (x, y))
            y += line_height
        
        # Robot status
        y += 10
        robot_title = self.font_small.render("Robot Status:", True, self.COLOR_TEXT)
        self.screen.blit(robot_title, (x, y))
        y += line_height
        
        with self.lock:
            for i in range(self.num_robots):
                done = self.robot_dones[i] if i < len(self.robot_dones) else False
                deadlocked = i in self.deadlocked_robots
                
                # Get robot's package count if in shared mode
                pkg_count = ""
                if self.shared_packages and i < len(self.env.robots):
                    pkg_count = f" [{self.env.robots[i].packages_collected}]"
                
                if done:
                    status = f"✓ Done{pkg_count}"
                    color = self.COLOR_SUCCESS
                elif deadlocked:
                    status = f"⚠ Stuck{pkg_count}"
                    color = self.COLOR_FAIL
                else:
                    action = self.last_actions[i] if i < len(self.last_actions) else -1
                    action_name = ACTION_NAMES.get(action, "?")
                    status = f"→ {action_name}{pkg_count}"
                    color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                
                # Robot color indicator
                indicator_rect = pygame.Rect(x, y + 3, 12, 12)
                pygame.draw.rect(self.screen, ROBOT_COLORS[i % len(ROBOT_COLORS)], 
                               indicator_rect, border_radius=2)
                
                text = self.font_tiny.render(f"R{i}: {status}", True, color)
                self.screen.blit(text, (x + 18, y))
                y += 18
        
        # Message
        y = self.toolbar_height + self.grid_area_height - 50
        if self.last_message:
            msg_color = self.COLOR_SUCCESS if "done" in self.last_message.lower() else self.COLOR_TEXT_DIM
            msg_text = self.font_tiny.render(self.last_message[:35], True, msg_color)
            self.screen.blit(msg_text, (x, y))
    
    def draw_graph(self):
        """Draw training progress graph."""
        graph_y = self.window_height - self.graph_height
        graph_rect = pygame.Rect(0, graph_y, self.window_width, self.graph_height)
        pygame.draw.rect(self.screen, self.COLOR_GRAPH_BG, graph_rect)
        
        title = self.font_small.render("Training Progress", True, self.COLOR_TEXT)
        self.screen.blit(title, (10, graph_y + 5))
        
        with self.lock:
            if len(self.episode_rewards) > 1:
                graph_left = 50
                graph_right = self.window_width - 20
                graph_top = graph_y + 28
                graph_bottom = self.window_height - 15
                graph_width = graph_right - graph_left
                graph_height = graph_bottom - graph_top
                
                rewards = list(self.episode_rewards)
                if len(rewards) > graph_width:
                    step = len(rewards) / graph_width
                    rewards = [rewards[int(i * step)] for i in range(int(graph_width))]
                
                min_r = min(rewards) if rewards else 0
                max_r = max(rewards) if rewards else 1
                range_r = max_r - min_r if max_r != min_r else 1
                
                # Axes
                pygame.draw.line(self.screen, self.COLOR_TEXT_DIM,
                               (graph_left, graph_bottom), (graph_right, graph_bottom), 1)
                pygame.draw.line(self.screen, self.COLOR_TEXT_DIM,
                               (graph_left, graph_top), (graph_left, graph_bottom), 1)
                
                # Reward line
                points = []
                for i, r in enumerate(rewards):
                    px = graph_left + (i / max(1, len(rewards) - 1)) * graph_width
                    py = graph_bottom - ((r - min_r) / range_r) * graph_height
                    points.append((px, py))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.COLOR_GRAPH_LINE, False, points, 2)
                
                # Labels
                max_label = self.font_tiny.render(f"{max_r:.0f}", True, self.COLOR_TEXT_DIM)
                min_label = self.font_tiny.render(f"{min_r:.0f}", True, self.COLOR_TEXT_DIM)
                self.screen.blit(max_label, (5, graph_top))
                self.screen.blit(min_label, (5, graph_bottom - 12))
    
    def render(self):
        """Render complete UI."""
        self.screen.fill(self.COLOR_BG)
        
        self.draw_toolbar()
        self.draw_grid()
        self.draw_stats_panel()
        self.draw_graph()
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def run(self):
        """Main loop."""
        print("=" * 65)
        print("  Multi-Robot RL Training Visualizer")
        print("=" * 65)
        print("\nControls:")
        print("  SPACE      - Start/Stop training")
        print("  T          - Test trained agents")
        print("  R          - Reset agents")
        print("  +/-        - Adjust visualization speed")
        print("  LEFT/RIGHT - Adjust episode count")
        print("  UP/DOWN    - Adjust number of robots")
        print("  ESC        - Exit")
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


def main():
    """Run the multi-robot RL UI."""
    ui = MultiRobotRLUI(grid_size=12, num_robots=3)
    ui.run()


if __name__ == "__main__":
    main()


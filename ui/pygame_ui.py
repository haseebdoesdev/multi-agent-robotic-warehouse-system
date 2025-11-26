import pygame
import random
from typing import List, Dict, Tuple, Optional
from warehouse.environment import Warehouse
from warehouse.robot import Robot


class PygameUI:
    """Interactive Pygame UI with mouse-based editing capabilities."""
    
    # Colors
    COLOR_BACKGROUND = (255, 255, 255)
    COLOR_GRID_LINE = (200, 200, 200)
    COLOR_OBSTACLE = (80, 80, 80)
    COLOR_PACKAGE = (0, 200, 0)
    COLOR_PATH = (200, 220, 255)
    COLOR_TEXT = (0, 0, 0)
    COLOR_PANEL = (240, 240, 240)
    COLOR_TOOLBAR = (50, 50, 60)
    COLOR_BUTTON_ACTIVE = (100, 180, 100)
    COLOR_BUTTON_INACTIVE = (80, 80, 90)
    COLOR_HOVER = (255, 255, 150, 100)
    
    ROBOT_COLORS = [
        (255, 50, 50),    # Red
        (50, 100, 255),   # Blue
        (255, 200, 0),    # Yellow
        (0, 200, 200),    # Cyan
        (200, 50, 200),   # Magenta
        (255, 150, 50),   # Orange
        (100, 255, 100),  # Light Green
        (150, 100, 255),  # Purple
    ]
    
    # Edit modes
    MODE_RUN = 'run'
    MODE_OBSTACLE = 'obstacle'
    MODE_PACKAGE = 'package'
    MODE_ROBOT = 'robot'
    MODE_ERASE = 'erase'
    
    def __init__(self, warehouse: Warehouse, robots: List[Robot], 
                 window_size: int = 800, fps: int = 4):
        self.warehouse = warehouse
        self.robots = robots
        self.window_size = window_size
        self.base_fps = fps
        self.fps = fps
        
        # Layout
        self.toolbar_height = 50
        self.panel_height = 100
        self.grid_size = window_size - self.panel_height - self.toolbar_height
        self.cell_size = self.grid_size // max(warehouse.width, warehouse.height)
        
        self.grid_width = self.cell_size * warehouse.width
        self.grid_height = self.cell_size * warehouse.height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((
            max(self.grid_width, 600), 
            self.grid_height + self.panel_height + self.toolbar_height
        ))
        pygame.display.set_caption("Multi-Agent Warehouse - Interactive Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)
        
        # State
        self.running = True
        self.paused = True  # Start paused for setup
        self.edit_mode = self.MODE_OBSTACLE
        self.show_paths = True
        self.show_help = False
        self.show_settings = False
        self.mouse_held = False
        self.hover_cell: Optional[Tuple[int, int]] = None
        self.hover_button: Optional[Dict] = None  # For button hover effect
        self.next_robot_id = len(robots)
        self.step_once = False  # For step-by-step mode
        
        # Grid size options
        self.grid_sizes = [8, 10, 12, 15, 20, 25, 30]
        self.current_grid_size_idx = self.grid_sizes.index(warehouse.width) if warehouse.width in self.grid_sizes else 1
        
        # FPS options
        self.fps_options = [1, 2, 4, 6, 8, 10, 15, 20, 30]
        self.current_fps_idx = self.fps_options.index(fps) if fps in self.fps_options else 2
        
        # Track placed robots for editing
        self.placed_robots: List[Robot] = list(robots)
        
        # Callback for grid resize (set by main_gui.py)
        self.on_grid_resize = None
        
        # Toolbar buttons
        self.buttons = self._create_buttons()
    
    def _create_buttons(self) -> List[Dict]:
        """Create toolbar buttons."""
        buttons = []
        button_width = 65
        button_height = 35
        spacing = 3
        x = spacing
        y = 8
        
        button_defs = [
            ('RUN', self.MODE_RUN, pygame.K_SPACE),
            ('Obstacle', self.MODE_OBSTACLE, pygame.K_1),
            ('Package', self.MODE_PACKAGE, pygame.K_2),
            ('Robot', self.MODE_ROBOT, pygame.K_3),
            ('Erase', self.MODE_ERASE, pygame.K_4),
            ('Random', 'random', pygame.K_r),
            ('Clear', 'clear', pygame.K_c),
            ('Settings', 'settings', pygame.K_s),
            ('Help', 'help', pygame.K_h),
        ]
        
        for label, mode, key in button_defs:
            buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'label': label,
                'mode': mode,
                'key': key
            })
            x += button_width + spacing
        
        return buttons
    
    def _get_cell_from_mouse(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to grid cell coordinates."""
        mx, my = mouse_pos
        grid_top = self.toolbar_height + self.panel_height
        
        if my < grid_top or my >= grid_top + self.grid_height:
            return None
        if mx < 0 or mx >= self.grid_width:
            return None
        
        cell_x = mx // self.cell_size
        cell_y = (my - grid_top) // self.cell_size
        
        if 0 <= cell_x < self.warehouse.width and 0 <= cell_y < self.warehouse.height:
            return (cell_x, cell_y)
        return None
    
    def _handle_cell_click(self, cell: Tuple[int, int], button: int):
        """Handle click on a grid cell."""
        x, y = cell
        
        if self.edit_mode == self.MODE_RUN:
            return
        
        # Check if there's a robot at this position
        robot_at_cell = None
        for robot in self.placed_robots:
            if robot.position == cell:
                robot_at_cell = robot
                break
        
        if self.edit_mode == self.MODE_OBSTACLE:
            if button == 1:  # Left click - add obstacle
                if not robot_at_cell and cell not in self.warehouse.packages:
                    self.warehouse.grid[y, x] = Warehouse.OBSTACLE
            elif button == 3:  # Right click - remove obstacle
                if self.warehouse.grid[y, x] == Warehouse.OBSTACLE:
                    self.warehouse.grid[y, x] = Warehouse.EMPTY
        
        elif self.edit_mode == self.MODE_PACKAGE:
            if button == 1:  # Left click - add package
                if self.warehouse.grid[y, x] == Warehouse.EMPTY and not robot_at_cell:
                    self.warehouse.grid[y, x] = Warehouse.PACKAGE
                    if cell not in self.warehouse.packages:
                        self.warehouse.packages.append(cell)
            elif button == 3:  # Right click - remove package
                if cell in self.warehouse.packages:
                    self.warehouse.grid[y, x] = Warehouse.EMPTY
                    self.warehouse.packages.remove(cell)
        
        elif self.edit_mode == self.MODE_ROBOT:
            if button == 1:  # Left click - add robot
                if (self.warehouse.grid[y, x] != Warehouse.OBSTACLE and 
                    not robot_at_cell):
                    new_robot = Robot(robot_id=self.next_robot_id, start_position=cell)
                    self.placed_robots.append(new_robot)
                    self.robots.append(new_robot)
                    self.next_robot_id += 1
            elif button == 3:  # Right click - remove robot
                if robot_at_cell:
                    self.placed_robots.remove(robot_at_cell)
                    self.robots.remove(robot_at_cell)
        
        elif self.edit_mode == self.MODE_ERASE:
            if button == 1:  # Left click - erase anything
                # Remove robot
                if robot_at_cell:
                    self.placed_robots.remove(robot_at_cell)
                    self.robots.remove(robot_at_cell)
                # Remove package
                if cell in self.warehouse.packages:
                    self.warehouse.packages.remove(cell)
                # Clear cell
                self.warehouse.grid[y, x] = Warehouse.EMPTY
    
    def _handle_button_click(self, button: Dict):
        """Handle toolbar button click."""
        mode = button['mode']
        
        if mode == 'clear':
            self._clear_grid()
        elif mode == 'random':
            self._generate_random()
        elif mode == 'help':
            self.show_help = not self.show_help
            self.show_settings = False
        elif mode == 'settings':
            self.show_settings = not self.show_settings
            self.show_help = False
        elif mode == self.MODE_RUN:
            self.paused = not self.paused
            if not self.paused:
                self.edit_mode = self.MODE_RUN
                self.show_settings = False
                self.show_help = False
        else:
            self.edit_mode = mode
            if mode != self.MODE_RUN:
                self.paused = True
    
    def _clear_grid(self):
        """Clear all obstacles, packages, and robots."""
        # Clear grid
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                self.warehouse.grid[y, x] = Warehouse.EMPTY
        
        # Clear packages
        self.warehouse.packages.clear()
        
        # Clear robots
        self.robots.clear()
        self.placed_robots.clear()
        self.next_robot_id = 0
    
    def _adjust_fps(self, delta: int):
        """Adjust FPS and sync with settings index."""
        self.fps = max(1, min(30, self.fps + delta))
        # Sync with fps_options index
        if self.fps in self.fps_options:
            self.current_fps_idx = self.fps_options.index(self.fps)
        else:
            # Find closest
            closest_idx = 0
            closest_diff = abs(self.fps_options[0] - self.fps)
            for i, fps_val in enumerate(self.fps_options):
                diff = abs(fps_val - self.fps)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
            self.current_fps_idx = closest_idx
    
    def _generate_random(self):
        
        # Clear everything first
        self._clear_grid()
        
        width = self.warehouse.width
        height = self.warehouse.height
        
        # Random parameters
        obstacle_density = random.uniform(0.10, 0.25)
        num_packages = random.randint(3, max(3, min(10, (width * height) // 20)))
        num_robots = random.randint(2, max(2, min(5, (width * height) // 30)))
        
        # Place random obstacles
        num_obstacles = int(width * height * obstacle_density)
        
        placed_obstacles = 0
        attempts = 0
        while placed_obstacles < num_obstacles and attempts < num_obstacles * 3:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            if self.warehouse.grid[y, x] == Warehouse.EMPTY:
                self.warehouse.grid[y, x] = Warehouse.OBSTACLE
                placed_obstacles += 1
            attempts += 1
        
        # Place random packages
        placed_packages = 0
        attempts = 0
        while placed_packages < num_packages and attempts < num_packages * 10:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            if self.warehouse.grid[y, x] == Warehouse.EMPTY:
                self.warehouse.grid[y, x] = Warehouse.PACKAGE
                self.warehouse.packages.append((x, y))
                placed_packages += 1
            attempts += 1
        
        # Place robots at RANDOM valid positions
        used_positions = set()
        for i in range(num_robots):
            pos = None
            attempts = 0
            
            # Try to find a random valid position
            while attempts < 100:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                
                if ((x, y) not in used_positions and 
                    self.warehouse.is_valid_move(x, y, ignore_packages=True)):
                    pos = (x, y)
                    break
                attempts += 1
            
            if pos is None:
                # Fallback: find any valid position
                for y in range(height):
                    for x in range(width):
                        if ((x, y) not in used_positions and 
                            self.warehouse.is_valid_move(x, y, ignore_packages=True)):
                            pos = (x, y)
                            break
                    if pos:
                        break
            
            if pos:
                new_robot = Robot(robot_id=self.next_robot_id, start_position=pos)
                self.placed_robots.append(new_robot)
                self.robots.append(new_robot)
                used_positions.add(pos)
                self.next_robot_id += 1
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if not self.paused:
                        self.edit_mode = self.MODE_RUN
                elif event.key == pygame.K_1:
                    self.edit_mode = self.MODE_OBSTACLE
                    self.paused = True
                    self.show_help = False
                    self.show_settings = False
                elif event.key == pygame.K_2:
                    self.edit_mode = self.MODE_PACKAGE
                    self.paused = True
                    self.show_help = False
                    self.show_settings = False
                elif event.key == pygame.K_3:
                    self.edit_mode = self.MODE_ROBOT
                    self.paused = True
                    self.show_help = False
                    self.show_settings = False
                elif event.key == pygame.K_4:
                    self.edit_mode = self.MODE_ERASE
                    self.paused = True
                    self.show_help = False
                    self.show_settings = False
                elif event.key == pygame.K_c:
                    self._clear_grid()
                elif event.key == pygame.K_r:
                    self._generate_random()
                elif event.key == pygame.K_s:
                    self.show_settings = not self.show_settings
                    self.show_help = False
                elif event.key == pygame.K_p:
                    self.show_paths = not self.show_paths
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    self.show_settings = False
                elif event.key == pygame.K_PERIOD:
                    # Step mode: advance one step when paused
                    if self.paused:
                        self.step_once = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._adjust_fps(1)
                elif event.key == pygame.K_MINUS:
                    self._adjust_fps(-1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check settings overlay first
                if self.show_settings and self._handle_settings_click(mouse_pos):
                    continue
                
                # Check toolbar buttons
                for button in self.buttons:
                    if button['rect'].collidepoint(mouse_pos):
                        self._handle_button_click(button)
                        break
                else:
                    # Check grid cell (only if not showing overlays)
                    if not self.show_settings and not self.show_help:
                        cell = self._get_cell_from_mouse(mouse_pos)
                        if cell:
                            self._handle_cell_click(cell, event.button)
                            self.mouse_held = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_held = False
            
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                self.hover_cell = self._get_cell_from_mouse(mouse_pos)
                
                # Check button hover
                self.hover_button = None
                for button in self.buttons:
                    if button['rect'].collidepoint(mouse_pos):
                        self.hover_button = button
                        break
                
                # Handle drag for obstacles
                if self.mouse_held and self.edit_mode in [self.MODE_OBSTACLE, self.MODE_ERASE]:
                    cell = self._get_cell_from_mouse(mouse_pos)
                    if cell:
                        buttons = pygame.mouse.get_pressed()
                        if buttons[0]:
                            self._handle_cell_click(cell, 1)
                        elif buttons[2]:
                            self._handle_cell_click(cell, 3)
        
        return True
    
    def draw_toolbar(self):
        """Draw the toolbar with buttons."""
        # Toolbar background
        toolbar_rect = pygame.Rect(0, 0, self.screen.get_width(), self.toolbar_height)
        pygame.draw.rect(self.screen, self.COLOR_TOOLBAR, toolbar_rect)
        
        # Draw buttons
        for button in self.buttons:
            is_active = False
            is_hovered = (self.hover_button == button)
            
            if button['mode'] == self.edit_mode:
                is_active = True
            elif button['mode'] == self.MODE_RUN and not self.paused:
                is_active = True
            elif button['mode'] == 'help' and self.show_help:
                is_active = True
            elif button['mode'] == 'settings' and self.show_settings:
                is_active = True
            
            # Determine button color
            if is_active:
                color = self.COLOR_BUTTON_ACTIVE
            elif is_hovered:
                color = (100, 100, 120)  # Hover color
            else:
                color = self.COLOR_BUTTON_INACTIVE
            
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=5)
            
            # Border - brighter when hovered
            border_color = (200, 200, 200) if is_hovered else (150, 150, 150)
            pygame.draw.rect(self.screen, border_color, button['rect'], 1, border_radius=5)
            
            # Button label
            label_text = button['label']
            if button['mode'] == self.MODE_RUN:
                label_text = "PAUSE" if not self.paused else "RUN"
            
            label = self.small_font.render(label_text, True, (255, 255, 255))
            label_rect = label.get_rect(center=button['rect'].center)
            self.screen.blit(label, label_rect)
        
        # Mode indicator with step hint
        mode_text = f"Mode: {self.edit_mode.upper()}"
        if self.paused:
            mode_text += " [.]Step"
        mode_surface = self.small_font.render(mode_text, True, (200, 200, 200))
        self.screen.blit(mode_surface, (self.screen.get_width() - 180, 15))
    
    def draw_panel(self, timestep: int, stats: Dict):
        """Draw the info panel."""
        panel_top = self.toolbar_height
        panel_rect = pygame.Rect(0, panel_top, self.screen.get_width(), self.panel_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        pygame.draw.line(
            self.screen, self.COLOR_GRID_LINE,
            (0, panel_top + self.panel_height), 
            (self.screen.get_width(), panel_top + self.panel_height), 2
        )
        
        # Title
        title = self.font.render("Multi-Agent Warehouse Simulation", True, self.COLOR_TEXT)
        self.screen.blit(title, (10, panel_top + 8))
        
        # Stats
        y_offset = panel_top + 35
        col1_x = 10
        col2_x = 200
        col3_x = 400
        
        stats_col1 = [
            f"Timestep: {timestep}",
            f"Packages: {len(self.warehouse.packages)}",
            f"Robots: {len(self.robots)}",
        ]
        
        stats_col2 = [
            f"Collected: {stats.get('total_packages_collected', 0)}",
            f"Distance: {stats.get('total_distance_traveled', 0)}",
            f"Conflicts: {stats.get('conflicts_resolved', 0)}",
        ]
        
        stats_col3 = [
            f"FPS: {self.fps}",
            f"Paths: {'ON' if self.show_paths else 'OFF'}",
        ]
        
        for i, text in enumerate(stats_col1):
            surface = self.small_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (col1_x, y_offset + i * 20))
        
        for i, text in enumerate(stats_col2):
            surface = self.small_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (col2_x, y_offset + i * 20))
        
        for i, text in enumerate(stats_col3):
            surface = self.small_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (col3_x, y_offset + i * 20))
        
        # Robot status
        robot_x = 520
        for i, robot in enumerate(self.robots[:5]):  # Show max 5 robots
            color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
            square_rect = pygame.Rect(robot_x, y_offset + i * 18, 12, 12)
            pygame.draw.rect(self.screen, color, square_rect)
            
            status_text = f"R{robot.id}: {robot.status[:8]}"
            surface = self.tiny_font.render(status_text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (robot_x + 16, y_offset + i * 18))
    
    def draw_grid(self):
        """Draw the warehouse grid."""
        grid_top = self.toolbar_height + self.panel_height
        
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size + grid_top,
                    self.cell_size,
                    self.cell_size
                )
                
                cell_value = self.warehouse.get_cell_value(x, y)
                if cell_value == Warehouse.OBSTACLE:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_BACKGROUND, rect)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)
        
        # Draw hover highlight
        if self.hover_cell and self.edit_mode != self.MODE_RUN:
            hx, hy = self.hover_cell
            hover_rect = pygame.Rect(
                hx * self.cell_size,
                hy * self.cell_size + grid_top,
                self.cell_size,
                self.cell_size
            )
            hover_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            
            # Color based on mode
            if self.edit_mode == self.MODE_OBSTACLE:
                hover_surface.fill((100, 100, 100, 100))
            elif self.edit_mode == self.MODE_PACKAGE:
                hover_surface.fill((0, 200, 0, 100))
            elif self.edit_mode == self.MODE_ROBOT:
                hover_surface.fill((255, 100, 100, 100))
            elif self.edit_mode == self.MODE_ERASE:
                hover_surface.fill((255, 50, 50, 100))
            
            self.screen.blit(hover_surface, hover_rect.topleft)
            
            # Draw coordinate tooltip
            coord_text = f"({hx}, {hy})"
            coord_surface = self.tiny_font.render(coord_text, True, (50, 50, 50))
            coord_bg = pygame.Surface((coord_surface.get_width() + 6, coord_surface.get_height() + 4), pygame.SRCALPHA)
            coord_bg.fill((255, 255, 200, 220))
            
            # Position tooltip near cursor but within screen bounds
            tooltip_x = min(hover_rect.right + 5, self.screen.get_width() - coord_bg.get_width() - 5)
            tooltip_y = hover_rect.top
            
            self.screen.blit(coord_bg, (tooltip_x, tooltip_y))
            self.screen.blit(coord_surface, (tooltip_x + 3, tooltip_y + 2))
    
    def draw_paths(self):
        """Draw robot paths."""
        if not self.show_paths:
            return
        
        grid_top = self.toolbar_height + self.panel_height
        
        for robot in self.robots:
            if robot.path:
                color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
                path_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                path_surface.fill((*color, 50))
                
                for pos in robot.path:
                    x, y = pos
                    self.screen.blit(
                        path_surface,
                        (x * self.cell_size, y * self.cell_size + grid_top)
                    )
    
    def draw_packages(self):
        """Draw packages."""
        grid_top = self.toolbar_height + self.panel_height
        
        for package_pos in self.warehouse.packages:
            x, y = package_pos
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2 + grid_top
            radius = self.cell_size // 3
            
            # Draw package with border
            pygame.draw.circle(self.screen, self.COLOR_PACKAGE, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, (0, 150, 0), (center_x, center_y), radius, 2)
    
    def draw_robots(self):
        """Draw robots."""
        grid_top = self.toolbar_height + self.panel_height
        
        for robot in self.robots:
            x, y = robot.position
            color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
            
            robot_size = int(self.cell_size * 0.65)
            offset = (self.cell_size - robot_size) // 2
            rect = pygame.Rect(
                x * self.cell_size + offset,
                y * self.cell_size + offset + grid_top,
                robot_size,
                robot_size
            )
            
            # Robot body
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, (30, 30, 30), rect, 2, border_radius=4)
            
            # Robot ID
            label = self.small_font.render(str(robot.id), True, (255, 255, 255))
            label_rect = label.get_rect(center=rect.center)
            self.screen.blit(label, label_rect)
    
    def draw_help_overlay(self):
        """Draw help overlay."""
        if not self.show_help:
            return
        
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        help_text = [
            "KEYBOARD SHORTCUTS",
            "─" * 30,
            "SPACE    - Start/Pause simulation",
            ".        - Step once (when paused)",
            "1        - Obstacle mode (draw walls)",
            "2        - Package mode (place goals)",
            "3        - Robot mode (add/remove robots)",
            "4        - Erase mode",
            "R        - Generate random layout",
            "C        - Clear entire grid",
            "S        - Settings (grid size, FPS)",
            "P        - Toggle path display",
            "+/-      - Increase/decrease speed",
            "H        - Toggle this help",
            "ESC      - Exit",
            "",
            "MOUSE CONTROLS",
            "─" * 30,
            "Left Click   - Place item",
            "Right Click  - Remove item",
            "Drag         - Draw obstacles",
            "",
            "Press H to close"
        ]
        
        y = 100
        for line in help_text:
            if line.startswith("─"):
                color = (100, 100, 100)
            elif line.isupper() and line:
                color = (100, 200, 255)
            else:
                color = (220, 220, 220)
            
            text = self.font.render(line, True, color)
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, y))
            self.screen.blit(text, text_rect)
            y += 28
    
    def draw_settings_overlay(self):
        """Draw settings overlay with grid size and FPS options."""
        if not self.show_settings:
            return
        
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        center_x = self.screen.get_width() // 2
        
        # Title
        title = self.font.render("SETTINGS", True, (100, 200, 255))
        title_rect = title.get_rect(center=(center_x, 80))
        self.screen.blit(title, title_rect)
        
        # Grid Size Section
        y = 140
        grid_label = self.font.render("Grid Size:", True, (220, 220, 220))
        self.screen.blit(grid_label, (center_x - 150, y))
        
        # Grid size buttons
        btn_width = 45
        btn_height = 35
        btn_y = y + 35
        start_x = center_x - (len(self.grid_sizes) * (btn_width + 5)) // 2
        
        self.grid_size_buttons = []
        for i, size in enumerate(self.grid_sizes):
            btn_x = start_x + i * (btn_width + 5)
            btn_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)
            
            is_selected = (i == self.current_grid_size_idx)
            color = (100, 180, 100) if is_selected else (80, 80, 90)
            
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=5)
            pygame.draw.rect(self.screen, (150, 150, 150), btn_rect, 1, border_radius=5)
            
            label = self.small_font.render(str(size), True, (255, 255, 255))
            label_rect = label.get_rect(center=btn_rect.center)
            self.screen.blit(label, label_rect)
            
            self.grid_size_buttons.append({'rect': btn_rect, 'size': size, 'index': i})
        
        # FPS Section
        y = 240
        fps_label = self.font.render("Simulation Speed (FPS):", True, (220, 220, 220))
        self.screen.blit(fps_label, (center_x - 150, y))
        
        # FPS buttons
        btn_y = y + 35
        start_x = center_x - (len(self.fps_options) * (btn_width + 5)) // 2
        
        self.fps_buttons = []
        for i, fps_val in enumerate(self.fps_options):
            btn_x = start_x + i * (btn_width + 5)
            btn_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)
            
            is_selected = (i == self.current_fps_idx)
            color = (100, 180, 100) if is_selected else (80, 80, 90)
            
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=5)
            pygame.draw.rect(self.screen, (150, 150, 150), btn_rect, 1, border_radius=5)
            
            label = self.small_font.render(str(fps_val), True, (255, 255, 255))
            label_rect = label.get_rect(center=btn_rect.center)
            self.screen.blit(label, label_rect)
            
            self.fps_buttons.append({'rect': btn_rect, 'fps': fps_val, 'index': i})
        
        # Current values display
        y = 340
        current_text = f"Current: {self.warehouse.width}x{self.warehouse.height} grid, {self.fps} FPS"
        current_surface = self.font.render(current_text, True, (150, 200, 150))
        current_rect = current_surface.get_rect(center=(center_x, y))
        self.screen.blit(current_surface, current_rect)
        
        # Instructions
        y = 400
        note1 = self.small_font.render("Click a grid size to change (clears current layout)", True, (180, 180, 180))
        note1_rect = note1.get_rect(center=(center_x, y))
        self.screen.blit(note1, note1_rect)
        
        y = 425
        note2 = self.small_font.render("Click FPS to change simulation speed", True, (180, 180, 180))
        note2_rect = note2.get_rect(center=(center_x, y))
        self.screen.blit(note2, note2_rect)
        
        y = 470
        close_text = self.small_font.render("Press S or click Settings to close", True, (120, 120, 120))
        close_rect = close_text.get_rect(center=(center_x, y))
        self.screen.blit(close_text, close_rect)
    
    def _handle_settings_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle clicks on settings overlay. Returns True if a button was clicked."""
        if not self.show_settings:
            return False
        
        # Check grid size buttons
        if hasattr(self, 'grid_size_buttons'):
            for btn in self.grid_size_buttons:
                if btn['rect'].collidepoint(mouse_pos):
                    new_size = btn['size']
                    if new_size != self.warehouse.width:
                        self.current_grid_size_idx = btn['index']
                        self._resize_grid(new_size)
                    return True
        
        # Check FPS buttons
        if hasattr(self, 'fps_buttons'):
            for btn in self.fps_buttons:
                if btn['rect'].collidepoint(mouse_pos):
                    self.current_fps_idx = btn['index']
                    self.fps = btn['fps']
                    return True
        
        return False
    
    def _resize_grid(self, new_size: int):
        """Resize the grid to a new size."""
        import numpy as np
        
        # Clear current state
        self._clear_grid()
        
        # Resize warehouse grid
        self.warehouse.width = new_size
        self.warehouse.height = new_size
        self.warehouse.grid = np.zeros((new_size, new_size), dtype=int)
        
        # Recalculate cell size
        self.cell_size = self.grid_size // max(self.warehouse.width, self.warehouse.height)
        self.grid_width = self.cell_size * self.warehouse.width
        self.grid_height = self.cell_size * self.warehouse.height
        
        # Resize window
        new_window_width = max(self.grid_width, 600)
        new_window_height = self.grid_height + self.panel_height + self.toolbar_height
        self.screen = pygame.display.set_mode((new_window_width, new_window_height))
        
        # Notify callback if set
        if self.on_grid_resize:
            self.on_grid_resize(new_size)
    
    def render(self, timestep: int, stats: Dict):
        """Render the complete UI."""
        self.screen.fill(self.COLOR_BACKGROUND)
        
        self.draw_toolbar()
        self.draw_panel(timestep, stats)
        self.draw_grid()
        self.draw_paths()
        self.draw_packages()
        self.draw_robots()
        self.draw_help_overlay()
        self.draw_settings_overlay()
        
        pygame.display.flip()
        self.clock.tick(self.fps if not self.paused else 30)
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self.paused
    
    def should_step(self) -> bool:
        """Check if we should execute one step (step mode). Consumes the step flag."""
        if self.step_once:
            self.step_once = False
            return True
        return False
    
    def show_completion_message(self, stats: Dict):
        """Show completion overlay."""
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 220))
        self.screen.blit(overlay, (0, 0))
        
        big_font = pygame.font.Font(None, 48)
        title = big_font.render("SIMULATION COMPLETE!", True, (0, 150, 0))
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, 180))
        self.screen.blit(title, title_rect)
        
        y_offset = 250
        stats_lines = [
            f"Total Timesteps: {stats['timesteps']}",
            f"Packages Collected: {stats['total_packages_collected']}",
            f"Total Distance: {stats['total_distance_traveled']}",
            f"Conflicts Resolved: {stats['conflicts_resolved']}",
        ]
        
        if stats['total_packages_collected'] > 0:
            avg = stats['total_distance_traveled'] / stats['total_packages_collected']
            stats_lines.append(f"Avg Distance/Package: {avg:.2f}")
        
        for i, line in enumerate(stats_lines):
            text = self.font.render(line, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, y_offset + i * 35))
            self.screen.blit(text, text_rect)
        
        exit_text = self.small_font.render("Press SPACE to restart or ESC to exit", True, (100, 100, 100))
        exit_rect = exit_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 50))
        self.screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                    elif event.key == pygame.K_SPACE:
                        waiting = False
                        self.paused = True
                        return True  # Signal restart
            self.clock.tick(30)
        
        return False  # Signal exit
    
    def quit(self):
        """Clean up pygame."""
        pygame.quit()

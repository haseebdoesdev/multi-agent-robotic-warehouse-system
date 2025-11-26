import pygame
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
        self.mouse_held = False
        self.hover_cell: Optional[Tuple[int, int]] = None
        self.next_robot_id = len(robots)
        
        # Track placed robots for editing
        self.placed_robots: List[Robot] = list(robots)
        
        # Toolbar buttons
        self.buttons = self._create_buttons()
    
    def _create_buttons(self) -> List[Dict]:
        """Create toolbar buttons."""
        buttons = []
        button_width = 70
        button_height = 35
        spacing = 4
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
        elif mode == 'help':
            self.show_help = not self.show_help
        elif mode == self.MODE_RUN:
            self.paused = not self.paused
            if not self.paused:
                self.edit_mode = self.MODE_RUN
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
                elif event.key == pygame.K_2:
                    self.edit_mode = self.MODE_PACKAGE
                    self.paused = True
                elif event.key == pygame.K_3:
                    self.edit_mode = self.MODE_ROBOT
                    self.paused = True
                elif event.key == pygame.K_4:
                    self.edit_mode = self.MODE_ERASE
                    self.paused = True
                elif event.key == pygame.K_c:
                    self._clear_grid()
                elif event.key == pygame.K_p:
                    self.show_paths = not self.show_paths
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(30, self.fps + 1)
                elif event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check toolbar buttons
                for button in self.buttons:
                    if button['rect'].collidepoint(mouse_pos):
                        self._handle_button_click(button)
                        break
                else:
                    # Check grid cell
                    cell = self._get_cell_from_mouse(mouse_pos)
                    if cell:
                        self._handle_cell_click(cell, event.button)
                        self.mouse_held = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_held = False
            
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                self.hover_cell = self._get_cell_from_mouse(mouse_pos)
                
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
            if button['mode'] == self.edit_mode:
                is_active = True
            elif button['mode'] == self.MODE_RUN and not self.paused:
                is_active = True
            elif button['mode'] == 'help' and self.show_help:
                is_active = True
            
            color = self.COLOR_BUTTON_ACTIVE if is_active else self.COLOR_BUTTON_INACTIVE
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=5)
            pygame.draw.rect(self.screen, (150, 150, 150), button['rect'], 1, border_radius=5)
            
            # Button label
            label_text = button['label']
            if button['mode'] == self.MODE_RUN:
                label_text = "PAUSE" if not self.paused else "RUN"
            
            label = self.small_font.render(label_text, True, (255, 255, 255))
            label_rect = label.get_rect(center=button['rect'].center)
            self.screen.blit(label, label_rect)
        
        # Mode indicator
        mode_text = f"Mode: {self.edit_mode.upper()}"
        if self.paused:
            mode_text += " (PAUSED)"
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
            "1        - Obstacle mode (draw walls)",
            "2        - Package mode (place goals)",
            "3        - Robot mode (add/remove robots)",
            "4        - Erase mode",
            "C        - Clear entire grid",
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
        
        pygame.display.flip()
        self.clock.tick(self.fps if not self.paused else 30)
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self.paused
    
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

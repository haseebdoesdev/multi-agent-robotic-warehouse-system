import pygame
from typing import List, Dict, Tuple
from warehouse.environment import Warehouse
from warehouse.robot import Robot


class PygameUI:
    
    COLOR_BACKGROUND = (255, 255, 255)
    COLOR_GRID_LINE = (200, 200, 200)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_PACKAGE = (0, 200, 0)
    COLOR_PATH = (200, 220, 255)
    COLOR_TEXT = (0, 0, 0)
    COLOR_PANEL = (240, 240, 240)
    
    ROBOT_COLORS = [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    
    def __init__(self, warehouse: Warehouse, robots: List[Robot], 
                 window_size: int = 700, fps: int = 2):
        self.warehouse = warehouse
        self.robots = robots
        self.window_size = window_size
        self.fps = fps
        
        self.panel_height = 120
        self.grid_size = window_size - self.panel_height
        self.cell_size = self.grid_size // max(warehouse.width, warehouse.height)
        
        self.grid_width = self.cell_size * warehouse.width
        self.grid_height = self.cell_size * warehouse.height
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_width, self.grid_height + self.panel_height))
        pygame.display.set_caption("Multi-Agent Warehouse Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        self.running = True
        # Module 5 TODO:
        # - Initialize UI interactivity state here (e.g., show_paths flag,
        #   dynamic FPS, mode toggles). Optionally enable pygame.mixer for sounds.
    
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                # Module 5 TODO:
                # - Map keys to interactions (toggle paths, inc/dec FPS, restart).
                # - Example: 'p' to toggle path overlay; '+'/'-' to adjust speed.
            # Module 5 TODO:
            # - Handle mouse events to place/remove obstacles during runtime.
            #   Translate mouse position to grid cell and toggle obstacle state.
        return True
    
    def draw_grid(self):
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size + self.panel_height,
                    self.cell_size,
                    self.cell_size
                )
                
                cell_value = self.warehouse.get_cell_value(x, y)
                if cell_value == Warehouse.OBSTACLE:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_BACKGROUND, rect)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)
    
    def draw_paths(self):
        for robot in self.robots:
            if robot.path:
                color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
                path_surface = pygame.Surface((self.cell_size, self.cell_size))
                path_surface.set_alpha(50)
                path_surface.fill(color)
                
                for pos in robot.path:
                    x, y = pos
                    self.screen.blit(
                        path_surface,
                        (x * self.cell_size, y * self.cell_size + self.panel_height)
                    )
        # Module 5 TODO:
        # - Add simple animation: fade or pulsing effect along the planned path,
        #   or animate only the frontier/next few steps for clarity.
    
    def draw_packages(self):
        for package_pos in self.warehouse.packages:
            x, y = package_pos
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2 + self.panel_height
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, self.COLOR_PACKAGE, (center_x, center_y), radius)
    
    def draw_robots(self):
        for robot in self.robots:
            x, y = robot.position
            color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
            
            robot_size = int(self.cell_size * 0.6)
            offset = (self.cell_size - robot_size) // 2
            rect = pygame.Rect(
                x * self.cell_size + offset,
                y * self.cell_size + offset + self.panel_height,
                robot_size,
                robot_size
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
            
            label = self.small_font.render(str(robot.id), True, (255, 255, 255))
            label_rect = label.get_rect(center=rect.center)
            self.screen.blit(label, label_rect)
    
    def draw_panel(self, timestep: int, stats: Dict):
        panel_rect = pygame.Rect(0, 0, self.grid_width, self.panel_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        pygame.draw.line(
            self.screen, self.COLOR_GRID_LINE,
            (0, self.panel_height), (self.grid_width, self.panel_height), 2
        )
        
        title = self.font.render("Multi-Agent Warehouse Simulation", True, self.COLOR_TEXT)
        self.screen.blit(title, (10, 10))
        
        y_offset = 40
        stats_text = [
            f"Timestep: {timestep}",
            f"Packages Remaining: {len(self.warehouse.packages)}",
            f"Collected: {stats.get('total_packages_collected', 0)}",
            f"Distance: {stats.get('total_distance_traveled', 0)}",
            f"Conflicts: {stats.get('conflicts_resolved', 0)}"
        ]
        
        for i, text in enumerate(stats_text):
            if i < 3:
                x_pos = 10
                y_pos = y_offset + (i * 25)
            else:
                x_pos = 300
                y_pos = y_offset + ((i - 3) * 25)
            
            surface = self.small_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (x_pos, y_pos))
        
        robot_status_x = 500
        for i, robot_info in enumerate(stats.get('robots', [])):
            robot_id = robot_info['id']
            status = robot_info['status']
            
            color = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]
            
            square_size = 15
            square_rect = pygame.Rect(robot_status_x, y_offset + i * 25, square_size, square_size)
            pygame.draw.rect(self.screen, color, square_rect)
            
            text = f"R{robot_id}: {status}"
            surface = self.small_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (robot_status_x + 20, y_offset + i * 25 - 2))
    
    def render(self, timestep: int, stats: Dict):
        self.screen.fill(self.COLOR_BACKGROUND)
        
        self.draw_panel(timestep, stats)
        self.draw_grid()
        self.draw_paths()
        self.draw_packages()
        self.draw_robots()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def show_completion_message(self, stats: Dict):
        overlay = pygame.Surface((self.grid_width, self.grid_height + self.panel_height))
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))
        
        big_font = pygame.font.Font(None, 48)
        title = big_font.render("SIMULATION COMPLETE!", True, (0, 150, 0))
        title_rect = title.get_rect(center=(self.grid_width // 2, 150))
        self.screen.blit(title, title_rect)
        
        y_offset = 220
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
            text_rect = text.get_rect(center=(self.grid_width // 2, y_offset + i * 35))
            self.screen.blit(text, text_rect)
        
        exit_text = self.small_font.render("Press ESC to exit", True, (100, 100, 100))
        exit_rect = exit_text.get_rect(center=(self.grid_width // 2, self.grid_height + self.panel_height - 30))
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
            self.clock.tick(30)
    
    def quit(self):
        pygame.quit()

import os
import time
from typing import List, Dict
from warehouse.environment import Warehouse
from warehouse.robot import Robot


class TextUI:
    
    COLORS = {
        'reset': '\033[0m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    
    ROBOT_COLORS = ['red', 'blue', 'yellow', 'cyan', 'magenta']
    
    def __init__(self, warehouse: Warehouse, robots: List[Robot], use_colors: bool = True):
        self.warehouse = warehouse
        self.robots = robots
        self.use_colors = use_colors
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def colorize(self, text: str, color: str) -> str:
        if not self.use_colors or color not in self.COLORS:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
    
    def render_grid(self, show_paths: bool = False) -> str:
        display_grid = []
        
        for y in range(self.warehouse.height):
            row = []
            for x in range(self.warehouse.width):
                cell_value = self.warehouse.get_cell_value(x, y)
                
                robot_at_pos = None
                for robot in self.robots:
                    if robot.position == (x, y):
                        robot_at_pos = robot
                        break
                
                if robot_at_pos:
                    robot_label = f"R{robot_at_pos.id}"
                    color = self.ROBOT_COLORS[robot_at_pos.id % len(self.ROBOT_COLORS)]
                    cell_str = self.colorize(robot_label, color)
                elif cell_value == Warehouse.OBSTACLE:
                    cell_str = self.colorize('##', 'gray')
                elif cell_value == Warehouse.PACKAGE:
                    cell_str = self.colorize('P ', 'green')
                else:
                    if show_paths:
                        in_path = False
                        for robot in self.robots:
                            if (x, y) in robot.path:
                                in_path = True
                                break
                        cell_str = self.colorize('· ', 'cyan') if in_path else '. '
                    else:
                        cell_str = '. '
                
                row.append(cell_str)
            display_grid.append(' '.join(row))
        
        return '\n'.join(display_grid)
    
    def render_status(self, timestep: int, stats: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Multi-Agent Warehouse Simulation - Timestep: {timestep}")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append(f"Packages Remaining: {len(self.warehouse.packages)}")
        lines.append(f"Total Packages Collected: {stats.get('total_packages_collected', 0)}")
        lines.append(f"Total Distance Traveled: {stats.get('total_distance_traveled', 0)}")
        lines.append(f"Conflicts Resolved: {stats.get('conflicts_resolved', 0)}")
        lines.append("")
        
        lines.append("Robot Status:")
        for robot_info in stats.get('robots', []):
            robot_id = robot_info['id']
            pos = robot_info['position']
            status = robot_info['status']
            collected = robot_info['packages_collected']
            
            color = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]
            robot_label = self.colorize(f"Robot {robot_id}", color)
            lines.append(f"  {robot_label}: Pos={pos}, Status={status}, Collected={collected}")
        
        lines.append("")
        return '\n'.join(lines)
    
    def render(self, timestep: int, stats: Dict, show_paths: bool = False):
        self.clear_screen()
        print(self.render_status(timestep, stats))
        print(self.render_grid(show_paths))
        print()
    
    def show_completion_message(self, stats: Dict):
        print("\n" + "=" * 60)
        print("  SIMULATION COMPLETE!")
        print("=" * 60)
        print(f"\nTotal Timesteps: {stats['timesteps']}")
        print(f"Packages Collected: {stats['total_packages_collected']}")
        print(f"Total Distance: {stats['total_distance_traveled']}")
        print(f"Conflicts Resolved: {stats['conflicts_resolved']}")
        
        if stats['total_packages_collected'] > 0:
            avg_distance = stats['total_distance_traveled'] / stats['total_packages_collected']
            print(f"Average Distance per Package: {avg_distance:.2f}")
        
        print("\nRobot Performance:")
        for robot_info in stats['robots']:
            print(f"  Robot {robot_info['id']}: "
                  f"{robot_info['packages_collected']} packages, "
                  f"{robot_info['distance_traveled']} distance")
        print()

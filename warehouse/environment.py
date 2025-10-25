import numpy as np
from typing import List, Tuple, Optional


class Warehouse:
    
    EMPTY = 0
    OBSTACLE = 1
    PACKAGE = 2
    
    def __init__(self, width: int = 10, height: int = 10, obstacle_density: float = 0.15):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.packages = []
        self.obstacle_density = obstacle_density
        
    def place_obstacles(self, density: Optional[float] = None, predefined: Optional[List[Tuple[int, int]]] = None):
        if predefined is not None:
            for x, y in predefined:
                if self.is_within_bounds(x, y):
                    self.grid[y, x] = self.OBSTACLE
        else:
            if density is None:
                density = self.obstacle_density
            
            num_obstacles = int(self.width * self.height * density)
            placed = 0
            
            while placed < num_obstacles:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                
                if self.grid[y, x] == self.EMPTY:
                    self.grid[y, x] = self.OBSTACLE
                    placed += 1
    
    def place_packages(self, num_packages: int, locations: Optional[List[Tuple[int, int]]] = None):
        self.packages = []
        
        if locations is not None:
            for x, y in locations[:num_packages]:
                if self.is_within_bounds(x, y) and self.grid[y, x] == self.EMPTY:
                    self.grid[y, x] = self.PACKAGE
                    self.packages.append((x, y))
        else:
            placed = 0
            while placed < num_packages:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                
                if self.grid[y, x] == self.EMPTY:
                    self.grid[y, x] = self.PACKAGE
                    self.packages.append((x, y))
                    placed += 1
    
    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_valid_move(self, x: int, y: int, ignore_packages: bool = False) -> bool:
        if not self.is_within_bounds(x, y):
            return False
        
        cell_value = self.grid[y, x]
        
        if cell_value == self.OBSTACLE:
            return False
        
        if cell_value == self.PACKAGE and not ignore_packages:
            return True
        
        return True
    
    def get_neighbors(self, x: int, y: int, ignore_packages: bool = False) -> List[Tuple[int, int]]:
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_move(nx, ny, ignore_packages):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def remove_package(self, x: int, y: int):
        if self.is_within_bounds(x, y) and self.grid[y, x] == self.PACKAGE:
            self.grid[y, x] = self.EMPTY
            if (x, y) in self.packages:
                self.packages.remove((x, y))
    
    def get_cell_value(self, x: int, y: int) -> int:
        if self.is_within_bounds(x, y):
            return self.grid[y, x]
        return -1
    
    def display(self) -> str:
        symbols = {
            self.EMPTY: '.',
            self.OBSTACLE: '#',
            self.PACKAGE: 'P'
        }
        
        lines = []
        for row in self.grid:
            line = ' '.join(symbols.get(cell, '?') for cell in row)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return f"Warehouse({self.width}x{self.height}, {len(self.packages)} packages)"

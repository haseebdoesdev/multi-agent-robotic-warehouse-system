import heapq
from typing import List, Tuple, Optional, Set
from .environment import Warehouse


class Node:
    
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent: Optional['Node'] = None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.position == other.position
    
    def __hash__(self) -> int:
        return hash(self.position)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def a_star(start: Tuple[int, int], goal: Tuple[int, int], warehouse: Warehouse, 
           heuristic: str = 'manhattan', occupied_cells: Optional[Set[Tuple[int, int]]] = None) -> Optional[List[Tuple[int, int]]]:
    if not warehouse.is_valid_move(start[0], start[1], ignore_packages=True):
        return None
    if not warehouse.is_valid_move(goal[0], goal[1], ignore_packages=True):
        return None
    
    if start == goal:
        return [start]
    
    heuristic_func = manhattan_distance if heuristic == 'manhattan' else euclidean_distance
    # Module 4 TODO:
    # - If using a probabilistic obstacle layer, consider augmenting the
    #   heuristic with a risk penalty term to bias away from risky cells.
    #   Example (later): h_cost = heuristic + lambda * risk(cell)
    
    open_set = []
    closed_set: Set[Tuple[int, int]] = set()
    
    start_node = Node(start, 0, heuristic_func(start, goal))
    heapq.heappush(open_set, start_node)
    
    g_costs = {start: 0}
    
    while open_set:
        current_node = heapq.heappop(open_set)
        current_pos = current_node.position
        
        if current_pos in closed_set:
            continue
        
        closed_set.add(current_pos)
        
        if current_pos == goal:
            path = []
            node = current_node
            while node is not None:
                path.append(node.position)
                node = node.parent
            path.reverse()
            return path
        
        neighbors = warehouse.get_neighbors(current_pos[0], current_pos[1], ignore_packages=True)
        
        for neighbor_pos in neighbors:
            if neighbor_pos in closed_set:
                continue
            
            if occupied_cells and neighbor_pos in occupied_cells and neighbor_pos != goal:
                continue
            
            tentative_g_cost = current_node.g_cost + 1
            
            if neighbor_pos in g_costs and tentative_g_cost >= g_costs[neighbor_pos]:
                continue
            
            g_costs[neighbor_pos] = tentative_g_cost
            
            h_cost = heuristic_func(neighbor_pos, goal)
            neighbor_node = Node(neighbor_pos, tentative_g_cost, h_cost, current_node)
            heapq.heappush(open_set, neighbor_node)
    
    return None


def path_length(path: Optional[List[Tuple[int, int]]]) -> int:
    if path is None:
        return float('inf')
    return len(path) - 1

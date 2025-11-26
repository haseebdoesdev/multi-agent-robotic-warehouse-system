"""
Scalability Experiments Module (Module 6)
-----------------------------------------
Run larger scenarios to assess performance and bottlenecks.
"""

import time
import random
import json
import csv
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warehouse import Warehouse, Robot, CoordinationManager
from metrics import MetricsCollector, compute_metrics, SimulationTrace


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    grid_width: int = 30
    grid_height: int = 30
    num_robots: int = 5
    num_packages: int = 10
    obstacle_density: float = 0.15
    max_timesteps: int = 1000
    random_seed: Optional[int] = 42
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"exp_{self.grid_width}x{self.grid_height}_r{self.num_robots}_p{self.num_packages}"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    success: bool = False
    completion_time: int = 0
    packages_collected: int = 0
    total_distance: int = 0
    conflicts_resolved: int = 0
    efficiency: float = 0.0
    wall_time_seconds: float = 0.0
    per_robot_stats: List[Dict] = field(default_factory=list)
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'config': asdict(self.config),
            'success': self.success,
            'completion_time': self.completion_time,
            'packages_collected': self.packages_collected,
            'total_distance': self.total_distance,
            'conflicts_resolved': self.conflicts_resolved,
            'efficiency': self.efficiency,
            'wall_time_seconds': self.wall_time_seconds,
            'per_robot_stats': self.per_robot_stats,
            'error_message': self.error_message
        }


def run_experiment(config: ExperimentConfig, 
                   verbose: bool = False,
                   collect_trace: bool = False) -> ExperimentResult:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        verbose: Print progress updates
        collect_trace: Collect per-step trace data
    
    Returns:
        ExperimentResult with outcomes
    """
    result = ExperimentResult(config=config)
    start_time = time.time()
    
    try:
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Create warehouse
        warehouse = Warehouse(
            width=config.grid_width,
            height=config.grid_height,
            obstacle_density=config.obstacle_density
        )
        warehouse.place_obstacles()
        warehouse.place_packages(config.num_packages)
        
        # Create robots with distributed starting positions
        robots = []
        start_positions = _generate_start_positions(
            config.grid_width, 
            config.grid_height, 
            config.num_robots,
            warehouse
        )
        
        for i in range(config.num_robots):
            robot = Robot(robot_id=i, start_position=start_positions[i])
            robots.append(robot)
        
        # Create coordinator
        coordinator = CoordinationManager(warehouse)
        for robot in robots:
            coordinator.add_robot(robot)
        
        # Optional trace collection
        metrics_collector = MetricsCollector() if collect_trace else None
        
        # Run simulation
        if verbose:
            print(f"Running experiment: {config.name}")
            print(f"  Grid: {config.grid_width}x{config.grid_height}")
            print(f"  Robots: {config.num_robots}, Packages: {config.num_packages}")
        
        while coordinator.time_step < config.max_timesteps:
            # Assign packages
            if warehouse.packages:
                coordinator.assign_packages(warehouse.packages)
            
            # Plan paths
            coordinator.plan_all_paths()
            
            # Collect metrics if enabled
            if metrics_collector:
                metrics_collector.record_step(
                    coordinator.time_step,
                    robots,
                    warehouse,
                    coordinator.conflict_count
                )
            
            # Check completion
            if coordinator.are_all_packages_collected():
                result.success = True
                break
            
            if coordinator.are_all_robots_idle() and not warehouse.packages:
                result.success = True
                break
            
            # Update robots
            coordinator.update_robots()
        
        # Collect final stats
        final_stats = coordinator.get_statistics()
        
        result.completion_time = final_stats['timesteps']
        result.packages_collected = final_stats['total_packages_collected']
        result.total_distance = final_stats['total_distance_traveled']
        result.conflicts_resolved = final_stats['conflicts_resolved']
        result.per_robot_stats = final_stats['robots']
        
        if result.packages_collected > 0:
            result.efficiency = result.total_distance / result.packages_collected
        
        result.success = (result.packages_collected == config.num_packages)
        
    except Exception as e:
        result.error_message = str(e)
        result.success = False
    
    result.wall_time_seconds = time.time() - start_time
    
    if verbose:
        status = "SUCCESS" if result.success else "INCOMPLETE"
        print(f"  Result: {status}, Time: {result.completion_time} steps, "
              f"Collected: {result.packages_collected}/{config.num_packages}, "
              f"Wall time: {result.wall_time_seconds:.2f}s")
    
    return result


def _generate_start_positions(width: int, height: int, 
                               num_robots: int,
                               warehouse: Warehouse) -> List[Tuple[int, int]]:
    """Generate starting positions distributed around the warehouse edges."""
    positions = []
    used_positions = set()
    corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
    
    for i in range(num_robots):
        if i < len(corners):
            pos = corners[i]
        else:
            # Generate positions along edges
            edge = i % 4
            if edge == 0:  # Top
                pos = (i % width, 0)
            elif edge == 1:  # Right
                pos = (width-1, i % height)
            elif edge == 2:  # Bottom
                pos = (i % width, height-1)
            else:  # Left
                pos = (0, i % height)
        
        # Ensure position is valid and not already used
        if (not warehouse.is_valid_move(pos[0], pos[1], ignore_packages=True) 
            or pos in used_positions):
            # Find nearest valid position
            found = False
            for radius in range(1, max(width, height)):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        new_pos = (max(0, min(width-1, pos[0]+dx)), 
                                   max(0, min(height-1, pos[1]+dy)))
                        if (warehouse.is_valid_move(new_pos[0], new_pos[1], ignore_packages=True)
                            and new_pos not in used_positions):
                            pos = new_pos
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        positions.append(pos)
        used_positions.add(pos)
    
    return positions


def run_sweep(param_grid: Dict[str, List], 
              base_config: ExperimentConfig = None,
              verbose: bool = True,
              output_dir: str = "experiment_results") -> List[ExperimentResult]:
    """
    Run a parameter sweep over multiple configurations.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        base_config: Base configuration to modify
        verbose: Print progress
        output_dir: Directory to save results
    
    Returns:
        List of ExperimentResults
    """
    if base_config is None:
        base_config = ExperimentConfig()
    
    # Generate all configurations
    configs = _generate_configs(param_grid, base_config)
    
    if verbose:
        print(f"Running sweep with {len(configs)} configurations...")
        print("=" * 60)
    
    results = []
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n[{i+1}/{len(configs)}]")
        
        result = run_experiment(config, verbose=verbose)
        results.append(result)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, output_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Sweep complete. Results saved to {output_dir}")
        _print_sweep_summary(results)
    
    return results


def _generate_configs(param_grid: Dict[str, List], 
                      base_config: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate all configurations from parameter grid."""
    import itertools
    
    # Get all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    configs = []
    for combo in combinations:
        # Create new config from base
        config_dict = asdict(base_config)
        
        # Apply parameter values
        for key, value in zip(keys, combo):
            if key in config_dict:
                config_dict[key] = value
        
        # Create config object
        config = ExperimentConfig(**config_dict)
        configs.append(config)
    
    return configs


def save_results(results: List[ExperimentResult], 
                 output_dir: str,
                 format: str = 'both') -> Dict[str, str]:
    """
    Save experiment results to files.
    
    Args:
        results: List of experiment results
        output_dir: Output directory
        format: 'json', 'csv', or 'both'
    
    Returns:
        Dictionary of saved file paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Save JSON
    if format in ['json', 'both']:
        json_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        saved_files['json'] = json_path
    
    # Save CSV
    if format in ['csv', 'both']:
        csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
        
        fieldnames = [
            'name', 'grid_width', 'grid_height', 'num_robots', 'num_packages',
            'obstacle_density', 'success', 'completion_time', 'packages_collected',
            'total_distance', 'conflicts_resolved', 'efficiency', 'wall_time_seconds'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'name': result.config.name,
                    'grid_width': result.config.grid_width,
                    'grid_height': result.config.grid_height,
                    'num_robots': result.config.num_robots,
                    'num_packages': result.config.num_packages,
                    'obstacle_density': result.config.obstacle_density,
                    'success': result.success,
                    'completion_time': result.completion_time,
                    'packages_collected': result.packages_collected,
                    'total_distance': result.total_distance,
                    'conflicts_resolved': result.conflicts_resolved,
                    'efficiency': result.efficiency,
                    'wall_time_seconds': result.wall_time_seconds
                }
                writer.writerow(row)
        
        saved_files['csv'] = csv_path
    
    return saved_files


def _print_sweep_summary(results: List[ExperimentResult]):
    """Print a summary of sweep results."""
    total = len(results)
    successes = sum(1 for r in results if r.success)
    
    print(f"\nSummary:")
    print(f"  Total experiments: {total}")
    print(f"  Successful: {successes} ({successes/total*100:.1f}%)")
    
    if results:
        avg_time = sum(r.completion_time for r in results) / total
        avg_efficiency = sum(r.efficiency for r in results if r.efficiency > 0) / max(1, sum(1 for r in results if r.efficiency > 0))
        avg_wall_time = sum(r.wall_time_seconds for r in results) / total
        
        print(f"  Avg completion time: {avg_time:.1f} steps")
        print(f"  Avg efficiency: {avg_efficiency:.2f} moves/package")
        print(f"  Avg wall time: {avg_wall_time:.2f}s")


def default_scalability_sweep() -> Dict[str, List]:
    """Return default parameter grid for scalability testing."""
    return {
        'grid_width': [30, 40, 50],
        'grid_height': [30, 40, 50],
        'num_robots': [5, 8, 10],
        'obstacle_density': [0.1, 0.2]
    }


def quick_scalability_test(verbose: bool = True) -> List[ExperimentResult]:
    """Run a quick scalability test with fewer configurations."""
    param_grid = {
        'grid_width': [20, 30],
        'grid_height': [20, 30],
        'num_robots': [3, 5],
        'num_packages': [8, 12]
    }
    
    base_config = ExperimentConfig(
        obstacle_density=0.15,
        max_timesteps=500,
        random_seed=42
    )
    
    return run_sweep(param_grid, base_config, verbose=verbose)


def compare_configurations(configs: List[ExperimentConfig],
                           runs_per_config: int = 3,
                           verbose: bool = True) -> Dict[str, List[ExperimentResult]]:
    """
    Compare multiple configurations with multiple runs each.
    
    Args:
        configs: List of configurations to compare
        runs_per_config: Number of runs per configuration
        verbose: Print progress
    
    Returns:
        Dictionary mapping config names to list of results
    """
    comparison = {}
    
    for config in configs:
        if verbose:
            print(f"\nRunning {runs_per_config} trials for: {config.name}")
        
        results = []
        for run in range(runs_per_config):
            # Modify seed for each run
            run_config = ExperimentConfig(**asdict(config))
            run_config.random_seed = (config.random_seed or 42) + run
            
            result = run_experiment(run_config, verbose=False)
            results.append(result)
            
            if verbose:
                status = "OK" if result.success else "FAIL"
                print(f"  Run {run+1}: {status} - {result.completion_time} steps")
        
        comparison[config.name] = results
    
    if verbose:
        print("\n" + "=" * 60)
        print("Comparison Summary:")
        for name, results in comparison.items():
            successes = sum(1 for r in results if r.success)
            avg_time = sum(r.completion_time for r in results) / len(results)
            print(f"  {name}: {successes}/{len(results)} success, avg {avg_time:.1f} steps")
    
    return comparison


# Predefined experiment configurations
SMALL_CONFIG = ExperimentConfig(
    grid_width=15,
    grid_height=15,
    num_robots=3,
    num_packages=5,
    obstacle_density=0.1,
    max_timesteps=300,
    name="small"
)

MEDIUM_CONFIG = ExperimentConfig(
    grid_width=25,
    grid_height=25,
    num_robots=5,
    num_packages=10,
    obstacle_density=0.15,
    max_timesteps=500,
    name="medium"
)

LARGE_CONFIG = ExperimentConfig(
    grid_width=40,
    grid_height=40,
    num_robots=8,
    num_packages=20,
    obstacle_density=0.2,
    max_timesteps=1000,
    name="large"
)

STRESS_CONFIG = ExperimentConfig(
    grid_width=50,
    grid_height=50,
    num_robots=10,
    num_packages=30,
    obstacle_density=0.25,
    max_timesteps=2000,
    name="stress"
)


if __name__ == "__main__":
    print("=" * 60)
    print("  Scalability Experiment Runner")
    print("=" * 60)
    
    # Run quick test
    print("\nRunning quick scalability test...")
    results = quick_scalability_test(verbose=True)
    
    # Or run comparison of predefined configs
    # configs = [SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG]
    # compare_configurations(configs, runs_per_config=3, verbose=True)

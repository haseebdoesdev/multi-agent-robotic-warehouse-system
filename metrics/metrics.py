"""
Metrics computation module (Module 5)
-------------------------------------
Functions to compute additional performance metrics beyond what
CoordinationManager provides.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StepData:
    """Data captured at each simulation step."""
    timestep: int
    robot_positions: List[Tuple[int, int]]
    robot_statuses: List[str]
    robot_targets: List[Optional[Tuple[int, int]]]
    packages_remaining: int
    packages_collected_this_step: int
    conflicts_this_step: int
    replans_this_step: int = 0
    distance_this_step: int = 0


@dataclass 
class SimulationTrace:
    """Container for per-step simulation data."""
    steps: List[StepData] = field(default_factory=list)
    
    def add_step(self, step_data: StepData):
        self.steps.append(step_data)
    
    def get_timesteps(self) -> List[int]:
        return [s.timestep for s in self.steps]
    
    def get_packages_remaining(self) -> List[int]:
        return [s.packages_remaining for s in self.steps]
    
    def get_cumulative_collected(self) -> List[int]:
        cumulative = []
        total = 0
        for s in self.steps:
            total += s.packages_collected_this_step
            cumulative.append(total)
        return cumulative
    
    def get_conflicts_per_step(self) -> List[int]:
        return [s.conflicts_this_step for s in self.steps]
    
    def get_cumulative_conflicts(self) -> List[int]:
        cumulative = []
        total = 0
        for s in self.steps:
            total += s.conflicts_this_step
            cumulative.append(total)
        return cumulative


def compute_metrics(sim_trace: SimulationTrace, 
                    coordinator_stats: Dict,
                    warehouse_info: Dict = None) -> Dict:
    """
    Compute comprehensive metrics from simulation data.
    
    Args:
        sim_trace: Per-step simulation trace data
        coordinator_stats: Final statistics from CoordinationManager
        warehouse_info: Optional warehouse configuration info
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic stats passthrough
    metrics['total_timesteps'] = coordinator_stats.get('timesteps', 0)
    metrics['total_packages_collected'] = coordinator_stats.get('total_packages_collected', 0)
    metrics['total_distance_traveled'] = coordinator_stats.get('total_distance_traveled', 0)
    metrics['total_conflicts'] = coordinator_stats.get('conflicts_resolved', 0)
    
    # Per-robot metrics
    robot_stats = coordinator_stats.get('robots', [])
    metrics['robot_metrics'] = compute_per_robot_metrics(robot_stats, sim_trace)
    
    # Efficiency metrics
    metrics['efficiency'] = compute_efficiency_metrics(
        sim_trace, coordinator_stats, warehouse_info
    )
    
    # Timing metrics
    metrics['timing'] = compute_timing_metrics(sim_trace)
    
    # Congestion metrics
    metrics['congestion'] = compute_congestion_metrics(sim_trace, len(robot_stats))
    
    # Utilization metrics
    metrics['utilization'] = compute_utilization_metrics(sim_trace, len(robot_stats))
    
    return metrics


def compute_per_robot_metrics(robot_stats: List[Dict], 
                               sim_trace: SimulationTrace) -> List[Dict]:
    """Compute metrics for each individual robot."""
    robot_metrics = []
    
    for robot_info in robot_stats:
        robot_id = robot_info['id']
        packages = robot_info.get('packages_collected', 0)
        distance = robot_info.get('distance_traveled', 0)
        
        robot_metric = {
            'id': robot_id,
            'packages_collected': packages,
            'distance_traveled': distance,
            'efficiency': distance / packages if packages > 0 else 0,
            'contribution': 0  # Will be computed below
        }
        
        robot_metrics.append(robot_metric)
    
    # Compute contribution percentages
    total_packages = sum(r['packages_collected'] for r in robot_metrics)
    if total_packages > 0:
        for rm in robot_metrics:
            rm['contribution'] = rm['packages_collected'] / total_packages * 100
    
    return robot_metrics


def compute_efficiency_metrics(sim_trace: SimulationTrace,
                                coordinator_stats: Dict,
                                warehouse_info: Dict = None) -> Dict:
    """Compute efficiency-related metrics."""
    total_packages = coordinator_stats.get('total_packages_collected', 0)
    total_distance = coordinator_stats.get('total_distance_traveled', 0)
    total_timesteps = coordinator_stats.get('timesteps', 0)
    
    efficiency = {
        'moves_per_package': total_distance / total_packages if total_packages > 0 else 0,
        'timesteps_per_package': total_timesteps / total_packages if total_packages > 0 else 0,
        'collection_rate': total_packages / total_timesteps if total_timesteps > 0 else 0,
    }
    
    # Path efficiency (would need ideal path lengths from warehouse)
    if warehouse_info:
        grid_size = warehouse_info.get('width', 10) * warehouse_info.get('height', 10)
        efficiency['grid_utilization'] = total_distance / grid_size
    
    return efficiency


def compute_timing_metrics(sim_trace: SimulationTrace) -> Dict:
    """Compute timing-related metrics."""
    if not sim_trace.steps:
        return {
            'first_collection_time': 0,
            'average_time_between_collections': 0,
            'last_collection_time': 0
        }
    
    collection_times = []
    for i, step in enumerate(sim_trace.steps):
        if step.packages_collected_this_step > 0:
            collection_times.append(step.timestep)
    
    if not collection_times:
        return {
            'first_collection_time': 0,
            'average_time_between_collections': 0,
            'last_collection_time': 0
        }
    
    # Time between collections
    intervals = []
    for i in range(1, len(collection_times)):
        intervals.append(collection_times[i] - collection_times[i-1])
    
    return {
        'first_collection_time': collection_times[0] if collection_times else 0,
        'average_time_between_collections': np.mean(intervals) if intervals else 0,
        'last_collection_time': collection_times[-1] if collection_times else 0,
        'collection_times': collection_times
    }


def compute_congestion_metrics(sim_trace: SimulationTrace, 
                                num_robots: int) -> Dict:
    """Compute congestion and conflict metrics."""
    if not sim_trace.steps:
        return {
            'conflict_frequency': 0,
            'conflicts_per_100_steps': 0,
            'congestion_index': 0,
            'peak_conflict_period': None
        }
    
    total_conflicts = sum(s.conflicts_this_step for s in sim_trace.steps)
    total_steps = len(sim_trace.steps)
    
    # Conflict frequency (conflicts per step per robot pair)
    robot_pairs = num_robots * (num_robots - 1) / 2 if num_robots > 1 else 1
    conflict_frequency = total_conflicts / (total_steps * robot_pairs) if total_steps > 0 else 0
    
    # Conflicts per 100 steps
    conflicts_per_100 = (total_conflicts / total_steps * 100) if total_steps > 0 else 0
    
    # Find peak conflict period (window of 10 steps)
    window_size = min(10, total_steps)
    peak_conflicts = 0
    peak_start = 0
    
    conflicts = sim_trace.get_conflicts_per_step()
    for i in range(len(conflicts) - window_size + 1):
        window_conflicts = sum(conflicts[i:i+window_size])
        if window_conflicts > peak_conflicts:
            peak_conflicts = window_conflicts
            peak_start = i
    
    # Congestion index: normalized measure of conflict density
    congestion_index = conflict_frequency * 100  # Scale for readability
    
    return {
        'conflict_frequency': conflict_frequency,
        'conflicts_per_100_steps': conflicts_per_100,
        'congestion_index': congestion_index,
        'peak_conflict_period': {
            'start': peak_start,
            'end': peak_start + window_size,
            'conflicts': peak_conflicts
        } if peak_conflicts > 0 else None
    }


def compute_utilization_metrics(sim_trace: SimulationTrace,
                                 num_robots: int) -> Dict:
    """Compute robot utilization metrics."""
    if not sim_trace.steps or num_robots == 0:
        return {
            'average_utilization': 0,
            'moving_percentage': 0,
            'idle_percentage': 0,
            'waiting_percentage': 0
        }
    
    status_counts = {
        'moving': 0,
        'idle': 0,
        'waiting': 0,
        'collecting': 0,
        'completed': 0
    }
    
    total_status_records = 0
    
    for step in sim_trace.steps:
        for status in step.robot_statuses:
            status_lower = status.lower()
            if status_lower in status_counts:
                status_counts[status_lower] += 1
            elif 'idle' in status_lower or 'completed' in status_lower:
                status_counts['idle'] += 1
            else:
                status_counts['moving'] += 1
            total_status_records += 1
    
    if total_status_records == 0:
        total_status_records = 1
    
    # Active utilization = time spent moving or collecting
    active_count = status_counts['moving'] + status_counts['collecting']
    
    return {
        'average_utilization': active_count / total_status_records * 100,
        'moving_percentage': status_counts['moving'] / total_status_records * 100,
        'idle_percentage': status_counts['idle'] / total_status_records * 100,
        'waiting_percentage': status_counts['waiting'] / total_status_records * 100,
        'status_breakdown': {k: v / total_status_records * 100 for k, v in status_counts.items()}
    }


def compute_path_efficiency(actual_distances: List[int],
                            optimal_distances: List[int]) -> Dict:
    """
    Compare actual path lengths to optimal (A*) lengths.
    
    Args:
        actual_distances: List of actual distances traveled per task
        optimal_distances: List of optimal distances per task
    
    Returns:
        Path efficiency metrics
    """
    if not actual_distances or not optimal_distances:
        return {
            'average_efficiency': 1.0,
            'total_wasted_distance': 0,
            'efficiency_distribution': []
        }
    
    efficiencies = []
    wasted = 0
    
    for actual, optimal in zip(actual_distances, optimal_distances):
        if actual > 0 and optimal > 0:
            eff = optimal / actual  # 1.0 = perfect, lower = worse
            efficiencies.append(eff)
            wasted += max(0, actual - optimal)
    
    return {
        'average_efficiency': np.mean(efficiencies) if efficiencies else 1.0,
        'min_efficiency': min(efficiencies) if efficiencies else 1.0,
        'max_efficiency': max(efficiencies) if efficiencies else 1.0,
        'total_wasted_distance': wasted,
        'efficiency_distribution': efficiencies
    }


def format_metrics_summary(metrics: Dict) -> str:
    """Format metrics as a readable summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  SIMULATION METRICS SUMMARY")
    lines.append("=" * 60)
    
    # Basic stats
    lines.append(f"\nTotal Timesteps: {metrics['total_timesteps']}")
    lines.append(f"Packages Collected: {metrics['total_packages_collected']}")
    lines.append(f"Total Distance: {metrics['total_distance_traveled']}")
    lines.append(f"Conflicts Resolved: {metrics['total_conflicts']}")
    
    # Efficiency
    eff = metrics.get('efficiency', {})
    lines.append(f"\nEfficiency Metrics:")
    lines.append(f"  Moves per Package: {eff.get('moves_per_package', 0):.2f}")
    lines.append(f"  Timesteps per Package: {eff.get('timesteps_per_package', 0):.2f}")
    lines.append(f"  Collection Rate: {eff.get('collection_rate', 0):.4f} packages/step")
    
    # Timing
    timing = metrics.get('timing', {})
    lines.append(f"\nTiming Metrics:")
    lines.append(f"  First Collection: Step {timing.get('first_collection_time', 0)}")
    lines.append(f"  Avg Time Between Collections: {timing.get('average_time_between_collections', 0):.1f} steps")
    
    # Congestion
    cong = metrics.get('congestion', {})
    lines.append(f"\nCongestion Metrics:")
    lines.append(f"  Conflicts per 100 Steps: {cong.get('conflicts_per_100_steps', 0):.2f}")
    lines.append(f"  Congestion Index: {cong.get('congestion_index', 0):.4f}")
    
    # Utilization
    util = metrics.get('utilization', {})
    lines.append(f"\nUtilization Metrics:")
    lines.append(f"  Active Utilization: {util.get('average_utilization', 0):.1f}%")
    lines.append(f"  Moving: {util.get('moving_percentage', 0):.1f}%")
    lines.append(f"  Idle: {util.get('idle_percentage', 0):.1f}%")
    lines.append(f"  Waiting: {util.get('waiting_percentage', 0):.1f}%")
    
    # Per-robot metrics
    robot_metrics = metrics.get('robot_metrics', [])
    if robot_metrics:
        lines.append(f"\nPer-Robot Performance:")
        for rm in robot_metrics:
            lines.append(f"  Robot {rm['id']}: {rm['packages_collected']} packages, "
                        f"{rm['distance_traveled']} distance, "
                        f"{rm['contribution']:.1f}% contribution")
    
    lines.append("\n" + "=" * 60)
    
    return '\n'.join(lines)


class MetricsCollector:
    """
    Helper class to collect metrics during simulation.
    Integrates with the simulation loop.
    """
    
    def __init__(self):
        self.trace = SimulationTrace()
        self.prev_packages_collected = 0
        self.prev_conflicts = 0
    
    def record_step(self, timestep: int, robots: List, warehouse, 
                    conflicts_total: int = 0, replans: int = 0):
        """Record data for a single simulation step."""
        # Calculate deltas
        current_collected = sum(r.packages_collected for r in robots)
        collected_this_step = current_collected - self.prev_packages_collected
        conflicts_this_step = conflicts_total - self.prev_conflicts
        
        step_data = StepData(
            timestep=timestep,
            robot_positions=[r.position for r in robots],
            robot_statuses=[r.status for r in robots],
            robot_targets=[r.target_package for r in robots],
            packages_remaining=len(warehouse.packages),
            packages_collected_this_step=collected_this_step,
            conflicts_this_step=conflicts_this_step,
            replans_this_step=replans
        )
        
        self.trace.add_step(step_data)
        
        self.prev_packages_collected = current_collected
        self.prev_conflicts = conflicts_total
    
    def get_trace(self) -> SimulationTrace:
        """Get the collected simulation trace."""
        return self.trace
    
    def reset(self):
        """Reset the collector for a new simulation."""
        self.trace = SimulationTrace()
        self.prev_packages_collected = 0
        self.prev_conflicts = 0

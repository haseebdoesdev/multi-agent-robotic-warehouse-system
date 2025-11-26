"""
Matplotlib reporting module (Module 5)
--------------------------------------
Create visual performance reports and save to disk.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

# Import metrics types
from .metrics import SimulationTrace

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def generate_report(final_stats: Dict, 
                    sim_trace: SimulationTrace,
                    out_dir: str = "reports",
                    show_plots: bool = False) -> Dict:
    """
    Generate visual performance reports and save to disk.
    
    Args:
        final_stats: Final computed metrics from simulation
        sim_trace: Per-step simulation trace data
        out_dir: Directory to save report files
        show_plots: Whether to display plots interactively
    
    Returns:
        Dictionary with paths to generated files
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping visual reports.")
        return {'error': 'matplotlib not installed'}
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_files = {}
    
    # Generate individual plots
    try:
        # 1. Packages collected over time
        path1 = generate_collection_timeline(sim_trace, out_dir, timestamp)
        if path1:
            generated_files['collection_timeline'] = path1
        
        # 2. Robot distance comparison
        path2 = generate_robot_distance_chart(final_stats, out_dir, timestamp)
        if path2:
            generated_files['robot_distances'] = path2
        
        # 3. Conflicts over time
        path3 = generate_conflict_chart(sim_trace, out_dir, timestamp)
        if path3:
            generated_files['conflicts'] = path3
        
        # 4. Robot utilization
        path4 = generate_utilization_chart(final_stats, out_dir, timestamp)
        if path4:
            generated_files['utilization'] = path4
        
        # 5. Combined summary dashboard
        path5 = generate_summary_dashboard(final_stats, sim_trace, out_dir, timestamp)
        if path5:
            generated_files['dashboard'] = path5
        
        # 6. Save text summary
        path6 = save_text_summary(final_stats, out_dir, timestamp)
        if path6:
            generated_files['text_summary'] = path6
        
        if show_plots:
            plt.show()
        
    except Exception as e:
        print(f"Error generating reports: {e}")
        generated_files['error'] = str(e)
    
    return generated_files


def generate_collection_timeline(sim_trace: SimulationTrace,
                                  out_dir: str,
                                  timestamp: str) -> Optional[str]:
    """Generate plot of packages collected over time."""
    if not sim_trace.steps:
        return None
    
    timesteps = sim_trace.get_timesteps()
    cumulative = sim_trace.get_cumulative_collected()
    remaining = sim_trace.get_packages_remaining()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis: cumulative collected
    color1 = '#2ecc71'
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Packages Collected (Cumulative)', color=color1, fontsize=12)
    ax1.plot(timesteps, cumulative, color=color1, linewidth=2, label='Collected')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(timesteps, cumulative, alpha=0.3, color=color1)
    
    # Secondary axis: remaining packages
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Packages Remaining', color=color2, fontsize=12)
    ax2.plot(timesteps, remaining, color=color2, linewidth=2, linestyle='--', label='Remaining')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Package Collection Progress', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(out_dir, f'collection_timeline_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_robot_distance_chart(final_stats: Dict,
                                   out_dir: str,
                                   timestamp: str) -> Optional[str]:
    """Generate bar chart comparing robot distances and packages."""
    robot_metrics = final_stats.get('robot_metrics', [])
    if not robot_metrics:
        # Try to get from raw stats
        robots = final_stats.get('robots', [])
        if not robots:
            return None
        robot_metrics = [{'id': r['id'], 
                          'distance_traveled': r.get('distance_traveled', 0),
                          'packages_collected': r.get('packages_collected', 0)}
                         for r in robots]
    
    robot_ids = [f"Robot {r['id']}" for r in robot_metrics]
    distances = [r.get('distance_traveled', 0) for r in robot_metrics]
    packages = [r.get('packages_collected', 0) for r in robot_metrics]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = range(len(robot_ids))
    width = 0.35
    
    # Distance bars
    bars1 = ax1.bar([i - width/2 for i in x], distances, width, 
                     label='Distance Traveled', color='#3498db', alpha=0.8)
    ax1.set_ylabel('Distance Traveled', fontsize=12)
    ax1.set_xlabel('Robot', fontsize=12)
    
    # Packages bars on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], packages, width,
                     label='Packages Collected', color='#2ecc71', alpha=0.8)
    ax2.set_ylabel('Packages Collected', fontsize=12)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(robot_ids)
    
    # Add value labels on bars
    for bar, val in zip(bars1, distances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    for bar, val in zip(bars2, packages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Robot Performance Comparison', fontsize=14, fontweight='bold')
    
    # Combined legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    filepath = os.path.join(out_dir, f'robot_distances_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_conflict_chart(sim_trace: SimulationTrace,
                            out_dir: str,
                            timestamp: str) -> Optional[str]:
    """Generate chart showing conflicts over time."""
    if not sim_trace.steps:
        return None
    
    timesteps = sim_trace.get_timesteps()
    conflicts_per_step = sim_trace.get_conflicts_per_step()
    cumulative_conflicts = sim_trace.get_cumulative_conflicts()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top plot: conflicts per step
    ax1.bar(timesteps, conflicts_per_step, color='#e74c3c', alpha=0.7, width=1)
    ax1.set_ylabel('Conflicts per Step', fontsize=12)
    ax1.set_title('Conflict Events Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: cumulative conflicts
    ax2.plot(timesteps, cumulative_conflicts, color='#c0392b', linewidth=2)
    ax2.fill_between(timesteps, cumulative_conflicts, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Cumulative Conflicts', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(out_dir, f'conflicts_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_utilization_chart(final_stats: Dict,
                                out_dir: str,
                                timestamp: str) -> Optional[str]:
    """Generate pie chart of robot utilization."""
    utilization = final_stats.get('utilization', {})
    status_breakdown = utilization.get('status_breakdown', {})
    
    if not status_breakdown:
        # Try to compute from basic stats
        return None
    
    # Filter out zero values and prepare data
    labels = []
    sizes = []
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    status_names = {
        'moving': 'Moving',
        'idle': 'Idle',
        'waiting': 'Waiting',
        'collecting': 'Collecting',
        'completed': 'Completed'
    }
    
    for status, pct in status_breakdown.items():
        if pct > 0.5:  # Only show significant portions
            labels.append(status_names.get(status, status.title()))
            sizes.append(pct)
    
    if not sizes:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        colors=colors[:len(sizes)],
        startangle=90,
        explode=[0.02] * len(sizes)
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('Robot Time Utilization', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    filepath = os.path.join(out_dir, f'utilization_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_summary_dashboard(final_stats: Dict,
                                sim_trace: SimulationTrace,
                                out_dir: str,
                                timestamp: str) -> Optional[str]:
    """Generate a combined summary dashboard with multiple plots."""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Collection timeline (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if sim_trace.steps:
        timesteps = sim_trace.get_timesteps()
        cumulative = sim_trace.get_cumulative_collected()
        ax1.plot(timesteps, cumulative, color='#2ecc71', linewidth=2)
        ax1.fill_between(timesteps, cumulative, alpha=0.3, color='#2ecc71')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Packages Collected')
    ax1.set_title('Collection Progress', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Robot performance (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    robot_metrics = final_stats.get('robot_metrics', [])
    if not robot_metrics:
        robots = final_stats.get('robots', [])
        robot_metrics = [{'id': r['id'], 'packages_collected': r.get('packages_collected', 0)}
                         for r in robots]
    
    if robot_metrics:
        robot_ids = [f"R{r['id']}" for r in robot_metrics]
        packages = [r.get('packages_collected', 0) for r in robot_metrics]
        colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
        ax2.bar(robot_ids, packages, color=colors[:len(robot_ids)], alpha=0.8)
        for i, v in enumerate(packages):
            ax2.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    ax2.set_ylabel('Packages Collected')
    ax2.set_title('Robot Contributions', fontweight='bold')
    
    # 3. Key metrics (bottom-left) - Text panel
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    total_timesteps = final_stats.get('total_timesteps', final_stats.get('timesteps', 0))
    total_packages = final_stats.get('total_packages_collected', 0)
    total_distance = final_stats.get('total_distance_traveled', 0)
    total_conflicts = final_stats.get('total_conflicts', final_stats.get('conflicts_resolved', 0))
    
    efficiency = final_stats.get('efficiency', {})
    moves_per_pkg = efficiency.get('moves_per_package', 
                                    total_distance / total_packages if total_packages > 0 else 0)
    
    metrics_text = f"""
    KEY METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Total Timesteps:     {total_timesteps}
    Packages Collected:  {total_packages}
    Total Distance:      {total_distance}
    Conflicts Resolved:  {total_conflicts}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Efficiency:          {moves_per_pkg:.2f} moves/pkg
    Robots:              {len(robot_metrics)}
    """
    
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    # 4. Conflicts over time (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    if sim_trace.steps:
        timesteps = sim_trace.get_timesteps()
        cumulative_conflicts = sim_trace.get_cumulative_conflicts()
        ax4.plot(timesteps, cumulative_conflicts, color='#e74c3c', linewidth=2)
        ax4.fill_between(timesteps, cumulative_conflicts, alpha=0.3, color='#e74c3c')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Cumulative Conflicts')
    ax4.set_title('Conflict Resolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Warehouse Simulation Summary', fontsize=16, fontweight='bold', y=0.98)
    
    filepath = os.path.join(out_dir, f'dashboard_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def save_text_summary(final_stats: Dict, 
                      out_dir: str,
                      timestamp: str) -> Optional[str]:
    """Save a text summary of the simulation results."""
    from .metrics import format_metrics_summary
    
    summary = format_metrics_summary(final_stats)
    
    filepath = os.path.join(out_dir, f'summary_{timestamp}.txt')
    with open(filepath, 'w') as f:
        f.write(summary)
        f.write(f"\n\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return filepath


def create_animation_frames(sim_trace: SimulationTrace,
                            warehouse_grid: list,
                            out_dir: str,
                            timestamp: str,
                            max_frames: int = 100) -> Optional[str]:
    """
    Generate animation frames for the simulation (optional advanced feature).
    Note: This is computationally expensive and optional.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # This is a placeholder for future animation support
    # Would require additional dependencies like matplotlib.animation
    pass

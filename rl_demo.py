"""
RL Training Demo - Visual Q-Learning Training Interface
--------------------------------------------------------
Run with: python rl_demo.py

This provides a visual interface for:
- Watching Q-Learning training in real-time
- Testing trained agents
- Visualizing Q-values and learning progress
- Saving and loading trained models

CONTROLS:
  SPACE   - Start/Stop training
  T       - Test the trained agent
  R       - Reset agent (clear Q-table)
  +/-     - Adjust visualization speed
  ESC     - Exit

BUTTONS:
  Train     - Start/stop training
  Pause     - Pause training/testing
  Test      - Run test episodes with trained agent
  Reset     - Clear Q-table and restart
  Save      - Save trained agent to file
  Load      - Load previously trained agent
  New Grid  - Generate new random warehouse
  Speed +/- - Adjust how fast training is visualized
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="RL Training Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=10,
        help="Grid size (default: 10)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=100000,
        help="Number of training episodes (default: 100000)"
    )
    
    parser.add_argument(
        "--obstacle-density", "-o",
        type=float,
        default=0.15,
        help="Obstacle density 0.0-1.0 (default: 0.15)"
    )
    
    parser.add_argument(
        "--window-width", "-W",
        type=int,
        default=900,
        help="Window width in pixels (default: 900)"
    )
    
    parser.add_argument(
        "--window-height", "-H",
        type=int,
        default=700,
        help="Window height in pixels (default: 700)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid pygame initialization at module load
    from warehouse import Warehouse
    from ui.rl_gui import RLTrainingUI
    
    print("=" * 60)
    print("  Q-Learning Training Visualizer")
    print("=" * 60)
    print(f"\n  Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Obstacle Density: {args.obstacle_density * 100:.0f}%")
    print()
    
    # Create warehouse
    warehouse = Warehouse(
        width=args.grid_size,
        height=args.grid_size,
        obstacle_density=args.obstacle_density
    )
    warehouse.place_obstacles()
    
    # Create and run UI
    try:
        ui = RLTrainingUI(
            warehouse=warehouse,
            window_width=args.window_width,
            window_height=args.window_height
        )
        ui.total_episodes = args.episodes
        ui.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




"""
main.py
-------
Pipeline orchestrator: TSP global planner -> SLSQP local planner -> summary.

Usage:
  python main.py
  python main.py --clearance 5.0 --n_waypoints 12 --smoothness 0.5
"""

import argparse
import time
from pathlib import Path

from global_planner import run_global_planner
from local_planner import run_local_planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone trajectory optimisation pipeline"
    )
    parser.add_argument(
        "--csv",
        default="sphere_positions.csv",
        help="Path to tree poses CSV (default: sphere_positions.csv)",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=5.0,
        help="Vertical clearance above tree tops in metres (default: 5.0)",
    )
    parser.add_argument(
        "--n_waypoints",
        type=int,
        default=8,
        help="Interior waypoints per segment (default: 12)",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.5,
        help="Smoothness penalty weight (default: 0.5)",
    )
    parser.add_argument(
        "--figures_dir", default="report/figures", help="Directory for output figures"
    )
    parser.add_argument(
        "--processed_dir", default="data/processed", help="Directory for processed data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("=" * 60)
    print("  DRONE TRAJECTORY OPTIMISATION — TC6039.1")
    print("=" * 60)

    # --- Stage 1: Global Planner (TSP) ---
    t1 = time.perf_counter()
    global_result = run_global_planner(
        csv_path=args.csv,
        figures_dir=args.figures_dir,
        processed_dir=args.processed_dir,
    )
    t1 = time.perf_counter() - t1

    # --- Stage 2: Local Planner (SLSQP) ---
    t2 = time.perf_counter()
    local_result = run_local_planner(
        ordered_wp_path=f"{args.processed_dir}/ordered_waypoints.csv",
        raw_poses_path=args.csv,
        figures_dir=args.figures_dir,
        processed_dir=args.processed_dir,
        n_waypoints=args.n_waypoints,
        clearance=args.clearance,
        poses=global_result["poses"],
        tour=global_result["tour"],
        smoothness_weight=args.smoothness,
    )
    t2 = time.perf_counter() - t2
    total_time = time.perf_counter() - t0

    # --- Summary ---
    nn = global_result["nn_length"]
    opt = global_result["opt_length"]
    improvement = 100 * (nn - opt) / nn

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Trees inspected          : {len(global_result['tour'])}")
    print(f"  NN initial tour length   : {nn:.4f} m")
    print(f"  2-opt optimised length   : {opt:.4f} m")
    print(f"  2-opt improvement        : {improvement:.1f} %")
    print(f"  Local planner segments   : {len(local_result['segment_lengths'])}")
    print(f"  Total 3-D path length    : {sum(local_result['segment_lengths']):.4f} m")
    print(f"  Clearance enforced       : {args.clearance} m")
    print(f"  Smoothness weight        : {args.smoothness}")
    tk = local_result["takeoff_xyz"]
    print(f"  Takeoff/landing point    : ({tk[0]:.3f}, {tk[1]:.3f}, {tk[2]:.3f})")
    print(f"  Waypoints in full path   : {len(local_result['full_trajectory'])}")
    print(f"  Stage 1 time (global)    : {t1:.2f} s")
    print(f"  Stage 2 time (local)     : {t2:.2f} s")
    print(f"  Total wall time          : {total_time:.2f} s")

    print("\n  Output artefacts:")
    artefacts = [
        f"{args.processed_dir}/ordered_waypoints.csv",
        f"{args.processed_dir}/full_trajectory.csv",
        f"{args.figures_dir}/global_convergence.png",
        f"{args.figures_dir}/global_tour_3d.png",
        f"{args.figures_dir}/local_convergence_all.png",
        f"{args.figures_dir}/local_trajectory_3d.png",
        f"{args.figures_dir}/local_altitude_profile.png",
    ]
    for path in artefacts:
        mark = "✓" if Path(path).exists() else "✗"
        print(f"    {mark}  {path}")

    print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()

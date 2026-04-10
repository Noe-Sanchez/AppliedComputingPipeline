"""
optimizer.py
------------
Entry point for the trajectory optimisation component.

Command-line usage:
    python optimizer.py              # run the full pipeline
    python optimizer.py test         # run pytest on optimizer/tests/
    python optimizer.py regen        # regenerate sphere_positions.csv from gs.blend
    python optimizer.py --help       # show pipeline CLI args
    python optimizer.py --clearance 5.0 --n_waypoints 12

Programmatic usage (from a parent pipeline):
    from optimizer import run
    result = run(clearance=4.2, n_waypoints=12)
    traj = result["full_trajectory"]
    takeoff = result["takeoff_xyz"]

Environment:
    BLENDER_BIN   Path to the Blender binary (default: 'blender' on $PATH).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Make `optimizer/` importable so `main`, `global_planner`, `local_planner` resolve.
PKG_DIR = Path(__file__).resolve().parent / "optimizer"
sys.path.insert(0, str(PKG_DIR))


def run(
    clearance: float = 5.0,
    n_waypoints: int = 8,
    smoothness: float = 0.5,
    csv: str = "sphere_positions.csv",
    figures_dir: str = "report/figures",
    processed_dir: str = "data/processed",
) -> dict:
    """Run the full optimisation pipeline programmatically.

    Intended for use from a parent pipeline script that wants to pass
    parameters (especially clearance, derived from the regression module)
    directly rather than via the command line.

    Returns a dict with keys:
        tour            : list[int]        — TSP visit order (node indices)
        nn_length       : float            — nearest-neighbour tour length (m)
        opt_length      : float            — 2-opt tour length (m)
        full_trajectory : np.ndarray (M,3) — optimised 3-D path
        segment_lengths : list[float]      — per-segment lengths (m)
        takeoff_xyz     : np.ndarray (3,)  — chosen takeoff/landing point
    """
    # Work from inside optimizer/ so relative paths resolve correctly.
    prev_cwd = os.getcwd()
    os.chdir(PKG_DIR)
    try:
        from global_planner import run_global_planner
        from local_planner import run_local_planner

        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        Path(processed_dir).mkdir(parents=True, exist_ok=True)

        global_result = run_global_planner(
            csv_path=csv,
            figures_dir=figures_dir,
            processed_dir=processed_dir,
        )
        local_result = run_local_planner(
            ordered_wp_path=f"{processed_dir}/ordered_waypoints.csv",
            raw_poses_path=csv,
            figures_dir=figures_dir,
            processed_dir=processed_dir,
            n_waypoints=n_waypoints,
            clearance=clearance,
            poses=global_result["poses"],
            tour=global_result["tour"],
            smoothness_weight=smoothness,
        )

        return {
            "opt_tour": global_result["tour"],
            "global_poses": global_result["poses"],
            "names": global_result["names"],
            "dist_matrix": global_result["dist_matrix"],
            "nn_length": global_result["nn_length"],
            "opt_length": global_result["opt_length"],
            "history": global_result["history"],

            "full_trajectory": local_result["full_trajectory"],
            "history_labels": local_result["history_labels"],
            "trees": local_result["trees"],
            "waypoints": local_result["waypoints"],
            "local_poses": local_result["poses"],
            "tour": local_result["tour"],
            "figures_dir": local_result["figures_dir"],
            "clearance": local_result["clearance"],
            "all_histories": local_result["all_histories"],
            "segment_lengths": local_result["segment_lengths"],
            "takeoff_xyz": local_result["takeoff_xyz"],
        }
    finally:
        os.chdir(prev_cwd)


def run_tests() -> int:
    """Run pytest on optimizer/tests/ and return its exit code."""
    tests_dir = PKG_DIR / "tests"
    return subprocess.call(
        [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
        cwd=str(PKG_DIR),
    )


def run_regen() -> int:
    """Regenerate sphere_positions.csv by running update_tree_poses.py inside Blender."""
    blend_file = PKG_DIR / "gs.blend"
    script_file = PKG_DIR / "update_tree_poses.py"
    if not blend_file.exists():
        print(f"ERROR: Blender file not found: {blend_file}")
        return 1
    if not script_file.exists():
        print(f"ERROR: Blender export script not found: {script_file}")
        return 1
    blender = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender:
        print(
            "ERROR: Blender not found. Install it on your PATH or set BLENDER_BIN,\n"
            "  e.g.  BLENDER_BIN=~/Downloads/blender-4.5.2-linux-x64/blender "
            "python optimizer.py regen"
        )
        return 1
    print(f"[regen] Blender : {blender}")
    print(f"[regen] Scene   : {blend_file}")
    print(f"[regen] Script  : {script_file}")
    rc = subprocess.call(
        [blender, "--background", str(blend_file), "--python", str(script_file)],
        cwd=str(PKG_DIR),  # so the script writes sphere_positions.csv here
    )
    if rc == 0:
        out_csv = PKG_DIR / "sphere_positions.csv"
        print(f"[regen] Wrote {out_csv}")
    else:
        print(f"[regen] Blender exited with code {rc}")
    return rc


def run_pipeline_cli() -> None:
    """Run the pipeline via the argparse CLI in optimizer/main.py."""
    os.chdir(PKG_DIR)
    from main import main

    main()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd in ("test", "tests"):
            sys.exit(run_tests())
        if cmd == "regen":
            sys.exit(run_regen())
    run_pipeline_cli()

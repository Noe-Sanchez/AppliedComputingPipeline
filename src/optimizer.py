"""
optimizer.py
------------
Entry point for the trajectory optimisation component.

Usage:
    python optimizer.py              # run the full pipeline
    python optimizer.py test         # run pytest on optimizer/tests/
    python optimizer.py regen        # regenerate sphere_positions.csv from gs.blend
    python optimizer.py --help       # show pipeline CLI args
    python optimizer.py --clearance 5.0 --n_waypoints 12

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


def run_pipeline() -> None:
    """Run the full trajectory optimisation pipeline."""
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
    run_pipeline()

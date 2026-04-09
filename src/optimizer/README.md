# Drone Trajectory Optimization

Component **C3 — Optimization** of the TC6039.1 Applied Computing final
project. Given a set of tree positions in an avocado orchard, this module
plans a constrained 3-D inspection trajectory that:

1. Visits every tree once in an optimal order (global TSP).
2. Stays above each tree canopy with a configurable vertical clearance,
   with a smooth path between inspection points (local SLSQP).
3. Departs from and returns to the same takeoff/landing point, whose
   `(x, y)` location is chosen by the optimiser.

## Pipeline overview

```
tree poses CSV
      │
      ▼
┌──────────────────┐      ordered waypoints      ┌──────────────────┐
│ global_planner   │ ───────────────────────────▶│  local_planner   │
│   TSP: NN + 2-opt│                             │  SLSQP (scipy)   │
└──────────────────┘                             └──────────────────┘
                                                          │
                                                          ▼
                                          full_trajectory.csv + figures
```

- **Global planner** (`global_planner.py`) builds a distance matrix,
  initialises with nearest-neighbour, and improves with 2-opt local search.
- **Local planner** (`local_planner.py`) optimises each segment with
  `scipy.optimize.minimize(method='SLSQP')`. The takeoff and landing legs
  are solved **jointly** over a shared `(x, y)` decision variable so the
  base station is placed optimally for the whole round trip.

## Repository layout

```
optimizer.py                    ← entry point (run from here)
optimizer/
├── main.py                     ← pipeline orchestrator
├── global_planner.py           ← C3a: TSP solver
├── local_planner.py            ← C3b: SLSQP trajectory optimiser
├── conftest.py                 ← pytest path setup
├── sphere_positions.csv        ← input: tree poses
├── gs.blend                    ← Blender scene (source of truth)
├── get_tree_poses.py           ← Blender export script
├── tests/
│   ├── test_global_planner.py  ← 5 tests
│   └── test_local_planner.py   ← 6 tests
├── data/processed/             ← generated CSVs
└── report/figures/             ← generated plots
```

## Requirements

- Python ≥ 3.9
- `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest`
- Optional (only for updating [`regen`] the tree positions): [Blender](https://www.blender.org/) ≥ 4.5, [3dgs-render-blender-addon](https://github.com/Kiri-Innovation/3dgs-render-blender-addon) release 4.1.5

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib pytest
```

## Running

All commands are run from the `src` directory, containing `optimizer.py`.

### Run the full pipeline

```bash
python3 optimizer.py
```

This reads `optimizer/sphere_positions.csv`, runs the TSP and SLSQP
stages, writes processed CSVs to `optimizer/data/processed/`, saves
figures to `optimizer/report/figures/`, and prints a summary.


### Custom parameters

```bash
python3 optimizer.py --clearance 5.0 --n_waypoints 12 --smoothness 0.5
```

| Flag | Default | What it controls |
|------|---------|------------------|
| `--clearance` | `5.0` | Vertical clearance above each tree top, in metres. |
| `--n_waypoints` | `12` | Interior waypoints per segment (higher = smoother, slower). |
| `--smoothness` | `0.5` | Weight on the curvature penalty in the SLSQP objective. |
| `--csv` | `sphere_positions.csv` | Input CSV with tree poses. |
| `--figures_dir` | `report/figures` | Output directory for plots. |
| `--processed_dir` | `data/processed` | Output directory for generated CSVs. |

Run `python3 optimizer.py --help` for the full list.

### Run the unit tests

```bash
python3 optimizer.py test
```

Runs `pytest` on all 11 tests (5 for the global planner, 6 for the local
planner). Expected output: **11 passed**. Tests use synthetic tree data
and do not require the real CSV.

### Regenerate tree poses from Blender

```bash
python3 optimizer.py regen
```

Runs `get_tree_poses.py` inside Blender in background mode, rewriting
`sphere_positions.csv` from the current state of `gs.blend`. Useful after
editing the orchard scene.

The wrapper looks for Blender in this order:

1. `$BLENDER_BIN` environment variable (if set)
2. `blender` on the system `$PATH`

If Blender is installed in a non-standard location, for example:

```bash
BLENDER_BIN=~/Downloads/blender-4.5.2-linux-x64/blender python3 optimizer.py regen
```

To make this permanent, add that `export` line to your `~/.bashrc`.

## Input format

`sphere_positions.csv` must contain the columns `name, x, y, z` (header
row required). Coordinates are in metres, with `z` being the tree top
height above ground.

```csv
name,x,y,z
Sphere,-0.23,4.12,8.45
Sphere.001,-5.87,-2.11,10.32
...
```

## Outputs

After a successful run:

**`data/processed/`**
- `ordered_waypoints.csv` — tree visit order after 2-opt
- `full_trajectory.csv` — dense 3-D path, one row per waypoint

**`report/figures/`**
- `global_convergence.png` — 2-opt tour length over improving swaps
- `global_tour_3d.png` — TSP tour plotted over cone-shaped trees
- `local_convergence_all.png` — SLSQP objective history per segment
- `local_trajectory_3d.png` — final optimised path with trees
- `local_altitude_profile.png` — drone altitude vs. clearance band

"""
local_planner.py
----------------
Constrained 3-D trajectory optimisation for the drone inspection problem.

Given a TSP visit order from the global planner, builds a smooth path that:
  - Visits every tree's inspection point.
  - Stays at z >= tree_top + clearance whenever it is within MAX_RADIUS_M
    horizontally of any tree.
  - Departs from and returns to the same takeoff/landing point, whose
    (x, y) is chosen by the optimiser.

Solver: scipy.optimize.minimize with method='SLSQP'.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from pathlib import Path


# ---------------------------------------------------------------------------
# Default parameters (used only when functions are called without explicit
# values; the real pipeline values come from main.py's CLI args).
# ---------------------------------------------------------------------------

CLEARANCE_M: float = 5.0  # vertical clearance above tree tops (m)
MAX_RADIUS_M: float = 2.0  # tree crown radius for the cylinder constraint (m)
N_WAYPOINTS: int = 8  # interior waypoints per segment
SMOOTHNESS_W: float = 0.5  # default smoothness penalty weight


# ---------------------------------------------------------------------------
# Constraint helper
# ---------------------------------------------------------------------------


def _altitude_constraints(
    interior_flat: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    trees: np.ndarray,
    clearance: float,
    samples_per_segment: int = 5,
) -> np.ndarray:
    """Sampled altitude clearance constraints along a piecewise-linear path.

    Returns a 1-D array; every entry must be >= 0 for the path to be feasible.
    Sampling along edges (not just at waypoints) prevents the drone from
    cutting through a tree cylinder between two safe waypoints.
    """
    waypoints = np.vstack([start, interior_flat.reshape(-1, 3), end])
    n_wp = len(waypoints)

    # Sample points along each edge.
    sample_pts = []
    for i in range(n_wp - 1):
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            sample_pts.append(waypoints[i] * (1 - t) + waypoints[i + 1] * t)
    sample_pts.append(waypoints[-1])
    pts = np.array(sample_pts)

    dx = pts[:, 0:1] - trees[:, 0]
    dy = pts[:, 1:2] - trees[:, 1]
    horiz_dist = np.sqrt(dx**2 + dy**2)

    margin = pts[:, 2:3] - (trees[:, 2] + clearance)

    # Smoothstep blend across the cylinder boundary so SLSQP gets a
    # differentiable Jacobian instead of a step discontinuity.
    band = 0.5
    s = np.clip((horiz_dist - MAX_RADIUS_M) / band, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    margin = margin * (1.0 - blend) + 1.0 * blend

    return margin.flatten()


# ---------------------------------------------------------------------------
# Single-segment optimisation (used for tree-i -> tree-i+1 edges)
# ---------------------------------------------------------------------------


def optimise_segment(
    start: np.ndarray,
    end: np.ndarray,
    trees: np.ndarray,
    n_waypoints: int = N_WAYPOINTS,
    clearance: float = CLEARANCE_M,
    smoothness_weight: float = SMOOTHNESS_W,
    max_iter: int = 500,
) -> tuple[np.ndarray, OptimizeResult, list[float]]:
    """Optimise a single segment between two fixed endpoints.

    Returns the (n_waypoints + 2, 3) waypoint array, the SLSQP result, and
    the per-evaluation objective history (for convergence plots).
    """
    assert start.shape == (3,) and end.shape == (3,), "start/end must be (3,)."
    assert trees.ndim == 2 and trees.shape[1] == 3, "trees must be (n, 3)."

    # Initial guess: straight line, lifted to a globally safe cruise altitude
    # so SLSQP starts feasible regardless of clearance value.
    t = np.linspace(0, 1, n_waypoints + 2)[1:-1]
    x0 = start + t[:, None] * (end - start)
    safe_z = float(trees[:, 2].max()) + clearance
    x0[:, 2] = np.maximum(x0[:, 2], safe_z)
    x0_flat = x0.flatten()

    obj_history: list[float] = []

    def objective(flat: np.ndarray) -> float:
        pts = np.vstack([start, flat.reshape(-1, 3), end])
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        accel = pts[2:] - 2 * pts[1:-1] + pts[:-2]
        smoothness = float((accel**2).sum())
        val = length + smoothness_weight * smoothness
        obj_history.append(val)
        return val

    def constraint_fn(flat: np.ndarray) -> np.ndarray:
        return _altitude_constraints(flat, start, end, trees, clearance)

    result = minimize(
        objective,
        x0_flat,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraint_fn}],
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )

    waypoints = np.vstack([start, result.x.reshape(-1, 3), end])

    if constraint_fn(result.x).min() < -1e-4:
        print(
            f"  [local_planner] WARNING: segment constraint violated by "
            f"{abs(constraint_fn(result.x).min()):.5f} m."
        )

    return waypoints, result, obj_history


# ---------------------------------------------------------------------------
# Joint round-trip optimisation: takeoff = landing, both legs solved together
# ---------------------------------------------------------------------------


def optimise_round_trip(
    first_inspection: np.ndarray,
    last_inspection: np.ndarray,
    trees: np.ndarray,
    start_xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    n_waypoints: int = N_WAYPOINTS,
    clearance: float = CLEARANCE_M,
    smoothness_weight: float = SMOOTHNESS_W,
    max_iter: int = 500,
    ground_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult, list[float]]:
    """Jointly optimise takeoff -> first_inspection AND last_inspection -> landing.

    The takeoff/landing point (x, y, ground_z) is a single shared decision
    variable: it appears as the start of the outbound leg and as the end of
    the return leg, so SLSQP picks the (x, y) that minimises the sum of
    both legs' lengths under clearance constraints.

    `start_xy_bounds = ((x_min, x_max), (y_min, y_max))` keeps the takeoff
    inside a sane box (otherwise SLSQP can drift to infinity).

    Returns: outbound_wps, return_wps, takeoff_xyz, SLSQP result, history.
    """
    assert first_inspection.shape == (3,) and last_inspection.shape == (3,)
    assert trees.ndim == 2 and trees.shape[1] == 3

    # Seed takeoff at the midpoint of the two endpoints.
    tx0 = 0.5 * (first_inspection[0] + last_inspection[0])
    ty0 = 0.5 * (first_inspection[1] + last_inspection[1])
    seed_start = np.array([tx0, ty0, ground_z])

    # Both legs start with a straight-line guess lifted to safe cruise.
    safe_z = float(trees[:, 2].max()) + clearance
    t = np.linspace(0, 1, n_waypoints + 2)[1:-1]

    out_init = seed_start + t[:, None] * (first_inspection - seed_start)
    out_init[:, 2] = np.maximum(out_init[:, 2], safe_z)

    ret_init = last_inspection + t[:, None] * (seed_start - last_inspection)
    ret_init[:, 2] = np.maximum(ret_init[:, 2], safe_z)

    x0_flat = np.concatenate([[tx0, ty0], out_init.flatten(), ret_init.flatten()])

    # Bounds: only the takeoff (x, y) are constrained; interiors are free.
    bounds = (
        [start_xy_bounds[0], start_xy_bounds[1]]
        + [(None, None)] * (3 * n_waypoints)  # outbound interior
        + [(None, None)] * (3 * n_waypoints)  # return interior
    )

    n_int = n_waypoints

    def unpack(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tk = np.array([flat[0], flat[1], ground_z])
        out_int = flat[2 : 2 + 3 * n_int].reshape(-1, 3)
        ret_int = flat[2 + 3 * n_int :].reshape(-1, 3)
        return tk, out_int, ret_int

    obj_history: list[float] = []

    def objective(flat: np.ndarray) -> float:
        tk, out_int, ret_int = unpack(flat)
        out_pts = np.vstack([tk, out_int, first_inspection])
        ret_pts = np.vstack([last_inspection, ret_int, tk])

        length = float(
            np.linalg.norm(np.diff(out_pts, axis=0), axis=1).sum()
            + np.linalg.norm(np.diff(ret_pts, axis=0), axis=1).sum()
        )
        out_acc = out_pts[2:] - 2 * out_pts[1:-1] + out_pts[:-2]
        ret_acc = ret_pts[2:] - 2 * ret_pts[1:-1] + ret_pts[:-2]
        smoothness = float((out_acc**2).sum() + (ret_acc**2).sum())

        val = length + smoothness_weight * smoothness
        obj_history.append(val)
        return val

    def constraint_fn(flat: np.ndarray) -> np.ndarray:
        tk, out_int, ret_int = unpack(flat)
        out_c = _altitude_constraints(
            out_int.flatten(), tk, first_inspection, trees, clearance
        )
        ret_c = _altitude_constraints(
            ret_int.flatten(), last_inspection, tk, trees, clearance
        )
        return np.concatenate([out_c, ret_c])

    result = minimize(
        objective,
        x0_flat,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint_fn}],
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )

    tk, out_int, ret_int = unpack(result.x)
    outbound_wps = np.vstack([tk, out_int, first_inspection])
    return_wps = np.vstack([last_inspection, ret_int, tk])

    if constraint_fn(result.x).min() < -1e-4:
        print(
            f"  [local_planner] WARNING: round-trip constraint violated by "
            f"{abs(constraint_fn(result.x).min()):.5f} m."
        )

    return outbound_wps, return_wps, tk, result, obj_history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_local_planner(
    ordered_wp_path: str = "data/processed/ordered_waypoints.csv",
    raw_poses_path: str = "data/raw/tree_poses.csv",
    figures_dir: str = "report/figures",
    processed_dir: str = "data/processed",
    n_waypoints: int = N_WAYPOINTS,
    clearance: float = CLEARANCE_M,
    poses: np.ndarray = None,
    tour: list = None,
    smoothness_weight: float = SMOOTHNESS_W,
) -> dict:
    """Optimise the full inspection trajectory and write outputs.

    Returns a dict with keys: full_trajectory, all_histories, segment_lengths,
    takeoff_xyz.
    """
    print("\n[local_planner] Loading waypoints …")
    wp_df = pd.read_csv(ordered_wp_path)
    all_trees_df = pd.read_csv(raw_poses_path)

    assert not wp_df.empty, "Ordered waypoints CSV is empty."
    assert {"x", "y", "z"}.issubset(wp_df.columns), "CSV must contain x, y, z."

    trees = all_trees_df[["x", "y", "z"]].to_numpy(dtype=float)
    waypoints = wp_df[["x", "y", "z"]].to_numpy(dtype=float)
    waypoints[:, 2] += clearance  # lift to inspection altitude
    names = wp_df["name"].tolist()

    print(f"  X range: {trees[:, 0].min():.3f} to {trees[:, 0].max():.3f}")
    print(f"  Y range: {trees[:, 1].min():.3f} to {trees[:, 1].max():.3f}")
    print(f"  Z range: {trees[:, 2].min():.3f} to {trees[:, 2].max():.3f}")

    # Box constraint on the takeoff/landing (x, y): tree bbox + small margin.
    box_margin = 5.0
    takeoff_bounds = (
        (float(trees[:, 0].min()) - box_margin, float(trees[:, 0].max()) + box_margin),
        (float(trees[:, 1].min()) - box_margin, float(trees[:, 1].max()) + box_margin),
    )

    n_inspection = len(waypoints)
    print(
        f"  {n_inspection} inspection waypoints, "
        f"{len(trees)} trees used for clearance constraints."
    )
    print(f"  Clearance = {clearance} m,  interior waypoints/segment = {n_waypoints}")

    all_segment_wps: list[np.ndarray] = []
    all_histories: list[list[float]] = []
    segment_lengths: list[float] = []
    seg_names: list[str] = []

    # --- Joint round-trip: BASE -> first tree AND last tree -> BASE ---
    print(
        f"  Round-trip leg: BASE → {names[0]}  AND  {names[-1]} → BASE  (joint SLSQP)"
    )
    outbound_wps, return_wps, takeoff_xyz, rt_res, rt_history = optimise_round_trip(
        first_inspection=waypoints[0],
        last_inspection=waypoints[-1],
        trees=trees,
        start_xy_bounds=takeoff_bounds,
        n_waypoints=n_waypoints,
        clearance=clearance,
        smoothness_weight=smoothness_weight,
    )
    print(
        f"    base chosen at (x={takeoff_xyz[0]:.3f}, "
        f"y={takeoff_xyz[1]:.3f}, z={takeoff_xyz[2]:.3f})"
    )
    out_len = float(np.linalg.norm(np.diff(outbound_wps, axis=0), axis=1).sum())
    ret_len = float(np.linalg.norm(np.diff(return_wps, axis=0), axis=1).sum())
    rt_status = "OK" if rt_res.success else f"WARN({rt_res.message})"
    print(
        f"    outbound={out_len:.4f} m  return={ret_len:.4f} m  "
        f"iters={rt_res.nit}  {rt_status}"
    )

    all_segment_wps.append(outbound_wps)
    all_histories.append(rt_history)
    segment_lengths.append(out_len)
    seg_names.append(f"BASE→{names[0]}")

    # --- Middle segments: tree i -> tree i+1 ---
    for seg_idx in range(n_inspection - 1):
        start = waypoints[seg_idx]
        end = waypoints[seg_idx + 1]
        print(
            f"  Segment {seg_idx + 2}/{n_inspection + 1}: "
            f"{names[seg_idx]} → {names[seg_idx + 1]}"
        )

        seg_wps, res, history = optimise_segment(
            start,
            end,
            trees,
            n_waypoints=n_waypoints,
            clearance=clearance,
            smoothness_weight=smoothness_weight,
        )

        length = float(np.linalg.norm(np.diff(seg_wps, axis=0), axis=1).sum())
        status = "OK" if res.success else f"WARN({res.message})"
        print(f"    length={length:.4f} m  iters={res.nit}  {status}")

        all_segment_wps.append(seg_wps[1:])  # skip duplicated join point
        all_histories.append(history)
        segment_lengths.append(length)
        seg_names.append(f"{names[seg_idx]}→{names[seg_idx + 1]}")

    # --- Return leg already solved jointly above; just stitch it in ---
    print(
        f"  Return leg: {names[-1]} → BASE  (length={ret_len:.4f} m, from joint solve)"
    )
    all_segment_wps.append(return_wps[1:])
    segment_lengths.append(ret_len)
    seg_names.append(f"{names[-1]}→BASE")

    full_trajectory = np.vstack(all_segment_wps)
    total_length = sum(segment_lengths)
    closing_gap = float(np.linalg.norm(full_trajectory[0] - full_trajectory[-1]))
    print(f"\n  Total optimised trajectory length: {total_length:.4f} m")
    print(f"  Total waypoints in full path     : {len(full_trajectory)}")
    print(f"  Round-trip closing gap           : {closing_gap:.6f} m")

    # Save full trajectory CSV
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    traj_df = pd.DataFrame(full_trajectory, columns=["x", "y", "z"])
    traj_path = f"{processed_dir}/full_trajectory.csv"
    traj_df.to_csv(traj_path, index=False)
    print(f"  Full trajectory saved → {traj_path}")

    # Convergence labels: one per history entry.
    history_labels = [f"BASE↔{names[0]}/{names[-1]} (joint)"] + seg_names[1:-1]

    _plot_all_convergence(all_histories, history_labels, figures_dir)
    # _plot_trajectory_3d(full_trajectory, trees, waypoints, poses, tour, figures_dir)
    _plot_altitude_profile(full_trajectory, trees, clearance, figures_dir)

    return {
        "full_trajectory": full_trajectory,
        "history_labels": history_labels,
        "trees": trees,
        "waypoints": waypoints,
        "poses": poses,
        "tour": tour,
        "figures_dir": figures_dir,
        "clearance": clearance,
        "all_histories": all_histories,
        "segment_lengths": segment_lengths,
        "takeoff_xyz": takeoff_xyz,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _plot_all_convergence(
    histories: list[list[float]],
    seg_labels: list[str],
    figures_dir: str,
) -> None:
    """Plot SLSQP convergence curves for all segments on a single axis."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(histories)))

    for idx, (hist, color) in enumerate(zip(histories, colors)):
        ax.plot(
            hist,
            color=color,
            linewidth=1.2,
            alpha=0.85,
            label=f"Seg {idx + 1}: {seg_labels[idx]}",
        )

    ax.set_xlabel("SLSQP function evaluation #")
    ax.set_ylabel("Objective value (m + smoothness penalty)")
    ax.set_title("SLSQP convergence — local planner (all segments)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = f"{figures_dir}/local_convergence_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [local_planner] convergence plot saved → {out}")


def _plot_altitude_profile(
    trajectory: np.ndarray,
    trees: np.ndarray,
    clearance: float,
    figures_dir: str,
) -> None:
    """Drone altitude vs. path progress, with the clearance band drawn in."""
    path_len = np.cumsum(
        np.concatenate([[0], np.linalg.norm(np.diff(trajectory, axis=0), axis=1)])
    )
    drone_z = trajectory[:, 2]
    max_tree_z = trees[:, 2].max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        path_len, drone_z, color="#185FA5", linewidth=1.6, label="Drone altitude (z)"
    )
    ax.axhline(
        max_tree_z + clearance,
        color="#D85A30",
        linewidth=1.0,
        linestyle="--",
        label=f"Min required z (max tree + {clearance} m)",
    )
    ax.axhline(
        max_tree_z,
        color="#3B6D11",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label="Max tree top",
    )
    ax.fill_between(
        path_len,
        max_tree_z + clearance,
        drone_z,
        where=(drone_z >= max_tree_z + clearance),
        alpha=0.12,
        color="#185FA5",
        label="Clearance margin",
    )
    ax.set_xlabel("Path progress (m)")
    ax.set_ylabel("Altitude Z (m)")
    ax.set_title("Altitude profile — drone vs. tree clearance constraint")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = f"{figures_dir}/local_altitude_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [local_planner] altitude profile saved → {out}")


if __name__ == "__main__":
    result = run_local_planner()
    print(f"\nTotal path length : {sum(result['segment_lengths']):.4f} m")

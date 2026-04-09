"""
global_planner.py
-----------------
Solves the TSP over the tree inspection waypoints to find the visit order
that minimises total 3-D flight distance.

Pipeline: build distance matrix -> nearest-neighbour init -> 2-opt local search.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------


def build_distance_matrix(poses: np.ndarray) -> np.ndarray:
    """Symmetric 3-D Euclidean distance matrix from an (n, 3) array of poses."""
    assert poses.ndim == 2 and poses.shape[1] >= 3, "poses must be (n, >=3)."
    assert poses.shape[0] >= 2, "Need at least 2 poses."

    delta = poses[:, None, :3] - poses[None, :, :3]
    dist_matrix = np.sqrt((delta**2).sum(axis=2))

    assert np.allclose(dist_matrix, dist_matrix.T), "Distance matrix not symmetric."
    return dist_matrix


# ---------------------------------------------------------------------------
# Nearest-neighbour heuristic (initialisation)
# ---------------------------------------------------------------------------


def nearest_neighbour_tour(dist_matrix: np.ndarray, start: int = 0) -> list[int]:
    """Greedy nearest-neighbour TSP tour starting from `start` (no return edge)."""
    n = dist_matrix.shape[0]
    assert 0 <= start < n, f"start={start} out of range [0, {n})."

    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        distances = dist_matrix[tour[-1]].copy()
        distances[visited] = np.inf
        nearest = int(np.argmin(distances))
        tour.append(nearest)
        visited[nearest] = True

    assert len(set(tour)) == n, "Tour contains repeated nodes."
    return tour


# ---------------------------------------------------------------------------
# 2-opt improvement
# ---------------------------------------------------------------------------


def tour_length(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Total Euclidean length of an open tour (no return-to-start edge)."""
    return float(sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def two_opt(
    tour: list[int],
    dist_matrix: np.ndarray,
    max_iterations: int = 5000,
) -> tuple[list[int], list[float]]:
    """Improve a TSP tour with 2-opt local search.

    Returns the improved tour and the convergence history (one entry per
    improving swap).
    """
    n = len(tour)
    assert n == dist_matrix.shape[0], "Tour length must match distance matrix size."

    best_tour = tour[:]
    best_length = tour_length(best_tour, dist_matrix)
    history = [best_length]

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Open tour: the last real edge is (n-2 -> n-1). We must NOT
        # include the closing edge (n-1 -> 0) — there is no such edge.
        # So j ranges only up to n - 2, and for j = n - 2 the second
        # current edge is (n-2 -> n-1).
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[j], best_tour[j + 1]
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                if new_cost < current_cost - 1e-10:
                    best_tour[i : j + 1] = best_tour[i : j + 1][::-1]
                    best_length = best_length - current_cost + new_cost
                    history.append(best_length)
                    improved = True

    return best_tour, history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_convergence(history: list[float], output_path: str) -> None:
    """Plot tour length vs. improving swap index."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#3B6D11", linewidth=1.8)
    ax.scatter([0], [history[0]], color="#E85D24", zorder=5, label="Initial (NN)")
    ax.scatter([len(history) - 1], [history[-1]], color="#3B6D11", zorder=5,
               label=f"Final ({history[-1]:.4f} m)")
    ax.set_xlabel("Improving swap #")
    ax.set_ylabel("Total tour length (m)")
    ax.set_title("2-opt convergence — TSP global planner")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [global_planner] convergence plot saved → {output_path}")


def plot_tour_3d(
    poses: np.ndarray,
    tour: list[int],
    names: list[str],
    output_path: str,
) -> None:
    """3-D visualisation of the optimised TSP tour over cone-shaped trees."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Trees as cones
    n_sides = 15
    r_base = 2.0
    theta = np.linspace(0, 2 * np.pi, n_sides)
    for pos in poses:
        height = pos[2]
        x_base = pos[0] + r_base * np.cos(theta)
        y_base = pos[1] + r_base * np.sin(theta)
        for i in range(n_sides - 1):
            ax.plot_trisurf(
                [x_base[i], x_base[i + 1], pos[0]],
                [y_base[i], y_base[i + 1], pos[1]],
                [0, 0, height],
                color="green", alpha=0.3, linewidth=0,
            )
        ax.plot_trisurf(
            np.append(x_base, pos[0]),
            np.append(y_base, pos[1]),
            np.zeros(n_sides + 1),
            color="darkgreen", alpha=0.6, linewidth=0,
        )

    ordered = poses[tour]
    ax.plot(ordered[:, 0], ordered[:, 1], ordered[:, 2],
            "-o", color="#185FA5", linewidth=1.4, markersize=4, label="Tour")

    ax.text(ordered[0, 0], ordered[0, 1], ordered[0, 2],
            f"  START\n  {names[tour[0]]}", fontsize=7, color="#D85A30")
    ax.text(ordered[-1, 0], ordered[-1, 1], ordered[-1, 2],
            f"  END\n  {names[tour[-1]]}", fontsize=7, color="#3B6D11")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("TSP tour — optimal inspection order")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [global_planner] 3-D tour plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_global_planner(
    csv_path: str = "data/raw/tree_poses.csv",
    figures_dir: str = "report/figures",
    processed_dir: str = "data/processed",
) -> dict:
    """Load tree poses, solve the TSP, save the ordered waypoints, return results."""
    print("\n[global_planner] Loading tree poses …")
    df = pd.read_csv(csv_path)
    assert not df.empty, "CSV file is empty."
    assert {"name", "x", "y", "z"}.issubset(df.columns), \
        "CSV must contain columns: name, x, y, z."

    names = df["name"].tolist()
    poses = df[["x", "y", "z"]].to_numpy(dtype=float)

    assert poses.shape[0] >= 2, "Need at least 2 trees."
    assert np.all(np.isfinite(poses)), "Poses contain NaN or Inf values."

    print(f"  Loaded {len(names)} trees.")

    dist_matrix = build_distance_matrix(poses)
    print(
        f"  Distance matrix: {dist_matrix.shape}  "
        f"max={dist_matrix.max():.4f} m  mean={dist_matrix.mean():.4f} m"
    )

    nn_tour = nearest_neighbour_tour(dist_matrix, start=0)
    nn_length = tour_length(nn_tour, dist_matrix)
    print(f"  Nearest-neighbour tour length : {nn_length:.4f} m")

    opt_tour, history = two_opt(nn_tour, dist_matrix)
    opt_length = tour_length(opt_tour, dist_matrix)
    improvement = 100 * (nn_length - opt_length) / nn_length
    print(
        f"  2-opt tour length             : {opt_length:.4f} m  "
        f"({improvement:.1f}% improvement)"
    )

    # Save ordered waypoints
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    ordered_df = pd.DataFrame({
        "visit_order": range(len(opt_tour)),
        "node_index": opt_tour,
        "name": [names[i] for i in opt_tour],
        "x": [poses[i, 0] for i in opt_tour],
        "y": [poses[i, 1] for i in opt_tour],
        "z": [poses[i, 2] for i in opt_tour],
    })
    wp_path = f"{processed_dir}/ordered_waypoints.csv"
    ordered_df.to_csv(wp_path, index=False)
    print(f"  Ordered waypoints saved → {wp_path}")

    plot_convergence(history, f"{figures_dir}/global_convergence.png")
    plot_tour_3d(poses, opt_tour, names, f"{figures_dir}/global_tour_3d.png")

    return {
        "tour": opt_tour,
        "poses": poses,
        "names": names,
        "dist_matrix": dist_matrix,
        "nn_length": nn_length,
        "opt_length": opt_length,
        "history": history,
    }


if __name__ == "__main__":
    result = run_global_planner()
    print("\nOptimal visit order:")
    for rank, idx in enumerate(result["tour"]):
        print(f"  {rank + 1:2d}. {result['names'][idx]}")

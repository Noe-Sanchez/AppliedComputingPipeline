import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

def _set_axes_equal(ax) -> None:
    """Force equal data scaling on all three axes of a 3D matplotlib plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.set_zlim3d(0, limits[2, 1])


def _resample_uniform(trajectory: np.ndarray, n_points: int = 500) -> np.ndarray:
    """Resample a path to uniform arc-length spacing for smooth animation."""
    dists = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))]
    )
    uniform = np.linspace(0, dists[-1], n_points)
    return np.array([np.interp(uniform, dists, trajectory[:, c]) for c in range(3)]).T

def _plot_trajectory_3d(
    trajectory: np.ndarray,
    trees: np.ndarray,
    inspection_wps: np.ndarray,
    poses: np.ndarray,
    tour: list[int],
    figures_dir: str,
) -> None:
    """Animated 3-D plot with rotating camera and a drone marker. Press Q to close."""
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    MAX_RADIUS_M = 2.0  # tree crown radius for the cylinder constraint (m)

    # Trees as cones
    n_sides = 15
    theta = np.linspace(0, 2 * np.pi, n_sides)
    for pos in trees:
        x_base = pos[0] + MAX_RADIUS_M * np.cos(theta)
        y_base = pos[1] + MAX_RADIUS_M * np.sin(theta)
        for i in range(n_sides - 1):
            ax.plot_trisurf(
                [x_base[i], x_base[i + 1], pos[0]],
                [y_base[i], y_base[i + 1], pos[1]],
                [0, 0, pos[2]],
                color="green",
                alpha=0.3,
                linewidth=0,
            )
        ax.plot_trisurf(
            np.append(x_base, pos[0]),
            np.append(y_base, pos[1]),
            np.zeros(n_sides + 1),
            color="darkgreen",
            alpha=0.6,
            linewidth=0,
        )

    # TSP reference
    ordered = poses[tour]
    ax.plot(
        ordered[:, 0],
        ordered[:, 1],
        ordered[:, 2],
        "--",
        color="orange",
        linewidth=1.2,
        alpha=0.7,
        label="TSP reference",
    )
    ax.scatter(
        ordered[:, 0],
        ordered[:, 1],
        ordered[:, 2],
        c="orange",
        s=30,
        zorder=4,
        alpha=0.7,
    )

    # Full trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#185FA5",
        linewidth=1.2,
        alpha=0.4,
        label="Drone trajectory",
    )

    # Inspection waypoints
    ax.scatter(
        inspection_wps[:, 0],
        inspection_wps[:, 1],
        inspection_wps[:, 2],
        c="#D85A30",
        s=60,
        marker="^",
        zorder=5,
        label="Inspection WPs",
    )

    # Takeoff/landing point (the very first point of the trajectory)
    ax.scatter(
        [trajectory[0, 0]],
        [trajectory[0, 1]],
        [trajectory[0, 2]],
        c="black",
        s=80,
        marker="s",
        zorder=6,
        label="Takeoff/Landing",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Drone trajectory — rotating view  (Q to close)")
    ax.legend(loc="upper left", fontsize=8)
    _set_axes_equal(ax)

    # Animated drone marker with trail
    anim_traj = _resample_uniform(trajectory, n_points=300)
    n_frames = len(anim_traj)
    azim_start = 30
    elev_start = 70

    (drone_dot,) = ax.plot(
        [anim_traj[0, 0]],
        [anim_traj[0, 1]],
        [anim_traj[0, 2]],
        "o",
        color="red",
        markersize=10,
        zorder=10,
        label="Drone",
    )
    TRAIL = 120
    (trail_line,) = ax.plot([], [], [], "-", color="red", linewidth=2.0, alpha=0.6)

    def update(frame: int):
        drone_dot.set_data([anim_traj[frame, 0]], [anim_traj[frame, 1]])
        drone_dot.set_3d_properties([anim_traj[frame, 2]])
        trail_start = max(0, frame - TRAIL)
        trail_line.set_data(
            anim_traj[trail_start : frame + 1, 0],
            anim_traj[trail_start : frame + 1, 1],
        )
        trail_line.set_3d_properties(anim_traj[trail_start : frame + 1, 2])
        # elevation sense
        ax.view_init(
            elev=120 * np.abs(frame / n_frames - 0.5) + 10,
            azim=azim_start + 360 * frame / n_frames,
        )
        return drone_dot, trail_line

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=20, blit=False)

    # Save animation as GIF
    gif_out = f"{figures_dir}/drone_animation.gif"
    ani.save(gif_out, writer="pillow", fps=30)
    print(f"  [local_planner] animation saved → {gif_out}")

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out = f"{figures_dir}/local_trajectory_3d.png"
    fig.savefig(out, dpi=150)
    print(f"  [local_planner] 3-D trajectory plot saved → {out}")
    # plt.show()

def generate_report(module_data):

  flight_data     = module_data['flight_data']
  regression_fits = module_data['regression_fits']
  im_data         = module_data['im_data']

  opt_tour = module_data["opt_tour"]
  global_poses = module_data["global_poses"]
  names = module_data["names"]
  dist_matrix = module_data["dist_matrix"]
  nn_length = module_data["nn_length"]
  opt_length = module_data["opt_length"]
  history = module_data["history"]
  
  full_trajectory = module_data["full_trajectory"]
  history_labels = module_data["history_labels"]
  trees = module_data["trees"]
  waypoints = module_data["waypoints"]
  local_poses = module_data["local_poses"]
  tour = module_data["tour"]
  figures_dir = module_data["figures_dir"]
  clearance = module_data["clearance"]
  all_histories = module_data["all_histories"]
  segment_lengths = module_data["segment_lengths"]
  takeoff_xyz = module_data["takeoff_xyz"]

  rel_alts  = [flight_data['rel_alt'][i] for i in range(800)]
  variances = [im_data[i][1] for i in range(800)]
  max_variance = max(variances)
  variances = [v / max_variance for v in variances]

  # Flight data is a pandas DataFrame, and has rel_alt, Frame, latitude, longitude 
  plt.figure(figsize=(10, 6))
  plt.plot(flight_data['Frame'], flight_data['rel_alt'], label='Relative Altitude', color='blue')
  plt.xlabel('Frame')
  plt.ylabel('Relative Altitude (m)')
  plt.title('Relative Altitude over Time')
  plt.yticks([i for i in range(0, int(float(flight_data['rel_alt'].max())) + 10, 10)])
  plt.savefig('report/figures/relative_altitude.png')

  for coeffs, R2, RMSE in regression_fits:
    plt.figure(figsize=(10, 6))
    plt.plot(rel_alts, variances, color='red')
    plt.xlabel('Alt')
    plt.ylabel('Variance')
    x = rel_alts
    y = variances
    degree = len(coeffs) - 1 
    plt.title('Variance over Rel_Alt Fit of Degree {}'.format(degree))
    regression_line = np.polyval(coeffs, x)
    plt.plot(x, regression_line, color='blue', label='Fitted Regression Line')
    plt.legend()
    plt.savefig('report/figures/variance_fitted_degree_{}.png'.format(degree))
    
  residuals = variances - regression_line
  plt.figure(figsize=(10, 6))
  plt.scatter(regression_line, residuals, color='purple')
  plt.axhline(0, color='black', linestyle='--')
  plt.xlabel('Fitted Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Fitted Values for Degree {}'.format(degree))
  plt.savefig('report/figures/residuals_fitted_degree_{}.png'.format(degree))

  _plot_trajectory_3d(full_trajectory, trees, waypoints, local_poses, tour, figures_dir)


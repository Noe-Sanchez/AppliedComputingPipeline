"""
test_local_planner.py
---------------------
Unit tests for the SLSQP local planner.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from local_planner import (
    _altitude_constraints,
    optimise_segment,
    optimise_round_trip,
    MAX_RADIUS_M,
)


# ---------------------------------------------------------------------------
# _altitude_constraints
# ---------------------------------------------------------------------------


def test_clearance_satisfied_for_high_path():
    """Normal case: a path well above all trees has positive margins everywhere."""
    trees = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 6.0]])
    start = np.array([-5.0, 0.0, 50.0])
    end = np.array([15.0, 0.0, 50.0])
    interior = np.array([[5.0, 0.0, 50.0]])
    margins = _altitude_constraints(interior.flatten(), start, end, trees, clearance=5.0)
    assert margins.min() >= 0.0


def test_clearance_violated_when_too_low():
    """Edge case: a path passing through a tree at low altitude reports a violation."""
    trees = np.array([[0.0, 0.0, 10.0]])
    start = np.array([-5.0, 0.0, 2.0])
    end = np.array([5.0, 0.0, 2.0])
    interior = np.array([[0.0, 0.0, 2.0]])  # right inside the cylinder
    margins = _altitude_constraints(interior.flatten(), start, end, trees, clearance=5.0)
    assert margins.min() < 0.0


# ---------------------------------------------------------------------------
# optimise_segment
# ---------------------------------------------------------------------------


def test_segment_with_no_obstacles_is_nearly_straight():
    """Normal case: with no nearby trees, the optimised path ≈ straight line."""
    # One distant tree so the constraint is trivially satisfied.
    trees = np.array([[100.0, 100.0, 5.0]])
    start = np.array([0.0, 0.0, 20.0])
    end = np.array([10.0, 0.0, 20.0])

    waypoints, result, _ = optimise_segment(
        start, end, trees,
        n_waypoints=5, clearance=5.0, smoothness_weight=0.5,
    )

    length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(end - start))
    # Allow a few percent slack from interior smoothness term.
    assert length == pytest.approx(straight, rel=0.05)
    assert result.success or "iteration" in str(result.message).lower()


def test_segment_climbs_over_obstacle():
    """Edge case: a tree directly between start and end forces the path up."""
    trees = np.array([[5.0, 0.0, 15.0]])  # tall tree on the line
    start = np.array([0.0, 0.0, 20.0])
    end = np.array([10.0, 0.0, 20.0])
    clearance = 5.0

    waypoints, _, _ = optimise_segment(
        start, end, trees,
        n_waypoints=8, clearance=clearance, smoothness_weight=0.5,
    )

    # Any waypoint horizontally inside the tree cylinder must be above the
    # required altitude.
    required_z = trees[0, 2] + clearance
    for wp in waypoints:
        horiz = float(np.hypot(wp[0] - trees[0, 0], wp[1] - trees[0, 1]))
        if horiz <= MAX_RADIUS_M:
            assert wp[2] >= required_z - 1e-3


# ---------------------------------------------------------------------------
# optimise_round_trip
# ---------------------------------------------------------------------------


def test_round_trip_closes_at_takeoff_point():
    """Normal case: outbound start == return end (the shared takeoff point)."""
    trees = np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 6.0]])
    first = np.array([0.0, 0.0, 11.0])
    last = np.array([10.0, 10.0, 11.0])
    bounds = ((-5.0, 15.0), (-5.0, 15.0))

    outbound, retleg, takeoff, _, _ = optimise_round_trip(
        first_inspection=first,
        last_inspection=last,
        trees=trees,
        start_xy_bounds=bounds,
        n_waypoints=5,
        clearance=5.0,
    )

    # Outbound starts at takeoff, return ends at takeoff.
    assert np.allclose(outbound[0], takeoff)
    assert np.allclose(retleg[-1], takeoff)
    # Takeoff is on the ground and inside the box.
    assert takeoff[2] == pytest.approx(0.0)
    assert bounds[0][0] <= takeoff[0] <= bounds[0][1]
    assert bounds[1][0] <= takeoff[1] <= bounds[1][1]


def test_round_trip_takeoff_lies_between_endpoints():
    """The chosen takeoff (x, y) should land between (or near) first and last."""
    trees = np.array([[0.0, 0.0, 3.0], [20.0, 0.0, 3.0]])
    first = np.array([0.0, 0.0, 8.0])
    last = np.array([20.0, 0.0, 8.0])
    bounds = ((-5.0, 25.0), (-5.0, 5.0))

    _, _, takeoff, _, _ = optimise_round_trip(
        first_inspection=first,
        last_inspection=last,
        trees=trees,
        start_xy_bounds=bounds,
        n_waypoints=5,
        clearance=3.0,
    )

    # For a symmetric setup with no obstacles between, the optimum (x, y)
    # should sit on the segment connecting the two endpoints.
    assert -1.0 <= takeoff[0] <= 21.0
    assert abs(takeoff[1]) <= 2.0

"""
test_global_planner.py
----------------------
Unit tests for the TSP global planner.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from global_planner import (
    build_distance_matrix,
    nearest_neighbour_tour,
    tour_length,
    two_opt,
)


# ---------------------------------------------------------------------------
# build_distance_matrix
# ---------------------------------------------------------------------------


def test_distance_matrix_symmetric_and_zero_diagonal():
    """Normal case: a square of 4 points produces a symmetric matrix with zero diagonal."""
    poses = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    D = build_distance_matrix(poses)

    assert D.shape == (4, 4)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    # Adjacent corners are distance 1, diagonal corners are sqrt(2).
    assert D[0, 1] == pytest.approx(1.0)
    assert D[0, 2] == pytest.approx(np.sqrt(2))


def test_distance_matrix_rejects_single_pose():
    """Edge case: a single pose has no pairwise distances to compute."""
    with pytest.raises(AssertionError):
        build_distance_matrix(np.array([[0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------
# nearest_neighbour_tour
# ---------------------------------------------------------------------------


def test_nn_visits_every_node_once():
    """Normal case: NN on 5 random nodes visits each node exactly once."""
    rng = np.random.default_rng(42)
    poses = rng.uniform(-10, 10, size=(5, 3))
    D = build_distance_matrix(poses)
    tour = nearest_neighbour_tour(D, start=0)
    assert len(tour) == 5
    assert set(tour) == set(range(5))


# ---------------------------------------------------------------------------
# two_opt — invariants matter more than specific values
# ---------------------------------------------------------------------------


def test_two_opt_never_increases_tour_length():
    """Key invariant: 2-opt is monotone improving — final length ≤ initial."""
    rng = np.random.default_rng(7)
    poses = rng.uniform(-50, 50, size=(15, 3))
    D = build_distance_matrix(poses)

    nn_tour = nearest_neighbour_tour(D)
    nn_len = tour_length(nn_tour, D)

    opt_tour, history = two_opt(nn_tour, D)
    opt_len = tour_length(opt_tour, D)

    assert opt_len <= nn_len + 1e-9
    # 2-opt history is non-increasing.
    assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1))
    # Output is still a valid permutation.
    assert sorted(opt_tour) == list(range(len(poses)))


def test_two_opt_finds_optimum_on_known_square():
    """Edge case: 4-corner square has a known optimum of 3 (perimeter minus one edge)."""
    poses = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    D = build_distance_matrix(poses)
    nn_tour = nearest_neighbour_tour(D, start=0)
    opt_tour, _ = two_opt(nn_tour, D)
    # Open tour visits 4 points along 3 unit edges.
    assert tour_length(opt_tour, D) == pytest.approx(3.0)

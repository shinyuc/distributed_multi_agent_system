import numpy as np

from dmasim.controllers.constraints import obstacle_signed_margin, pairwise_distance


def test_pairwise_distance():
    assert pairwise_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_obstacle_margin():
    m = obstacle_signed_margin(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 1.0)
    assert m == 1.0

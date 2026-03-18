from __future__ import annotations

import numpy as np



def pairwise_distance(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return float(np.linalg.norm(x_i - x_j))



def obstacle_signed_margin(x_i: np.ndarray, obstacle_center: np.ndarray, obstacle_radius: float) -> float:
    return float(np.linalg.norm(x_i - obstacle_center) - obstacle_radius)


# NOTE:
# Insert your paper's exact CBF/safety/connectivity inequality definitions here.
# Keep this module as the single source of truth for constraints used by both controllers.

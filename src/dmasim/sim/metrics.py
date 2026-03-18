from __future__ import annotations

import numpy as np


def final_distance_to_target(positions: np.ndarray, target_center: np.ndarray) -> float:
    d = np.linalg.norm(positions - target_center[None, :], axis=1)
    return float(np.mean(d))

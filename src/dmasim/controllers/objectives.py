from __future__ import annotations

import numpy as np



def nominal_input_to_target(x_i: np.ndarray, target_center: np.ndarray, gain: float) -> np.ndarray:
    """Template for paper nominal term."""
    _ = (x_i, target_center, gain)
    raise NotImplementedError("Implement nominal objective term from the paper equations.")

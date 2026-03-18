from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QPProblem:
    # Minimize: 0.5 z^T H z + f^T z
    H: np.ndarray
    f: np.ndarray
    # Subject to: A z <= b
    A: np.ndarray
    b: np.ndarray


class QPSolver:
    def solve(self, problem: QPProblem) -> np.ndarray:
        raise NotImplementedError

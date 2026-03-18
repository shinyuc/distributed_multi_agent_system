from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryLog:
    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)

    def add(self, t: float, positions: np.ndarray) -> None:
        self.times.append(float(t))
        self.positions.append(np.array(positions, copy=True))

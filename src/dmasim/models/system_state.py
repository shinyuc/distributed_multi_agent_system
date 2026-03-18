from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SystemState:
    t: float
    positions: np.ndarray

    @property
    def n_agents(self) -> int:
        return int(self.positions.shape[0])

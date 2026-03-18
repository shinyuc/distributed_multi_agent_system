from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Agent:
    agent_id: int
    position: np.ndarray
    radius: float
    max_speed: float

    def step(self, u: np.ndarray, dt: float) -> None:
        speed = np.linalg.norm(u)
        if speed > self.max_speed:
            u = (u / speed) * self.max_speed
        self.position = self.position + dt * u

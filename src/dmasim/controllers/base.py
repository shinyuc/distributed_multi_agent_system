from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dmasim.env.world import World
from dmasim.models.system_state import SystemState


class BaseDistributedController(ABC):
    @abstractmethod
    def compute_control(self, agent_id: int, state: SystemState, world: World, t: float) -> np.ndarray:
        """Return u_i in R^2 for one agent."""


class BaseCentralizedController(ABC):
    @abstractmethod
    def compute_controls(self, state: SystemState, world: World, t: float) -> np.ndarray:
        """Return U in R^(N x 2)."""

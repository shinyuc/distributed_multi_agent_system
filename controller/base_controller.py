"""Controller interface for agent-level control laws."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.agent import AgentState, Vector2
from models.environment import Environment


class Controller(ABC):
    """Abstract controller interface.

    Concrete implementations may use CLF/CBF-QP or any other control method
    consistent with the simulation setup.
    """

    @abstractmethod
    def compute_control(self, agent: AgentState, env: Environment, **kwargs: object) -> Vector2:
        """Return control input for one agent in the current environment."""

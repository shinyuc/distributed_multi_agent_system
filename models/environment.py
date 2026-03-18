"""Environment data containers for multi-agent simulations."""

from __future__ import annotations

from dataclasses import dataclass

from models.agent import AgentState, Vector2


@dataclass
class CircularObstacle:
    center: Vector2
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"CircularObstacle.radius must be > 0, got {self.radius}.")


@dataclass
class TargetRegion:
    center: Vector2
    radius: float
    Sigma: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"TargetRegion.radius must be > 0, got {self.radius}.")
        if self.Sigma is not None:
            if (
                len(self.Sigma) != 2
                or any(len(row) != 2 for row in self.Sigma)
            ):
                raise ValueError("TargetRegion.Sigma must be a 2x2 matrix when provided.")


@dataclass
class Environment:
    agents: list[AgentState]
    obstacles: list[CircularObstacle]
    target: TargetRegion

    def num_agents(self) -> int:
        return len(self.agents)

    def agent_positions(self) -> list[Vector2]:
        return [agent.position for agent in self.agents]

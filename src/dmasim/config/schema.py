from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationConfig:
    dt: float
    steps: int
    save_trajectories: bool = True
    output_dir: str = "outputs"


@dataclass
class AgentConfig:
    n_agents: int
    radius: float
    max_speed: float
    initial_positions: list[list[float]]


@dataclass
class ObstacleConfig:
    center: list[float]
    radius: float


@dataclass
class TargetRegionConfig:
    center: list[float]
    radius: float


@dataclass
class EnvironmentConfig:
    bounds: list[float]
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    target_region: TargetRegionConfig | None = None


@dataclass
class SafetyConfig:
    min_inter_agent_distance: float
    obstacle_margin: float
    enforce_connectivity: bool
    max_connectivity_distance: float


@dataclass
class QPConfig:
    weight_u: float
    weight_slack: float
    use_slack: bool


@dataclass
class ControllerConfig:
    method: str
    solver: str
    gains: dict[str, float]
    safety: SafetyConfig
    qp: QPConfig


@dataclass
class RootConfig:
    seed: int
    simulation: SimulationConfig
    agents: AgentConfig
    environment: EnvironmentConfig
    controller: ControllerConfig



def _obstacle_from_dict(d: dict[str, Any]) -> ObstacleConfig:
    return ObstacleConfig(center=d["center"], radius=d["radius"])



def from_dict(data: dict[str, Any]) -> RootConfig:
    env = data["environment"]
    ctrl = data["controller"]

    target = None
    if env.get("target_region") is not None:
        target = TargetRegionConfig(**env["target_region"])

    return RootConfig(
        seed=data["seed"],
        simulation=SimulationConfig(**data["simulation"]),
        agents=AgentConfig(**data["agents"]),
        environment=EnvironmentConfig(
            bounds=env["bounds"],
            obstacles=[_obstacle_from_dict(o) for o in env.get("obstacles", [])],
            target_region=target,
        ),
        controller=ControllerConfig(
            method=ctrl["method"],
            solver=ctrl["solver"],
            gains=ctrl["gains"],
            safety=SafetyConfig(**ctrl["safety"]),
            qp=QPConfig(**ctrl["qp"]),
        ),
    )

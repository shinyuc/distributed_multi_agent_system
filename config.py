"""Configuration loading and validation for simulation experiments.

This module defines typed dataclasses for the YAML schema and a single
`load_config` entrypoint that reads a YAML file with `yaml.safe_load`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class SimulationConfig:
    """Global simulation settings."""

    dt: float
    steps: int
    seed: int


@dataclass
class AgentsConfig:
    """Agent population and initial state."""

    num_agents: int
    initial_positions: list[list[float]]


@dataclass
class ObstaclesConfig:
    """Obstacle list for a 2D workspace.

    Each obstacle is represented as:
    `{"center": [x, y], "radius": r}`.
    """

    obstacles: list[dict[str, Any]]


@dataclass
class TargetConfig:
    """Target region and CLF shaping parameters."""

    center: list[float]
    radius: float
    Sigma: list[list[float]]
    epsilon: float


@dataclass
class ControllerConfig:
    """Controller hyperparameters placeholder."""

    u_max: float
    safety_distance: float
    clf_rate: float
    cbf_rate: float
    nominal_gain: float
    mode: str = "distributed"
    enable_connectivity_constraint: bool = True
    alpha_clf: float = 1.0
    alpha_connectivity: float = 1.0
    debug_disable_clf_coupling: bool = False


@dataclass
class FastDynamicsConfig:
    """Fast dynamics parameters for the two-time-scale algorithm."""

    tau: float
    xi: float
    d_c: float
    eta: float = 0.1
    epsilon_c: float = 0.5
    lambda_2: float = 0.0
    v2: list[float] | None = None


@dataclass
class Config:
    """Top-level strongly typed experiment configuration."""

    simulation: SimulationConfig
    agents: AgentsConfig
    obstacles: ObstaclesConfig
    target: TargetConfig
    controller: ControllerConfig
    fast_dynamics: FastDynamicsConfig


def _validate_positions(initial_positions: list[list[float]]) -> None:
    for idx, pos in enumerate(initial_positions):
        if not isinstance(pos, list) or len(pos) != 2:
            raise ValueError(
                f"agents.initial_positions[{idx}] must be a list of length 2, got: {pos!r}"
            )


def _validate_obstacles(obstacles: list[dict[str, Any]]) -> None:
    for idx, obs in enumerate(obstacles):
        if "center" not in obs or "radius" not in obs:
            raise ValueError(
                f"obstacles.obstacles[{idx}] must contain keys 'center' and 'radius'."
            )
        center = obs["center"]
        radius = obs["radius"]
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError(
                f"obstacles.obstacles[{idx}].center must be a list of length 2."
            )
        if radius <= 0:
            raise ValueError(
                f"obstacles.obstacles[{idx}].radius must be > 0, got {radius}."
            )


def load_config(path: str) -> Config:
    """Load YAML config from `path` and return validated dataclass Config.

    Validation performed:
    - `agents.num_agents` equals `len(agents.initial_positions)`
    - `simulation.dt > 0` and `simulation.steps > 0`
    - every obstacle radius is > 0
    - target radius is > 0
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        simulation = SimulationConfig(**raw["simulation"])
        agents = AgentsConfig(**raw["agents"])
        obstacles = ObstaclesConfig(**raw["obstacles"])
        target = TargetConfig(**raw["target"])
        controller = ControllerConfig(**raw["controller"])
        fast_dynamics = FastDynamicsConfig(**raw["fast_dynamics"])
    except KeyError as exc:
        raise ValueError(f"Missing required config section/key: {exc}") from exc
    except TypeError as exc:
        raise ValueError(f"Invalid config schema: {exc}") from exc

    if simulation.dt <= 0:
        raise ValueError(f"simulation.dt must be > 0, got {simulation.dt}.")
    if simulation.steps <= 0:
        raise ValueError(f"simulation.steps must be > 0, got {simulation.steps}.")

    _validate_positions(agents.initial_positions)
    if agents.num_agents != len(agents.initial_positions):
        raise ValueError(
            "agents.num_agents must match len(agents.initial_positions): "
            f"{agents.num_agents} != {len(agents.initial_positions)}"
        )

    _validate_obstacles(obstacles.obstacles)

    if not isinstance(target.center, list) or len(target.center) != 2:
        raise ValueError("target.center must be a list of length 2.")
    if target.radius <= 0:
        raise ValueError(f"target.radius must be > 0, got {target.radius}.")
    if (
        not isinstance(target.Sigma, list)
        or len(target.Sigma) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in target.Sigma)
    ):
        raise ValueError("target.Sigma must be a 2x2 matrix (list of two length-2 lists).")
    if target.epsilon <= 0:
        raise ValueError(f"target.epsilon must be > 0, got {target.epsilon}.")
    controller.mode = controller.mode.lower()
    if controller.mode not in {"distributed", "centralized"}:
        raise ValueError(
            f"controller.mode must be 'distributed' or 'centralized', got {controller.mode!r}."
        )
    if fast_dynamics.tau <= 0:
        raise ValueError(f"fast_dynamics.tau must be > 0, got {fast_dynamics.tau}.")
    if fast_dynamics.xi < 0:
        raise ValueError(f"fast_dynamics.xi must be >= 0, got {fast_dynamics.xi}.")
    if fast_dynamics.d_c <= 0:
        raise ValueError(f"fast_dynamics.d_c must be > 0, got {fast_dynamics.d_c}.")
    if fast_dynamics.eta < 0:
        raise ValueError(f"fast_dynamics.eta must be >= 0, got {fast_dynamics.eta}.")
    if fast_dynamics.epsilon_c <= 0:
        raise ValueError(
            f"fast_dynamics.epsilon_c must be > 0, got {fast_dynamics.epsilon_c}."
        )
    if fast_dynamics.epsilon_c >= fast_dynamics.d_c:
        raise ValueError(
            "fast_dynamics.epsilon_c must satisfy epsilon_c < d_c for connectivity graph."
        )
    if fast_dynamics.v2 is not None and len(fast_dynamics.v2) != agents.num_agents:
        raise ValueError(
            "fast_dynamics.v2 must have one scalar per agent: "
            f"len(v2)={len(fast_dynamics.v2)} != num_agents={agents.num_agents}"
        )

    return Config(
        simulation=simulation,
        agents=agents,
        obstacles=obstacles,
        target=target,
        controller=controller,
        fast_dynamics=fast_dynamics,
    )

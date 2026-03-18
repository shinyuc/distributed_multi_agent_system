"""Fast-variable state containers for the two-time-scale distributed algorithm.

This file is intentionally data-only. It does not implement fast dynamics updates
or optimization steps yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.agent import Vector2


@dataclass
class EdgeMismatchState:
    """Per-edge mismatch components stored locally by agent i.

    For fixed agent i and edge (i, j):
    - self_component      -> z_i^{i,j}
    - neighbor_component  -> z_j^{i,j}
    """

    self_component: float
    neighbor_component: float


@dataclass
class EdgeDualState:
    """Per-edge dual components stored locally by agent i.

    For fixed agent i and edge (i, j):
    - self_component      -> y_i^{i,j}
    - neighbor_component  -> y_j^{i,j}
    """

    self_component: float
    neighbor_component: float


@dataclass
class FastState:
    """Fast variables for one agent i.

    Indexing convention for pair variables:
    - ``z_pair[j].self_component`` corresponds to ``z_i^{i,j}``
    - ``z_pair[j].neighbor_component`` corresponds to ``z_j^{i,j}``
    - ``y_pair[j].self_component`` corresponds to ``y_i^{i,j}``
    - ``y_pair[j].neighbor_component`` corresponds to ``y_j^{i,j}``

    For one fixed agent i, store exactly one entry per other agent ``j != i``.
    Do not store duplicated mirrored edge keys.
    """

    w: Vector2
    z_pair: dict[int, EdgeMismatchState]
    z_c: float
    z_clf: float
    y_pair: dict[int, EdgeDualState]
    y_c: float
    y_clf: float
    y_obs: list[float]
    y_u: list[float]


def init_fast_state_complete_graph(i: int, n_agents: int, n_obstacles: int = 0) -> FastState:
    """Initialize FastState for agent i under a complete graph assumption.

    Args:
        i: Current agent index.
        n_agents: Total number of agents N.

    Returns:
        FastState with:
        - w = (0.0, 0.0)
        - z_c, z_clf, y_c, y_clf = 0.0
        - y_obs[k] = 0.0 for each obstacle k
        - y_u[l] = 0.0 for l = 0,1,2,3
        - for all j != i:
          z_pair[j].self_component = 0.0
          z_pair[j].neighbor_component = 0.0
          y_pair[j].self_component = 0.0
          y_pair[j].neighbor_component = 0.0
    """

    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}.")
    if i < 0 or i >= n_agents:
        raise ValueError(f"agent index i must satisfy 0 <= i < n_agents, got i={i}.")

    pair_indices = [j for j in range(n_agents) if j != i]
    z_pair = {
        j: EdgeMismatchState(self_component=0.0, neighbor_component=0.0)
        for j in pair_indices
    }
    y_pair = {
        j: EdgeDualState(self_component=0.0, neighbor_component=0.0)
        for j in pair_indices
    }

    return FastState(
        w=(0.0, 0.0),
        z_pair=z_pair,
        z_c=0.0,
        z_clf=0.0,
        y_pair=y_pair,
        y_c=0.0,
        y_clf=0.0,
        y_obs=[0.0 for _ in range(n_obstacles)],
        y_u=[0.0, 0.0, 0.0, 0.0],
    )

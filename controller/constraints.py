"""Mathematical helpers for control barrier-style constraints."""

from __future__ import annotations

from models.agent import Vector2


def obstacle_constraint_value(
    agent_position: Vector2,
    control_input: Vector2,
    obstacle_center: Vector2,
    obstacle_radius: float,
    gamma_2: float,
) -> float:
    r"""Compute the obstacle constraint scalar

    g_obs^{i,k}(u_i) =
        -2 (x_i - s_k)^T u_i
        - gamma_2 ( ||x_i - s_k||^2 - r_k^2 ).

    Safety condition: g_obs^{i,k}(u_i) <= 0.
    """
    if obstacle_radius <= 0:
        raise ValueError(f"obstacle_radius must be > 0, got {obstacle_radius}.")
    if gamma_2 <= 0:
        raise ValueError(f"gamma_2 must be > 0, got {gamma_2}.")

    dx = agent_position[0] - obstacle_center[0]
    dy = agent_position[1] - obstacle_center[1]
    ux, uy = control_input

    dist_sq = dx * dx + dy * dy
    return -2.0 * (dx * ux + dy * uy) - gamma_2 * (dist_sq - obstacle_radius * obstacle_radius)


def obstacle_constraint_coefficients(
    agent_position: Vector2,
    obstacle_center: Vector2,
    obstacle_radius: float,
    gamma_2: float,
) -> tuple[Vector2, float]:
    r"""Return affine coefficients (a, b) for a^T u_i <= b.

    From:
        g_obs^{i,k}(u_i) =
            -2 (x_i - s_k)^T u_i
            - gamma_2 ( ||x_i - s_k||^2 - r_k^2 ) <= 0

    Rearranged:
        (-2 (x_i - s_k))^T u_i <= gamma_2 ( ||x_i - s_k||^2 - r_k^2 ).

    Therefore:
        a = -2 (x_i - s_k),
        b = gamma_2 ( ||x_i - s_k||^2 - r_k^2 ).
    """
    if obstacle_radius <= 0:
        raise ValueError(f"obstacle_radius must be > 0, got {obstacle_radius}.")
    if gamma_2 <= 0:
        raise ValueError(f"gamma_2 must be > 0, got {gamma_2}.")

    dx = agent_position[0] - obstacle_center[0]
    dy = agent_position[1] - obstacle_center[1]
    dist_sq = dx * dx + dy * dy

    a: Vector2 = (-2.0 * dx, -2.0 * dy)
    b = gamma_2 * (dist_sq - obstacle_radius * obstacle_radius)
    return a, b

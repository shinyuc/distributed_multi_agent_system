from __future__ import annotations

from controller.constraints import (
    obstacle_constraint_coefficients,
    obstacle_constraint_value,
)


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def test_obstacle_constraint_value_hand_check_far_case() -> None:
    # Hand-checkable setup:
    # x=(3,0), s=(0,0), r=1, gamma_2=2, u=(0.5,0)
    # g = -2*(3*0.5) - 2*(9-1) = -3 - 16 = -19
    g = obstacle_constraint_value(
        agent_position=(3.0, 0.0),
        control_input=(0.5, 0.0),
        obstacle_center=(0.0, 0.0),
        obstacle_radius=1.0,
        gamma_2=2.0,
    )
    assert abs(g - (-19.0)) < 1e-12
    assert g <= 0.0


def test_affine_coefficients_equivalent_to_g_obs_far_and_near() -> None:
    # Far from obstacle
    x_far = (2.5, -0.5)
    u_far = (0.2, -0.1)
    s = (0.0, 0.0)
    r = 1.0
    gamma_2 = 1.5

    g_far = obstacle_constraint_value(x_far, u_far, s, r, gamma_2)
    a_far, b_far = obstacle_constraint_coefficients(x_far, s, r, gamma_2)
    lhs_far = _dot(a_far, u_far)
    assert abs(lhs_far - b_far - g_far) < 1e-12
    assert (g_far <= 0.0) == (lhs_far <= b_far)

    # Near obstacle
    x_near = (1.05, 0.0)
    u_near = (0.2, 0.0)
    g_near = obstacle_constraint_value(x_near, u_near, s, r, gamma_2)
    a_near, b_near = obstacle_constraint_coefficients(x_near, s, r, gamma_2)
    lhs_near = _dot(a_near, u_near)
    assert abs(lhs_near - b_near - g_near) < 1e-12
    assert (g_near <= 0.0) == (lhs_near <= b_near)

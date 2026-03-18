"""Fast dynamics update architecture for one agent.

This module defines placeholder update functions for the fast variables used by
the two-time-scale distributed algorithm. Mathematical formulas are intentionally
not implemented yet.
"""

from __future__ import annotations

import math
from typing import Any

from controller.constraints import obstacle_constraint_value
from models.agent import AgentState, Vector2
from models.environment import CircularObstacle
from models.fast_state import EdgeDualState, EdgeMismatchState, FastState


def compute_beta_i(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    d_c: float,
    epsilon_c: float,
    v2: dict[int, float] | None = None,
) -> Vector2:
    """Compute beta_i as a row vector for the connectivity term.

    beta_i = sum_{j in N_i} (partial alpha_ij / partial x_i) * (v2^i - v2^j)^2
    where N_i uses the connectivity graph threshold d_c - epsilon_c.
    """
    _ = (state, all_fast_states)
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if epsilon_c <= 0.0:
        raise ValueError(f"epsilon_c must be > 0, got {epsilon_c}.")
    if epsilon_c >= d_c:
        raise ValueError(f"epsilon_c must satisfy epsilon_c < d_c, got {epsilon_c} >= {d_c}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if v2 is None:
        raise ValueError("Runtime v2 is required to compute beta_i for connectivity.")
    if i not in v2:
        raise KeyError(f"Missing v2 entry for i={i}.")

    x_i = all_agent_states[i].position
    sigma_conn = connectivity_sigma(d_c=d_c, epsilon_c=epsilon_c)
    beta_x = 0.0
    beta_y = 0.0
    for j, x_j_state in all_agent_states.items():
        if j == i:
            continue
        if j not in v2:
            raise KeyError(f"Missing v2 entry for j={j}.")
        d_alpha_row = partial_alpha_ij_wrt_x_i(
            x_i=x_i,
            x_j=x_j_state.position,
            d_c=d_c,
            epsilon_c=epsilon_c,
            sigma=sigma_conn,
        )
        dv2 = v2[i] - v2[j]
        weight = dv2 * dv2
        beta_x += d_alpha_row[0] * weight
        beta_y += d_alpha_row[1] * weight
    return (beta_x, beta_y)


def connectivity_sigma(d_c: float, epsilon_c: float) -> float:
    """Compute sigma = ((d_c - epsilon_c)^4) / ln(2) for connectivity weights."""
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if epsilon_c <= 0.0:
        raise ValueError(f"epsilon_c must be > 0, got {epsilon_c}.")
    if epsilon_c >= d_c:
        raise ValueError(f"epsilon_c must satisfy epsilon_c < d_c, got {epsilon_c} >= {d_c}.")
    d_thresh = d_c - epsilon_c
    return (d_thresh ** 4) / math.log(2.0)


def partial_alpha_ij_wrt_x_i(
    x_i: Vector2,
    x_j: Vector2,
    d_c: float,
    epsilon_c: float,
    sigma: float,
) -> Vector2:
    """Return (partial alpha_ij / partial x_i) as a row vector."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    d_thresh = d_c - epsilon_c
    dx = x_i[0] - x_j[0]
    dy = x_i[1] - x_j[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > d_thresh * d_thresh:
        return (0.0, 0.0)

    delta = d_thresh * d_thresh - dist_sq
    exp_term = math.exp((delta * delta) / sigma)
    coeff = -(4.0 / sigma) * delta * exp_term
    return (coeff * dx, coeff * dy)


def alpha_ij_connectivity(
    x_i: Vector2,
    x_j: Vector2,
    d_c: float,
    epsilon_c: float,
    sigma: float,
) -> float:
    """Connectivity-graph edge weight alpha_ij (separate from h_ij graph)."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    d_thresh = d_c - epsilon_c
    dx = x_i[0] - x_j[0]
    dy = x_i[1] - x_j[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > d_thresh * d_thresh:
        return 0.0

    delta = d_thresh * d_thresh - dist_sq
    return math.exp((delta * delta) / sigma) - 1.0


def compute_V_i(
    x_i: Vector2,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
) -> float:
    """Compute V_i(x_i) = (x_i-c)^T Sigma (x_i-c) - 1 + epsilon."""
    if (
        len(sigma) != 2
        or any(len(row) != 2 for row in sigma)
    ):
        raise ValueError("sigma must be a 2x2 matrix.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")

    dx0 = x_i[0] - target_center[0]
    dx1 = x_i[1] - target_center[1]
    s00, s01 = sigma[0]
    s10, s11 = sigma[1]
    sigma_dx0 = s00 * dx0 + s01 * dx1
    sigma_dx1 = s10 * dx0 + s11 * dx1
    return dx0 * sigma_dx0 + dx1 * sigma_dx1 - 1.0 + epsilon


def compute_clf_gradient_term(
    x_i: Vector2,
    target_center: Vector2,
    sigma: list[list[float]],
    y_i_clf: float,
    v_i: float,
) -> Vector2:
    """Compute the CLF contribution to w dynamics.

    Returns:
        -2 exp(-1 / V_i) (1 + 1 / V_i) Sigma x_i y_i^{clf}, if V_i > 0
        (0, 0), otherwise.
    """
    if v_i <= 0.0:
        return (0.0, 0.0)
    if (
        len(sigma) != 2
        or any(len(row) != 2 for row in sigma)
    ):
        raise ValueError("sigma must be a 2x2 matrix.")

    dx0 = x_i[0] - target_center[0]
    dx1 = x_i[1] - target_center[1]
    s00, s01 = sigma[0]
    s10, s11 = sigma[1]
    sigma_x0 = s00 * dx0 + s01 * dx1
    sigma_x1 = s10 * dx0 + s11 * dx1
    coeff = -2.0 * math.exp(-1.0 / v_i) * (1.0 + 1.0 / v_i) * y_i_clf
    return (coeff * sigma_x0, coeff * sigma_x1)


def project_plus(a: float, b: float) -> float:
    """Projection operator [a]_b^+ with small roundoff tolerance on b.

    Tiny negative values of b (e.g. -1e-18) are treated as 0.0.
    """
    tol = 1e-12
    if b > -tol:
        b = max(0.0, b)

    if b > 0.0:
        return a
    if b == 0.0:
        return max(0.0, a)
    raise ValueError(f"Projection base b must satisfy b >= 0, got {b}.")


def pairwise_constraint_value(
    x_i: Vector2,
    x_j: Vector2,
    w_i: Vector2,
    z_i_ij: float,
    z_j_ij: float,
    gamma_1: float,
    d_0: float,
    d_c: float,
) -> float:
    """Compute g_i^{i,j}(w_i, x_i, x_j, z_i^{i,j}, z_j^{i,j})."""
    if gamma_1 <= 0.0:
        raise ValueError(f"gamma_1 must be > 0, got {gamma_1}.")
    if d_0 <= 0.0:
        raise ValueError(f"d_0 must be > 0, got {d_0}.")
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")

    dx = x_i[0] - x_j[0]
    dy = x_i[1] - x_j[1]
    h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
    dist_sq = dx * dx + dy * dy
    drift_term = -2.0 * (dx * w_i[0] + dy * w_i[1])
    barrier_term = -0.5 * gamma_1 * (dist_sq - d_0 * d_0)
    mismatch_term = z_i_ij - z_j_ij
    return h * (drift_term + barrier_term + mismatch_term)


def lambda_2_placeholder(*_: Any) -> float:
    """Placeholder for lambda_2 until consensus/estimation is implemented."""
    return 0.0


def connectivity_constraint_value(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    x_i: Vector2,
    all_agent_states: dict[int, AgentState],
    w_i: Vector2,
    eta: float,
    lambda_2: float,
    d_c: float,
    epsilon_c: float,
    v2: dict[int, float] | None,
    n_agents: int,
) -> float:
    """Compute g_i^c for connectivity dual dynamics."""
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}.")

    beta_i = compute_beta_i(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        all_agent_states=all_agent_states,
        d_c=d_c,
        epsilon_c=epsilon_c,
        v2=v2,
    )
    print(
        f"[DEBUG beta] agent={i}, beta=({beta_i[0]:.6f}, {beta_i[1]:.6f})"
        )
    beta_u_term = -(beta_i[0] * w_i[0] + beta_i[1] * w_i[1])

    z_i_c = state.z_c
    coupling_sum = 0.0
    for j, fast_j in all_fast_states.items():
        if j == i:
            continue
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        coupling_sum += h * (z_i_c - fast_j.z_c)

    return beta_u_term - (lambda_2 / n_agents) + (eta / n_agents) + coupling_sum


def clf_constraint_value(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    x_i: Vector2,
    all_agent_states: dict[int, AgentState],
    w_i: Vector2,
    gamma_3: float,
    d_c: float,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
) -> float:
    """Compute g_i^{clf}(w_i, x_i, {x_j}, z_i^{clf}, {z_j^{clf}})."""
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if gamma_3 <= 0.0:
        raise ValueError(f"gamma_3 must be > 0, got {gamma_3}.")
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if (
        len(sigma) != 2
        or any(len(row) != 2 for row in sigma)
    ):
        raise ValueError("sigma must be a 2x2 matrix.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")

    z_i_clf = state.z_clf
    v_i = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
    h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
    if h_i == 0.0:
        return 0.0

    coupling_sum = 0.0
    for j, fast_j in all_fast_states.items():
        if j == i:
            continue
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        v_j = compute_V_i(x_i=x_j, target_center=target_center, sigma=sigma, epsilon=epsilon)
        h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
        coupling_sum += h * h_i * h_j * (z_i_clf - fast_j.z_clf)

    s00, s01 = sigma[0]
    s10, s11 = sigma[1]
    sigma_w0 = s00 * w_i[0] + s01 * w_i[1]
    sigma_w1 = s10 * w_i[0] + s11 * w_i[1]
    dx0 = x_i[0] - target_center[0]
    dx1 = x_i[1] - target_center[1]
    x_sigma_w = dx0 * sigma_w0 + dx1 * sigma_w1

    exp_factor = math.exp(-1.0 / v_i)
    clf_drift = 2.0 * exp_factor * (1.0 + 1.0 / v_i) * x_sigma_w
    w_i_scalar = exp_factor * v_i
    return clf_drift + gamma_3 * w_i_scalar + coupling_sum


def update_w(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    dt: float,
    tau: float,
    d_c: float,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
    epsilon_c: float = 0.5,
    v2: dict[int, float] | None = None,
    **extra: Any,
) -> Vector2:
    """Update the fast variable w_i with forward Euler.

    Uses only the effective duals currently modeled:
    - y_i^{i,j} (from y_pair[j].self_component)
    - y_i^c
    - y_i^{clf}

    Obstacle and actuator dual terms are intentionally omitted at this stage.
    """
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")

    x_i = all_agent_states[i].position
    w_old = state.w
    step_idx = extra.get("step_idx")
    log_pairwise_debug = isinstance(step_idx, int) and step_idx in {0, 1, 2, 3}
    log_w_rhs_debug = isinstance(step_idx, int) and step_idx in {0, 1, 2}
    log_w_agent2 = isinstance(step_idx, int) and step_idx in {0, 1, 2, 3} and i == 2
    log_w_clf_agent2 = isinstance(step_idx, int) and step_idx in {1, 2} and i == 2
    obstacles = extra.get("obstacles")
    if obstacles is None:
        obstacles = []
    if not isinstance(obstacles, list):
        raise ValueError("obstacles must be a list[CircularObstacle].")
    if len(state.y_obs) != len(obstacles):
        raise ValueError(
            f"state.y_obs must have one scalar per obstacle: "
            f"len(y_obs)={len(state.y_obs)} != len(obstacles)={len(obstacles)}"
        )
    if len(state.y_u) != 4:
        raise ValueError(f"state.y_u must have length 4, got {len(state.y_u)}.")

    pair_sum_x = 0.0
    pair_sum_y = 0.0
    for j, y_edge in state.y_pair.items():
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        y_i_ij = y_edge.self_component
        diff_x = x_i[0] - x_j[0]
        diff_y = x_i[1] - x_j[1]
        coeff = 2.0 * h * y_i_ij
        pair_sum_x += coeff * diff_x
        pair_sum_y += coeff * diff_y
        if log_pairwise_debug:
            contribution_x = coeff * diff_x
            contribution_y = coeff * diff_y
            contribution_norm = math.sqrt(contribution_x * contribution_x + contribution_y * contribution_y)
            print(
                f"[DEBUG pairwise w term] step={step_idx}, agent={i}, neighbor={j}, "
                f"h_ij={h:.12f}, y_i_ij={y_i_ij:.12f}, "
                f"dx={diff_x:.12f}, dy={diff_y:.12f}, "
                f"contribution=({contribution_x:.12f}, {contribution_y:.12f}), "
                f"contribution_norm={contribution_norm:.12f}"
            )

    beta_i = compute_beta_i(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        all_agent_states=all_agent_states,
        d_c=d_c,
        epsilon_c=epsilon_c,
        v2=v2,
    )
    y_i_c = state.y_c
    beta_term = (beta_i[0] * y_i_c, beta_i[1] * y_i_c)

    v_i = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
    clf_term = compute_clf_gradient_term(
        x_i=x_i,
        target_center=target_center,
        sigma=sigma,
        y_i_clf=state.y_clf,
        v_i=v_i,
    )
    if log_w_clf_agent2:
        dx0 = x_i[0] - target_center[0]
        dx1 = x_i[1] - target_center[1]
        s00, s01 = sigma[0]
        s10, s11 = sigma[1]
        sigma_x0 = s00 * dx0 + s01 * dx1
        sigma_x1 = s10 * dx0 + s11 * dx1
        exp_factor_clf = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
        w_i_scalar_clf = exp_factor_clf * v_i if v_i > 0.0 else 0.0
        clf_coeff = (
            -2.0 * exp_factor_clf * (1.0 + 1.0 / v_i) * state.y_clf
            if v_i > 0.0
            else 0.0
        )
        print(
            f"[DEBUG W CLF] step={step_idx}, agent=2: "
            f"x_i=({x_i[0]:.12f}, {x_i[1]:.12f}), "
            f"w_old=({w_old[0]:.12f}, {w_old[1]:.12f}), "
            f"z_clf={state.z_clf:.12f}, y_clf={state.y_clf:.12f}, "
            f"V_i={v_i:.12f}, exp_factor={exp_factor_clf:.12f}, "
            f"W_i={w_i_scalar_clf:.12f}, "
            f"sigma_x=({sigma_x0:.12f}, {sigma_x1:.12f}), "
            f"clf_coeff={clf_coeff:.12f}, "
            f"clf_term=({clf_term[0]:.12f}, {clf_term[1]:.12f})"
        )
    obstacle_term_x = 0.0
    obstacle_term_y = 0.0
    for obstacle, y_obs in zip(obstacles, state.y_obs):
        dx_obs = x_i[0] - obstacle.center[0]
        dy_obs = x_i[1] - obstacle.center[1]
        obstacle_term_x += 2.0 * dx_obs * y_obs
        obstacle_term_y += 2.0 * dy_obs * y_obs

    input_bound_term = (
        -state.y_u[0] + state.y_u[2],
        -state.y_u[1] + state.y_u[3],
    )

    rhs_x = (
        -w_old[0]
        + pair_sum_x
        + beta_term[0]
        + clf_term[0]
        + obstacle_term_x
        + input_bound_term[0]
    )
    rhs_y = (
        -w_old[1]
        + pair_sum_y
        + beta_term[1]
        + clf_term[1]
        + obstacle_term_y
        + input_bound_term[1]
    )
    w_next = (
        w_old[0] + (dt / tau) * rhs_x,
        w_old[1] + (dt / tau) * rhs_y,
    )
    if log_pairwise_debug:
        pair_sum_norm = math.sqrt(pair_sum_x * pair_sum_x + pair_sum_y * pair_sum_y)
        print(
            f"[DEBUG pairwise w total] step={step_idx}, agent={i}, "
            f"pair_sum=({pair_sum_x:.12f}, {pair_sum_y:.12f}), "
            f"pair_sum_norm={pair_sum_norm:.12f}"
        )

    if log_w_rhs_debug:
        damping_term = (-w_old[0], -w_old[1])
        pair_sum_term = (pair_sum_x, pair_sum_y)
        obstacle_term = (obstacle_term_x, obstacle_term_y)
        total_rhs = (rhs_x, rhs_y)
        damping_norm = math.sqrt(damping_term[0] * damping_term[0] + damping_term[1] * damping_term[1])
        pair_sum_norm = math.sqrt(pair_sum_term[0] * pair_sum_term[0] + pair_sum_term[1] * pair_sum_term[1])
        beta_term_norm = math.sqrt(beta_term[0] * beta_term[0] + beta_term[1] * beta_term[1])
        clf_term_norm = math.sqrt(clf_term[0] * clf_term[0] + clf_term[1] * clf_term[1])
        obstacle_term_norm = math.sqrt(
            obstacle_term[0] * obstacle_term[0] + obstacle_term[1] * obstacle_term[1]
        )
        input_term_norm = math.sqrt(
            input_bound_term[0] * input_bound_term[0] + input_bound_term[1] * input_bound_term[1]
        )
        rhs_norm = math.sqrt(total_rhs[0] * total_rhs[0] + total_rhs[1] * total_rhs[1])
        print(f"[DEBUG update_w rhs] step={step_idx}, agent={i}")
        print(f"  w_old=({w_old[0]:.12f}, {w_old[1]:.12f})")
        print(f"  y_c={y_i_c:.12f}")
        print(f"  y_clf={state.y_clf:.12f}")
        print(f"  y_obs={[float(value) for value in state.y_obs]}")
        print(f"  y_u={[float(value) for value in state.y_u]}")
        print(
            f"  damping_term=({damping_term[0]:.12f}, {damping_term[1]:.12f}), "
            f"norm={damping_norm:.12f}"
        )
        print(
            f"  pairwise_term=({pair_sum_term[0]:.12f}, {pair_sum_term[1]:.12f}), "
            f"norm={pair_sum_norm:.12f}"
        )
        print(f"  beta_i=({beta_i[0]:.12f}, {beta_i[1]:.12f})")
        print(
            f"  connectivity_term=({beta_term[0]:.12f}, {beta_term[1]:.12f}), "
            f"norm={beta_term_norm:.12f}"
        )
        print(f"  v_i={v_i:.12f}")
        print(
            f"  clf_term=({clf_term[0]:.12f}, {clf_term[1]:.12f}), "
            f"norm={clf_term_norm:.12f}"
        )
        print(
            f"  obstacle_term=({obstacle_term[0]:.12f}, {obstacle_term[1]:.12f}), "
            f"norm={obstacle_term_norm:.12f}"
        )
        print(
            f"  input_bound_term=({input_bound_term[0]:.12f}, {input_bound_term[1]:.12f}), "
            f"norm={input_term_norm:.12f}"
        )
        print(
            f"  total_rhs=({total_rhs[0]:.12f}, {total_rhs[1]:.12f}), "
            f"norm={rhs_norm:.12f}"
        )
        print(f"  w_next=({w_next[0]:.12f}, {w_next[1]:.12f})")
    if log_w_agent2:
        damping_term = (-w_old[0], -w_old[1])
        pair_sum_term = (pair_sum_x, pair_sum_y)
        obstacle_term = (obstacle_term_x, obstacle_term_y)
        total_rhs = (rhs_x, rhs_y)
        print(f"[DEBUG W] step={step_idx}, agent={i}")
        print(f"  w_old=({w_old[0]:.12f}, {w_old[1]:.12f})")
        print(f"  pairwise=({pair_sum_term[0]:.12f}, {pair_sum_term[1]:.12f})")
        print(f"  obstacle=({obstacle_term[0]:.12f}, {obstacle_term[1]:.12f})")
        print(f"  connectivity=({beta_term[0]:.12f}, {beta_term[1]:.12f})")
        print(f"  clf=({clf_term[0]:.12f}, {clf_term[1]:.12f})")
        print(
            f"  actuator=({input_bound_term[0]:.12f}, {input_bound_term[1]:.12f})"
        )
        print(f"  damping=({damping_term[0]:.12f}, {damping_term[1]:.12f})")
        print(f"  total_rhs=({total_rhs[0]:.12f}, {total_rhs[1]:.12f})")
        print(f"  w_new=({w_next[0]:.12f}, {w_next[1]:.12f})")

    return w_next


def h_ij(x_i: Vector2, x_j: Vector2, d_c: float) -> float:
    """Communication weight h_{i,j}(x_i, x_j, d_c).

    h_{i,j}(x_i, x_j) =
        exp(-1 / (d_c - ||x_i - x_j||))   if d_c - ||x_i - x_j|| > 0
        0                                  otherwise
    """

    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")

    dx = x_i[0] - x_j[0]
    dy = x_i[1] - x_j[1]
    dist = math.sqrt(dx * dx + dy * dy)
    gap = d_c - dist
    if gap <= 0.0:
        return 0.0
    return math.exp(-1.0 / gap)


def update_z_pair(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    dt: float,
    tau: float,
    xi: float,
    d_c: float,
    **extra: Any,
) -> dict[int, EdgeMismatchState]:
    """Update z-pair states for agent i on all edges (i, j), j != i.

    Implemented part in this step:
    - Update only self component z_i^{i,j} using

      tau * z_dot_i^{i,j}
        = -xi * z_i^{i,j} - h_{i,j}(x_i, x_j) * (y_i^{i,j} - y_j^{i,j})

      with forward Euler:
        z_new = z_old + (dt / tau) * rhs

    Not implemented yet:
    - neighbor component z_j^{i,j} update (kept unchanged).
    """
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if xi < 0:
        raise ValueError(f"xi must be >= 0, got {xi}.")
    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")

    updated: dict[int, EdgeMismatchState] = {}
    x_i = all_agent_states[i].position
    step_idx = extra.get("step_idx")
    gamma_1 = extra.get("gamma_1")
    d_0 = extra.get("d_0")
    log_agent2_step3 = (
        step_idx == 3
        and i == 2
        and isinstance(gamma_1, (int, float))
        and isinstance(d_0, (int, float))
    )
    for j, z_edge in state.z_pair.items():
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        if j not in all_fast_states:
            raise KeyError(f"Missing fast state for j={j}.")
        if j not in all_fast_states[i].y_pair:
            raise KeyError(f"Missing y-pair entry for edge (i={i}, j={j}).")
        if i not in all_fast_states[j].y_pair:
            raise KeyError(f"Missing mirrored y-pair entry for edge (j={j}, i={i}).")

        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)

        y_edge = all_fast_states[i].y_pair[j]
        y_i_ij = y_edge.self_component
        y_j_ij = all_fast_states[j].y_pair[i].self_component
        damp_term = -xi * z_edge.self_component
        coupling_term = -h * (y_i_ij - y_j_ij)
        if log_agent2_step3:
            z_j_ij_live = all_fast_states[j].z_pair[i].self_component
            g_ij = pairwise_constraint_value(
                x_i=x_i,
                x_j=x_j,
                w_i=state.w,
                z_i_ij=z_edge.self_component,
                z_j_ij=z_j_ij_live,
                gamma_1=float(gamma_1),
                d_0=float(d_0),
                d_c=d_c,
            )
            print(
                f"[DEBUG STEP3 Z] pair (2,{j}): "
                f"w_2=({state.w[0]:.12f}, {state.w[1]:.12f}), "
                f"z_i={z_edge.self_component:.12f}, z_j={z_j_ij_live:.12f}, "
                f"z_diff={z_edge.self_component - z_j_ij_live:.12f}, "
                f"y_i={y_i_ij:.12f}, y_j={y_j_ij:.12f}, "
                f"z_rhs={damp_term + coupling_term:.12f}, "
                f"damp={damp_term:.12f}, coupling={coupling_term:.12f}"
            )

        z_old = z_edge.self_component
        rhs = damp_term + coupling_term
        z_new = z_old + (dt / tau) * rhs

        updated[j] = EdgeMismatchState(
            self_component=z_new,
            neighbor_component=z_edge.neighbor_component,  # keep unchanged for now
        )

    return updated


def update_z_c(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    dt: float,
    tau: float,
    xi: float,
    d_c: float,
    **_: Any,
) -> float:
    """Update z_i^c using forward Euler.

    Implements:
        tau * z_dot_i^c
          = -xi * z_i^c - sum_{j != i} h_{i,j}(x_i, x_j) * (y_i^c - y_j^c)

    with:
        z_c_new = z_c_old + (dt / tau) * rhs
    """
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if xi < 0:
        raise ValueError(f"xi must be >= 0, got {xi}.")
    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")

    x_i = all_agent_states[i].position
    y_i_c = all_fast_states[i].y_c
    coupling_sum = 0.0
    for j, fast_j in all_fast_states.items():
        if j == i:
            continue
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        coupling_sum += h * (y_i_c - fast_j.y_c)

    z_c_old = state.z_c
    rhs = -xi * z_c_old - coupling_sum
    return z_c_old + (dt / tau) * rhs


def update_z_clf(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    dt: float,
    tau: float,
    xi: float,
    alpha_clf: float,
    d_c: float,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
    **_: Any,
) -> float:
    """Update z_i^{clf} using forward Euler.

    Implements:
        tau * z_dot_i^{clf}
          = -xi * z_i^{clf}
            - sum_{j != i} h_{i,j}(x_i, x_j) * (y_i^{clf} - y_j^{clf})

    with:
        z_clf_new = z_clf_old + (dt / tau) * rhs
    """
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if xi < 0:
        raise ValueError(f"xi must be >= 0, got {xi}.")
    if alpha_clf < 0:
        raise ValueError(f"alpha_clf must be >= 0, got {alpha_clf}.")
    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if (
        len(sigma) != 2
        or any(len(row) != 2 for row in sigma)
    ):
        raise ValueError("sigma must be a 2x2 matrix.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")

    x_i = all_agent_states[i].position
    v_i = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
    h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
    y_i_clf = all_fast_states[i].y_clf
    coupling_sum = 0.0
    for j, fast_j in all_fast_states.items():
        if j == i:
            continue
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        v_j = compute_V_i(x_i=x_j, target_center=target_center, sigma=sigma, epsilon=epsilon)
        h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
        coupling_sum += h * h_i * h_j * (y_i_clf - fast_j.y_clf)

    z_clf_old = state.z_clf
    rhs = -xi * z_clf_old - alpha_clf * coupling_sum
    return z_clf_old + (dt / tau) * rhs


def update_y_components(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    x_i: Vector2,
    all_agent_states: dict[int, AgentState],
    u_i: Vector2,
    dt: float,
    tau: float,
    gamma_1: float,
    d_0: float,
    d_c: float,
    epsilon_c: float = 0.5,
    gamma_2: float | None = None,
    eta: float | None = None,
    lambda_2: float | None = None,
    v2: dict[int, float] | None = None,
    gamma_3: float | None = None,
    target_center: Vector2 | None = None,
    sigma: list[list[float]] | None = None,
    epsilon: float | None = None,
    obstacles: list[CircularObstacle] | None = None,
    u_max: float | None = None,
    n_agents: int | None = None,
    N: int | None = None,
    **extra: Any,
) -> tuple[dict[int, EdgeDualState], float, float, list[float], list[float]]:
    """Update rule placeholder for stacked dual variable y_i.

    In the paper notation:
        y_i = col({y_i^{i,j}}_{j != i}, y_i^c, y_i^{clf})

    This function updates all three parts and returns:
    - y_pair: {j -> (y_i^{i,j}, y_j^{i,j}) as EdgeDualState}
    - y_c: y_i^c
    - y_clf: y_i^{clf}

    Implemented in this step:
    - pairwise self-component y_i^{i,j}
    - connectivity dual y_i^c
    - CLF dual y_i^{clf}

    Still unchanged:
    - pairwise neighbor-component y_j^{i,j}
    """
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if gamma_1 <= 0:
        raise ValueError(f"gamma_1 must be > 0, got {gamma_1}.")
    if d_0 <= 0:
        raise ValueError(f"d_0 must be > 0, got {d_0}.")
    if d_c <= 0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if n_agents is None:
        n_agents = N
    if n_agents is None:
        n_agents = len(all_fast_states)
    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}.")
    if eta is None:
        eta = 0.0
    if lambda_2 is None:
        raise ValueError("Runtime lambda_2 is required for connectivity y_c dynamics.")
    if v2 is None:
        raise ValueError("Runtime v2 is required for connectivity y_c dynamics.")
    if gamma_3 is None:
        gamma_3 = 1.0  # TODO: thread paper gamma_3 explicitly from config.
    if gamma_2 is None:
        gamma_2 = gamma_1
    if target_center is None:
        target_center = (0.0, 0.0)
    if sigma is None:
        sigma = [[1.0, 0.0], [0.0, 1.0]]
    if epsilon is None:
        epsilon = 1e-2
    if obstacles is None:
        obstacles = []
    if len(state.y_obs) != len(obstacles):
        raise ValueError(
            f"state.y_obs must have one scalar per obstacle: "
            f"len(y_obs)={len(state.y_obs)} != len(obstacles)={len(obstacles)}"
        )
    if u_max is None:
        raise ValueError("u_max is required for actuator fast-dual dynamics.")
    if len(state.y_u) != 4:
        raise ValueError(f"state.y_u must have length 4, got {len(state.y_u)}.")
    step_idx = extra.get("step_idx")
    log_pairwise_debug = isinstance(step_idx, int) and step_idx in {0, 1, 2, 3}
    log_clf_debug = isinstance(step_idx, int) and step_idx in {0, 1, 2}
    log_agent2_step3 = step_idx == 3 and i == 2

    updated_y_pair: dict[int, EdgeDualState] = {}
    for j, y_edge in state.y_pair.items():
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        if j not in state.z_pair:
            raise KeyError(f"Missing z-pair entry for edge (i={i}, j={j}).")
        if j not in all_fast_states:
            raise KeyError(f"Missing fast state for j={j}.")
        if i not in all_fast_states[j].z_pair:
            raise KeyError(f"Missing mirrored z-pair entry for edge (j={j}, i={i}).")

        x_j = all_agent_states[j].position
        z_edge = state.z_pair[j]
        z_j_ij_live = all_fast_states[j].z_pair[i].self_component
        dx = x_i[0] - x_j[0]
        dy = x_i[1] - x_j[1]
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        dist_sq = dx * dx + dy * dy
        velocity_term = -2.0 * (dx * state.w[0] + dy * state.w[1])
        barrier_term = -0.5 * gamma_1 * (dist_sq - d_0 * d_0)
        mismatch_term = z_edge.self_component - z_j_ij_live
        g_ij = pairwise_constraint_value(
            x_i=x_i,
            x_j=x_j,
            w_i=state.w,
            z_i_ij=z_edge.self_component,
            z_j_ij=z_j_ij_live,
            gamma_1=gamma_1,
            d_0=d_0,
            d_c=d_c,
        )
        if log_pairwise_debug:
            print(
                f"[DEBUG pairwise g] step={step_idx}, agent={i}, neighbor={j}, "
                f"z_i_ij={z_edge.self_component:.12f}, y_i_ij={y_edge.self_component:.12f}, "
                f"z_j_ij_live={z_j_ij_live:.12f}, w_i=({state.w[0]:.12f}, {state.w[1]:.12f}), "
                f"velocity_term={velocity_term:.12f}, barrier_term={barrier_term:.12f}, "
                f"mismatch_term={mismatch_term:.12f}, h_ij={h:.12f}, g_ij={g_ij:.12f}"
            )
        if log_agent2_step3:
            print(
                f"[DEBUG STEP3 Y] pair (2,{j}): "
                f"w_2=({state.w[0]:.12f}, {state.w[1]:.12f}), "
                f"z_i={z_edge.self_component:.12f}, z_j={z_j_ij_live:.12f}, "
                f"z_diff={z_edge.self_component - z_j_ij_live:.12f}, "
                f"y_i={y_edge.self_component:.12f}, "
                f"y_j={all_fast_states[j].y_pair[i].self_component:.12f}, "
                f"g_i={g_ij:.12f}"
            )

        y_old = y_edge.self_component
        projected_rhs = project_plus(a=g_ij, b=y_old)
        y_new = y_old + (dt / tau) * projected_rhs
        y_new = max(0.0, y_new)

        updated_y_pair[j] = EdgeDualState(
            self_component=y_new,
            neighbor_component=y_edge.neighbor_component,
        )

    g_c = connectivity_constraint_value(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        x_i=x_i,
        all_agent_states=all_agent_states,
        w_i=state.w,
        eta=eta,
        lambda_2=lambda_2,
        d_c=d_c,
        epsilon_c=epsilon_c,
        v2=v2,
        n_agents=n_agents,
    )
    y_c_old = state.y_c
    projected_rhs_c = project_plus(a=g_c, b=y_c_old)
    y_c_new = y_c_old + (dt / tau) * projected_rhs_c
    y_c_new = max(0.0, y_c_new)

    v_i_clf = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
    h_i_clf = math.exp(-1.0 / v_i_clf) if v_i_clf > 0.0 else 0.0
    exp_factor_clf = math.exp(-1.0 / v_i_clf) if v_i_clf > 0.0 else 0.0
    w_i_scalar_clf = exp_factor_clf * v_i_clf if v_i_clf > 0.0 else 0.0
    gamma_w_term_clf = gamma_3 * w_i_scalar_clf
    s00, s01 = sigma[0]
    s10, s11 = sigma[1]
    sigma_w0 = s00 * state.w[0] + s01 * state.w[1]
    sigma_w1 = s10 * state.w[0] + s11 * state.w[1]
    dx0 = x_i[0] - target_center[0]
    dx1 = x_i[1] - target_center[1]
    x_sigma_w = dx0 * sigma_w0 + dx1 * sigma_w1
    clf_drift_term = (
        2.0 * exp_factor_clf * (1.0 + 1.0 / v_i_clf) * x_sigma_w
        if v_i_clf > 0.0
        else 0.0
    )
    clf_coupling_sum = 0.0
    clf_coupling_contributions: list[tuple[int, float, float, float, float]] = []
    if h_i_clf != 0.0:
        for j, fast_j in all_fast_states.items():
            if j == i:
                continue
            if j not in all_agent_states:
                raise KeyError(f"Missing agent state for j={j}.")
            x_j = all_agent_states[j].position
            h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
            v_j = compute_V_i(x_i=x_j, target_center=target_center, sigma=sigma, epsilon=epsilon)
            h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
            contribution = h * h_i_clf * h_j * (state.z_clf - fast_j.z_clf)
            clf_coupling_sum += contribution
            clf_coupling_contributions.append((j, h, v_j, h_j, contribution))
    g_clf = clf_constraint_value(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        x_i=x_i,
        all_agent_states=all_agent_states,
        w_i=state.w,
        gamma_3=gamma_3,
        d_c=d_c,
        target_center=target_center,
        sigma=sigma,
        epsilon=epsilon,
    )
    y_clf_old = state.y_clf
    projected_rhs_clf = project_plus(a=g_clf, b=y_clf_old)
    y_clf_new = y_clf_old + (dt / tau) * projected_rhs_clf
    y_clf_new = max(0.0, y_clf_new)
    if log_clf_debug:
        print(f"[DEBUG clf g] step={step_idx}, agent={i}")
        print(f"  z_clf={state.z_clf:.12f}")
        print(f"  y_clf={state.y_clf:.12f}")
        print(f"  V_i={v_i_clf:.12f}")
        print(f"  W_i={w_i_scalar_clf:.12f}")
        print(f"  gamma3W={gamma_w_term_clf:.12f}")
        print(f"  clf_drift={clf_drift_term:.12f}")
        for j, h, v_j, h_j, contribution in clf_coupling_contributions:
            print(
                f"  neighbor={j}, h_ij={h:.12f}, V_j={v_j:.12f}, h_j={h_j:.12f}, "
                f"clf_coupling_contribution={contribution:.12f}"
            )
        print(f"  clf_coupling_sum={clf_coupling_sum:.12f}")
        print(f"  g_clf={g_clf:.12f}")
        print(f"  projected_rhs_clf={projected_rhs_clf:.12f}")
        print(f"  y_clf_new={y_clf_new:.12f}")

    y_obs_new: list[float] = []
    for y_obs_old, obstacle in zip(state.y_obs, obstacles):
        g_obs = obstacle_constraint_value(
            agent_position=x_i,
            control_input=state.w,
            obstacle_center=obstacle.center,
            obstacle_radius=obstacle.radius,
            gamma_2=gamma_2,
        )
        projected_rhs_obs = project_plus(a=g_obs, b=y_obs_old)
        y_obs_updated = y_obs_old + (dt / tau) * projected_rhs_obs
        y_obs_new.append(max(0.0, y_obs_updated))

    actuator_rows = [
        state.w[0] - u_max,
        state.w[1] - u_max,
        -state.w[0] - u_max,
        -state.w[1] - u_max,
    ]
    y_u_new: list[float] = []
    for y_u_old, g_u in zip(state.y_u, actuator_rows):
        projected_rhs_u = project_plus(a=g_u, b=y_u_old)
        y_u_updated = y_u_old + (dt / tau) * projected_rhs_u
        y_u_new.append(max(0.0, y_u_updated))

    return updated_y_pair, y_c_new, y_clf_new, y_obs_new, y_u_new


def update_fast_state(state: FastState, **kwargs: Any) -> FastState:
    """Orchestrate one fast-state update for a single agent.

    This calls the component update functions in a clear order and returns a new
    FastState. Mathematical update formulas are deferred to a later step.

    The fast-state update uses synchronous explicit Euler from the old snapshot
    provided in ``state``/``all_fast_states``. In particular, ``update_y_components``
    intentionally uses ``state.w`` rather than ``w_new``.
    """

    w_new = update_w(state=state, **kwargs)
    z_pair_new = update_z_pair(state=state, **kwargs)
    z_c_new = update_z_c(state=state, **kwargs)
    z_clf_new = update_z_clf(state=state, **kwargs)
    y_pair_new, y_c_new, y_clf_new, y_obs_new, y_u_new = update_y_components(state=state, **kwargs)

    return FastState(
        w=w_new,
        z_pair=z_pair_new,
        z_c=z_c_new,
        z_clf=z_clf_new,
        y_pair=y_pair_new,
        y_c=y_c_new,
        y_clf=y_clf_new,
        y_obs=y_obs_new,
        y_u=y_u_new,
    )

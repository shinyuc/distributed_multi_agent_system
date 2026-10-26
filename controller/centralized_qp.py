"""Centralized augmented QP assembly and solve helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from controller.constraints import obstacle_constraint_coefficients
from controller.fast_dynamics import clf_coupling_gates, compute_V_i, compute_beta_i, h_ij
from models.agent import AgentState, Vector2
from models.environment import CircularObstacle
from models.fast_state import FastState


@dataclass
class CentralizedQPData:
    """Container for the stacked centralized QP."""

    P: list[list[float]]
    q: list[float]
    A: list[list[float]]
    b: list[float]
    row_labels: list[str]
    row_blocks: list[str]
    decision_labels: list[str]
    control_dim: int
    z_dim: int
    block_counts: dict[str, int]


def _append_joint_row(
    A: list[list[float]],
    b: list[float],
    row_labels: list[str],
    row_blocks: list[str],
    a_row: np.ndarray,
    b_val: float,
    label: str,
    block: str,
) -> None:
    a = np.asarray(a_row, dtype=float)
    b_scalar = float(b_val)
    a_norm = float(np.linalg.norm(a))
    scale = max(a_norm, abs(b_scalar), 1e-8)
    A.append((a / scale).tolist())
    b.append(b_scalar / scale)
    row_labels.append(label)
    row_blocks.append(block)


def _ordered_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def build_centralized_qp_data(
    *,
    all_agent_states: dict[int, AgentState],
    all_fast_states: dict[int, FastState],
    obstacles: list[CircularObstacle],
    u_max: float,
    xi: float,
    gamma_1: float,
    d_0: float,
    gamma_2: float,
    eta: float,
    d_c: float,
    epsilon_c: float,
    gamma_3: float,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
    enable_connectivity_constraint: bool,
    alpha_clf: float,
    alpha_connectivity: float,
    lambda_2: float | None,
    v2: dict[int, float] | None,
    clf_coupling: str = "gated",
) -> CentralizedQPData:
    """Build the centralized QP over stacked controls and mismatch variables.

    CLF mismatch weights follow ``clf_coupling`` (see controller.fast_dynamics.clf_coupling_gates);
    in the communication mode an agent inside its inner target region keeps its z_clf coupling.
    """
    clf_coupling_gates(0.0, 0.0, clf_coupling)  # validate the mode
    agent_ids = sorted(all_agent_states.keys())
    n_agents = len(agent_ids)
    if n_agents == 0:
        raise ValueError("At least one agent is required.")
    if xi < 0.0:
        raise ValueError(f"xi must be >= 0, got {xi}.")
    if enable_connectivity_constraint and lambda_2 is None:
        raise ValueError("Runtime lambda_2 is required when connectivity is enabled.")
    if enable_connectivity_constraint and v2 is None:
        raise ValueError("Runtime v2 is required when connectivity is enabled.")

    control_dim = 2 * n_agents
    decision_labels: list[str] = []
    u_index: dict[int, tuple[int, int]] = {}
    for offset, i in enumerate(agent_ids):
        base = 2 * offset
        u_index[i] = (base, base + 1)
        decision_labels.extend([f"u[{i}].x", f"u[{i}].y"])

    pair_index: dict[tuple[int, int], tuple[int, int]] = {}
    next_idx = control_dim
    for a, i in enumerate(agent_ids):
        for j in agent_ids[a + 1:]:
            idx_i = next_idx
            idx_j = next_idx + 1
            pair_index[(i, j)] = (idx_i, idx_j)
            decision_labels.extend([f"z_pair[{i},{j}].{i}", f"z_pair[{i},{j}].{j}"])
            next_idx += 2

    z_c_index: dict[int, int] = {}
    for i in agent_ids:
        z_c_index[i] = next_idx
        decision_labels.append(f"z_c[{i}]")
        next_idx += 1

    z_clf_index: dict[int, int] = {}
    for i in agent_ids:
        z_clf_index[i] = next_idx
        decision_labels.append(f"z_clf[{i}]")
        next_idx += 1

    n_dec = next_idx
    z_dim = n_dec - control_dim
    P = np.zeros((n_dec, n_dec), dtype=float)
    for idx in range(control_dim):
        P[idx, idx] = 1.0
    for idx in range(control_dim, n_dec):
        P[idx, idx] = xi
    q = np.zeros((n_dec,), dtype=float)

    A: list[list[float]] = []
    b: list[float] = []
    row_labels: list[str] = []
    row_blocks: list[str] = []

    for a, i in enumerate(agent_ids):
        x_i = all_agent_states[i].position
        for j in agent_ids[a + 1:]:
            x_j = all_agent_states[j].position
            dx = x_i[0] - x_j[0]
            dy = x_i[1] - x_j[1]
            h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
            dist_sq = dx * dx + dy * dy
            geom = h * (0.5 * gamma_1 * (dist_sq - d_0 * d_0))
            idx_zi, idx_zj = pair_index[(i, j)]

            row_i = np.zeros((n_dec,), dtype=float)
            ux_i, uy_i = u_index[i]
            row_i[ux_i] = -2.0 * h * dx
            row_i[uy_i] = -2.0 * h * dy
            row_i[idx_zi] = h
            row_i[idx_zj] = -h
            _append_joint_row(
                A,
                b,
                row_labels,
                row_blocks,
                row_i,
                geom,
                f"pairwise(i={i},j={j})",
                "pairwise",
            )

            row_j = np.zeros((n_dec,), dtype=float)
            ux_j, uy_j = u_index[j]
            row_j[ux_j] = 2.0 * h * dx
            row_j[uy_j] = 2.0 * h * dy
            row_j[idx_zi] = -h
            row_j[idx_zj] = h
            _append_joint_row(
                A,
                b,
                row_labels,
                row_blocks,
                row_j,
                geom,
                f"pairwise(i={j},j={i})",
                "pairwise",
            )

    for i in agent_ids:
        x_i = all_agent_states[i].position
        ux_i, uy_i = u_index[i]
        for k, obstacle in enumerate(obstacles):
            a_row_2d, b_val = obstacle_constraint_coefficients(
                agent_position=x_i,
                obstacle_center=obstacle.center,
                obstacle_radius=obstacle.radius,
                gamma_2=gamma_2,
            )
            row = np.zeros((n_dec,), dtype=float)
            row[ux_i] = float(a_row_2d[0])
            row[uy_i] = float(a_row_2d[1])
            _append_joint_row(
                A,
                b,
                row_labels,
                row_blocks,
                row,
                b_val,
                f"obstacle(i={i},k={k})",
                "obstacle",
            )

    if enable_connectivity_constraint:
        for i in agent_ids:
            state_i = all_fast_states[i]
            beta_i = compute_beta_i(
                i=i,
                state=state_i,
                all_fast_states=all_fast_states,
                all_agent_states=all_agent_states,
                d_c=d_c,
                epsilon_c=epsilon_c,
                v2=v2,
            )
            row = np.zeros((n_dec,), dtype=float)
            ux_i, uy_i = u_index[i]
            row[ux_i] = -float(beta_i[0])
            row[uy_i] = -float(beta_i[1])
            x_i = all_agent_states[i].position
            for j in agent_ids:
                if j == i:
                    continue
                x_j = all_agent_states[j].position
                h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
                row[z_c_index[i]] += alpha_connectivity * h
                row[z_c_index[j]] -= alpha_connectivity * h
            rhs = float(lambda_2) / n_agents - eta / n_agents
            _append_joint_row(
                A,
                b,
                row_labels,
                row_blocks,
                row,
                rhs,
                f"connectivity(i={i})",
                "connectivity",
            )

    for i in agent_ids:
        row = np.zeros((n_dec,), dtype=float)
        x_i = all_agent_states[i].position
        ux_i, uy_i = u_index[i]
        v_i = compute_V_i(
            x_i=x_i,
            target_center=target_center,
            sigma=sigma,
            epsilon=epsilon,
        )
        h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
        if h_i > 0.0:
            exp_factor = math.exp(-1.0 / v_i)
            alpha = 2.0 * exp_factor * (1.0 + 1.0 / v_i)
            w_i = exp_factor * v_i
            dx0 = x_i[0] - target_center[0]
            dx1 = x_i[1] - target_center[1]
            row[ux_i] = alpha * (dx0 * sigma[0][0] + dx1 * sigma[1][0])
            row[uy_i] = alpha * (dx0 * sigma[0][1] + dx1 * sigma[1][1])
            for j in agent_ids:
                if j == i:
                    continue
                x_j = all_agent_states[j].position
                v_j = compute_V_i(
                    x_i=x_j,
                    target_center=target_center,
                    sigma=sigma,
                    epsilon=epsilon,
                )
                h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
                g_i, g_j = clf_coupling_gates(h_i, h_j, clf_coupling)
                coeff = alpha_clf * h_ij(x_i=x_i, x_j=x_j, d_c=d_c) * g_i * g_j
                row[z_clf_index[i]] += coeff
                row[z_clf_index[j]] -= coeff
            rhs = -gamma_3 * w_i
            label = f"clf(i={i},active)"
        else:
            if clf_coupling == "communication":
                # Agent inside its inner target region: W_i = 0, grad W_i = 0; the mismatch
                # coupling row sum_j h_ij (z_i^clf - z_j^clf) <= 0 is kept.
                for j in agent_ids:
                    if j == i:
                        continue
                    coeff = alpha_clf * h_ij(x_i=x_i, x_j=all_agent_states[j].position, d_c=d_c)
                    row[z_clf_index[i]] += coeff
                    row[z_clf_index[j]] -= coeff
            rhs = 0.0
            label = f"clf(i={i},inactive)"
        _append_joint_row(
            A,
            b,
            row_labels,
            row_blocks,
            row,
            rhs,
            label,
            "clf",
        )

    for i in agent_ids:
        ux_i, uy_i = u_index[i]
        rows = [
            ("bounds", f"actuator(i={i},ux<=umax)", ux_i, 1.0),
            ("bounds", f"actuator(i={i},-ux<=umax)", ux_i, -1.0),
            ("bounds", f"actuator(i={i},uy<=umax)", uy_i, 1.0),
            ("bounds", f"actuator(i={i},-uy<=umax)", uy_i, -1.0),
        ]
        for block, label, idx, coeff in rows:
            row = np.zeros((n_dec,), dtype=float)
            row[idx] = coeff
            _append_joint_row(A, b, row_labels, row_blocks, row, u_max, label, block)

    block_counts = dict(Counter(row_blocks))
    return CentralizedQPData(
        P=P.tolist(),
        q=q.tolist(),
        A=A,
        b=b,
        row_labels=row_labels,
        row_blocks=row_blocks,
        decision_labels=decision_labels,
        control_dim=control_dim,
        z_dim=z_dim,
        block_counts=block_counts,
    )


def solve_centralized_qp_with_status(
    qp_data: CentralizedQPData,
) -> tuple[np.ndarray, bool, str]:
    """Solve the centralized QP and return (decision, success, reason)."""
    try:
        import cvxpy as cp
    except ImportError:
        return np.zeros((len(qp_data.q),), dtype=float), False, "cvxpy import failed"

    try:
        P = np.asarray(qp_data.P, dtype=float)
        q = np.asarray(qp_data.q, dtype=float)
        A = np.asarray(qp_data.A, dtype=float)
        b = np.asarray(qp_data.b, dtype=float)
        n_dec = q.shape[0]
        if P.shape != (n_dec, n_dec):
            return np.zeros((n_dec,), dtype=float), False, f"invalid objective matrix shape P={P.shape}"
        if A.size == 0:
            A = np.zeros((0, n_dec), dtype=float)
            b = np.zeros((0,), dtype=float)
        if A.ndim != 2 or A.shape[1] != n_dec or b.ndim != 1 or A.shape[0] != b.shape[0]:
            return np.zeros((n_dec,), dtype=float), False, f"invalid inequality dimensions A={A.shape}, b={b.shape}"

        x = cp.Variable(n_dec)
        objective = 0.5 * cp.quad_form(x, P) + q @ x
        constraints: list[Any] = []
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, polish=False)
        if problem.status not in {"optimal", "optimal_inaccurate"} or x.value is None:
            return np.zeros((n_dec,), dtype=float), False, f"solver status={problem.status}"
        return np.asarray(x.value, dtype=float).reshape((-1,)), True, str(problem.status)
    except Exception as exc:
        return np.zeros((len(qp_data.q),), dtype=float), False, f"exception: {exc}"


def _feasibility_status(A_rows: list[list[float]], b_rows: list[float], n_dec: int) -> dict[str, object]:
    if not A_rows:
        return {"feasible": True, "status": "no_constraints"}
    try:
        import cvxpy as cp
    except ImportError:
        return {"feasible": None, "status": "cvxpy import failed"}

    try:
        A = np.asarray(A_rows, dtype=float)
        b = np.asarray(b_rows, dtype=float)
        x = cp.Variable(n_dec)
        objective = 1e-9 * cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(objective), [A @ x <= b])
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, polish=False)
        feasible = problem.status in {"optimal", "optimal_inaccurate"}
        return {"feasible": feasible, "status": str(problem.status)}
    except Exception as exc:
        return {"feasible": None, "status": f"exception: {exc}"}


def analyze_block_feasibility(qp_data: CentralizedQPData) -> dict[str, object]:
    """Run lightweight feasibility checks by constraint block for debugging."""
    order = ["bounds", "pairwise", "obstacle", "connectivity", "clf"]
    block_to_rows: dict[str, list[int]] = {name: [] for name in order}
    for idx, block in enumerate(qp_data.row_blocks):
        if block not in block_to_rows:
            block_to_rows[block] = []
        block_to_rows[block].append(idx)

    n_dec = len(qp_data.q)
    individual: dict[str, object] = {}
    for block in order:
        rows = block_to_rows.get(block, [])
        A_rows = [qp_data.A[idx] for idx in rows]
        b_rows = [qp_data.b[idx] for idx in rows]
        individual[block] = _feasibility_status(A_rows, b_rows, n_dec)

    cumulative: dict[str, object] = {}
    active_rows: list[int] = []
    first_infeasible_block: str | None = None
    for block in order:
        active_rows.extend(block_to_rows.get(block, []))
        A_rows = [qp_data.A[idx] for idx in active_rows]
        b_rows = [qp_data.b[idx] for idx in active_rows]
        status = _feasibility_status(A_rows, b_rows, n_dec)
        cumulative[block] = status
        if first_infeasible_block is None and status.get("feasible") is False:
            first_infeasible_block = block

    return {
        "individual": individual,
        "cumulative": cumulative,
        "first_infeasible_block": first_infeasible_block,
    }

"""Local QP data construction for one agent.

This module builds the per-agent quadratic objective and affine inequality
constraints in the form:

    minimize    1/2 u_i^T P u_i + q^T u_i
    subject to  A u_i <= b
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from controller.constraints import obstacle_constraint_coefficients
from controller.fast_dynamics import (
    compute_V_i,
    compute_beta_i,
    h_ij,
)
from models.agent import AgentState, Vector2
from models.environment import CircularObstacle
from models.fast_state import FastState


@dataclass
class LocalQPData:
    """Container for one-agent QP objective and inequality constraints."""

    P: list[list[float]]
    q: list[float]
    A: list[list[float]]
    b: list[float]
    row_labels: list[str]
    # Unscaled rows (a^T u <= b before the positive row scaling in _append_row).
    # Kept for diagnostics only; the solver uses A/b above.
    A_raw: list[list[float]] = field(default_factory=list)
    b_raw: list[float] = field(default_factory=list)


@dataclass
class LocalQPSolveResult:
    """Outcome of one local QP solve, including the unmodified solver status."""

    u: tuple[float, float]
    success: bool
    reason: str
    status: str | None
    num_iters: int | None = None
    solve_time: float | None = None
    wall_time: float | None = None
    pri_res: float | None = None
    dua_res: float | None = None


def solve_local_qp(qp_data: LocalQPData) -> tuple[float, float]:
    """Solve local QP with OSQP and return (u_x, u_y)."""
    u, _, _ = solve_local_qp_with_status(qp_data)
    return u


def solve_local_qp_with_status(
    qp_data: LocalQPData,
) -> tuple[tuple[float, float], bool, str]:
    """Solve local QP and return (u, success, reason)."""
    result = solve_local_qp_detailed(qp_data)
    return result.u, result.success, result.reason


# The local QP used to be solved through cvxpy (solver=OSQP); it is now passed to OSQP
# directly, reproducing what that call did so that results and reported statuses are
# unchanged:
#   - same OSQP data: P, q, A (csc), l = -inf, u = b;
#   - the settings that were in effect: cvxpy's defaults eps_abs = eps_rel = 1e-5 and
#     max_iter = 10000, verbose off, and polishing off (cvxpy forwarded polish=False,
#     which OSQP >= 1 maps onto "polishing"); OSQP's warm_starting default is left as is
#     and has no effect on a freshly set-up solver;
#   - cvxpy's status names (cvxpy OSQP.STATUS_MAP for OSQP >= 1, keyed by status_val);
#   - cvxpy's reporting: iteration count only when a solution is present, solve time =
#     OSQP run_time, and the same exception messages for solver errors and NaN data.
_OSQP_SETTINGS = dict(eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, verbose=False, polishing=False)
# OSQP otherwise picks the first importable backend among cuda, mkl, builtin (or the one named
# by the OSQP_ALGEBRA_BACKEND environment variable), so installing osqp-cuda / osqp-mkl would
# silently change the linear solver. The builtin CPU backend is the one all results so far used.
_OSQP_ALGEBRA = "builtin"
_OSQP_STATUS = {
    1: "optimal",
    2: "optimal_inaccurate",
    3: "infeasible",
    4: "infeasible_inaccurate",
    5: "unbounded",
    6: "unbounded_inaccurate",
    7: "user_limit",
    8: "user_limit",
    10: "solver_error",
    11: "solver_error",
}
_OSQP_SOLUTION_PRESENT = {"optimal", "optimal_inaccurate", "user_limit"}
_OSQP_FAILED_MESSAGE = (
    "Solver 'OSQP' failed. Try another solver, or solve with verbose=True for more information."
)


def local_qp_solver_metadata() -> dict[str, Any]:
    """Describe the local QP solve (solver, version, backend, effective settings) for run metadata.

    Backend, linear solver and precision are read from an OSQP instance set up exactly as in
    solve_local_qp_detailed, so they report what is actually used.
    """
    meta: dict[str, Any] = {
        "solver": "OSQP, called directly by controller.local_qp.solve_local_qp_detailed (no cvxpy)",
        "osqp_version": None,
        "setup": "fresh OSQP instance per QP; no warm start from previous solves",
        "settings": dict(_OSQP_SETTINGS),
        "requested_algebra": _OSQP_ALGEBRA,
        "algebra": None,
        "linear_solver": None,
        "float_precision": None,
    }
    try:
        import osqp
        import scipy.sparse as sp

        meta["osqp_version"] = osqp.__version__
        probe = osqp.OSQP(algebra=_OSQP_ALGEBRA)
        probe.setup(sp.csc_matrix(np.eye(2)), np.zeros(2), sp.csc_matrix(np.eye(2)), np.full(2, -np.inf),
                    np.ones(2), **_OSQP_SETTINGS)
        meta["algebra"] = probe.algebra
        meta["linear_solver"] = f"{probe.solver_type} ({probe.settings.linsys_solver.name})"
        meta["float_precision"] = "float32" if probe.ext.OSQP_USE_FLOAT == 1 else "float64"
    except Exception as exc:  # metadata must never break a run
        meta["probe_error"] = f"{type(exc).__name__}: {exc}"
    return meta


def _osqp_residuals(results: Any) -> tuple[float | None, float | None]:
    info = getattr(results, "info", None)
    if info is None:
        return None, None
    pri = getattr(info, "pri_res", getattr(info, "prim_res", None))
    dua = getattr(info, "dua_res", getattr(info, "dual_res", None))
    return (float(pri) if pri is not None else None, float(dua) if dua is not None else None)


def solve_local_qp_detailed(qp_data: LocalQPData) -> LocalQPSolveResult:
    """Solve local QP and return the solution with the raw solver status."""
    # Fallback policy: if assembly/solve fails or problem is not solved reliably,
    # return (0.0, 0.0) so control remains well-defined and simulation can continue.
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError:
        # WARNING: Fallback path if OSQP is unavailable.
        return LocalQPSolveResult((0.0, 0.0), False, "osqp import failed", None)

    try:
        P = np.asarray(qp_data.P, dtype=float)
        q = np.asarray(qp_data.q, dtype=float)
        A = np.asarray(qp_data.A, dtype=float)
        b = np.asarray(qp_data.b, dtype=float)

        if P.shape != (2, 2) or q.shape != (2,):
            # WARNING: Fallback path if QP dimensions are invalid.
            return LocalQPSolveResult(
                (0.0, 0.0), False, f"invalid objective dimensions P={P.shape}, q={q.shape}", None
            )
        if A.size == 0:
            A = np.zeros((0, 2), dtype=float)
            b = np.zeros((0,), dtype=float)
        if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            # WARNING: Fallback path if inequality dimensions are invalid.
            return LocalQPSolveResult(
                (0.0, 0.0),
                False,
                f"invalid inequality dimensions A={A.shape}, b={b.shape}",
                None,
            )

        # Same data checks (and messages) cvxpy applied before calling OSQP.
        if not np.allclose(P, P.T):
            raise ValueError("Quadratic form matrices must be symmetric/Hermitian.")
        if not (np.isfinite(P).all() and np.isfinite(q).all() and np.isfinite(A).all()):
            raise ValueError("Problem data contains NaN or Inf. Check your parameter values and constants.")
        if np.isnan(b).any():
            raise ValueError("Problem data contains NaN. Check your parameter values and constants.")

        t0 = time.perf_counter()
        solver = osqp.OSQP(algebra=_OSQP_ALGEBRA)
        solver.setup(sp.csc_matrix(P), q, sp.csc_matrix(A), np.full(b.shape, -np.inf), b, **_OSQP_SETTINGS)
        results = solver.solve(raise_error=False)
        wall = time.perf_counter() - t0
        status = _OSQP_STATUS.get(results.info.status_val, "solver_error")
        if status == "solver_error":
            raise RuntimeError(_OSQP_FAILED_MESSAGE)
        num_iters = results.info.iter if status in _OSQP_SOLUTION_PRESENT else None
        solve_time = results.info.run_time
        pri_res, dua_res = _osqp_residuals(results)
        details = dict(
            status=status,
            num_iters=int(num_iters) if num_iters is not None else None,
            solve_time=float(solve_time) if solve_time is not None else None,
            wall_time=wall,
            pri_res=pri_res,
            dua_res=dua_res,
        )

        if status not in {"optimal", "optimal_inaccurate"} or results.x is None:
            # WARNING: Fallback path if QP is infeasible or solver failed.
            return LocalQPSolveResult((0.0, 0.0), False, f"solver status={status}", **details)

        return LocalQPSolveResult((float(results.x[0]), float(results.x[1])), True, status, **details)
    except Exception as exc:
        # WARNING: Fallback path for any runtime solver/assembly error.
        return LocalQPSolveResult((0.0, 0.0), False, f"exception: {exc}", "exception")


def _append_row(
    A: list[list[float]],
    b: list[float],
    row_labels: list[str],
    a_row: Vector2,
    b_val: float,
    label: str,
    A_raw: list[list[float]] | None = None,
    b_raw: list[float] | None = None,
) -> None:
    # Numerically equivalent positive row scaling: (a^T u <= b) -> ((a/s)^T u <= b/s)
    # with s = max(||a||_2, |b|, 1e-8).
    a0 = float(a_row[0])
    a1 = float(a_row[1])
    b_scalar = float(b_val)
    a_norm = math.sqrt(a0 * a0 + a1 * a1)
    s = max(a_norm, abs(b_scalar), 1e-8)

    A.append([a0 / s, a1 / s])
    b.append(b_scalar / s)
    row_labels.append(label)
    if A_raw is not None and b_raw is not None:
        A_raw.append([a0, a1])
        b_raw.append(b_scalar)


def build_pairwise_constraint_row(
    i: int,
    j: int,
    x_i: Vector2,
    x_j: Vector2,
    z_i_ij: float,
    z_j_ij: float,
    gamma_1: float,
    d_0: float,
    d_c: float,
    y_i_ij: float | None = None,
    y_j_ij: float | None = None,
    step_idx: int | None = None,
    w_i: Vector2 | None = None,
) -> tuple[Vector2, float, str]:
    """Build affine row for pairwise safety constraint g_i^{i,j}(u_i) <= 0."""
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
    a: Vector2 = (-2.0 * h * dx, -2.0 * h * dy)
    b_val = h * (0.5 * gamma_1 * (dist_sq - d_0 * d_0) - z_i_ij + z_j_ij)
    if step_idx == 3 and i == 2 and w_i is not None:
        g_ij = h * (
            -2.0 * (dx * w_i[0] + dy * w_i[1])
            - 0.5 * gamma_1 * (dist_sq - d_0 * d_0)
            + z_i_ij
            - z_j_ij
        )
        print(
            f"[DEBUG STEP3 QP] pair (2,{j}): "
            f"w_2=({w_i[0]:.12f}, {w_i[1]:.12f}), "
            f"z_i={z_i_ij:.12f}, z_j={z_j_ij:.12f}, "
            f"z_diff={z_i_ij - z_j_ij:.12f}, "
            f"y_i={float(y_i_ij or 0.0):.12f}, "
            f"y_j={float(y_j_ij or 0.0):.12f}, "
            f"g_i={g_ij:.12f}, "
            f"a=({a[0]:.12f}, {a[1]:.12f}), b={b_val:.12f}"
        )
    return a, b_val, f"pairwise(i={i},j={j})"


def build_obstacle_constraint_row(
    i: int,
    k: int,
    x_i: Vector2,
    obstacle: CircularObstacle,
    gamma_2: float,
) -> tuple[Vector2, float, str]:
    """Build affine row for obstacle constraint g_obs^{i,k}(u_i) <= 0."""
    a, b_val = obstacle_constraint_coefficients(
        agent_position=x_i,
        obstacle_center=obstacle.center,
        obstacle_radius=obstacle.radius,
        gamma_2=gamma_2,
    )
    return a, b_val, f"obstacle(i={i},k={k})"


def build_connectivity_constraint_row(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    eta: float,
    lambda_2: float,
    alpha_connectivity: float,
    d_c: float,
    epsilon_c: float,
    v2: dict[int, float] | None,
    n_agents: int,
) -> tuple[Vector2, float, str]:
    """Build affine row for connectivity constraint g_i^c(u_i) <= 0."""
    if d_c <= 0.0:
        raise ValueError(f"d_c must be > 0, got {d_c}.")
    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}.")
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")

    x_i = all_agent_states[i].position
    beta_i = compute_beta_i(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        all_agent_states=all_agent_states,
        d_c=d_c,
        epsilon_c=epsilon_c,
        v2=v2,
    )

    coupling_sum = 0.0
    for j, fast_j in all_fast_states.items():
        if j == i:
            continue
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        x_j = all_agent_states[j].position
        h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
        coupling_sum += h * (state.z_c - fast_j.z_c)

    # beta_i is a row vector: g_i^c(u_i) = -beta_i u_i + const <= 0.
    # Equivalent affine form: (-beta_i) u_i <= -const.
    coupling_sum_effective = alpha_connectivity * coupling_sum
    const_term = -lambda_2 / n_agents + eta / n_agents + coupling_sum_effective
    a: Vector2 = (-beta_i[0], -beta_i[1])
    b_val = -const_term
    print(
        f"[DEBUG connectivity row] agent={i}, label=connectivity(i={i}), "
        f"raw_connectivity_coupling_sum={coupling_sum:.6e}, "
        f"alpha_connectivity={alpha_connectivity:.6e}, "
        f"scaled_connectivity_coupling_sum={coupling_sum_effective:.6e}, "
        f"a_row=({a[0]:.6e}, {a[1]:.6e}), b_val={b_val:.6e}"
    )
    return a, b_val, f"connectivity(i={i})"


def build_clf_constraint_row(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    gamma_3: float,
    d_c: float,
    target_center: Vector2,
    sigma: list[list[float]],
    epsilon: float,
    alpha_clf: float = 1.0,
    step_idx: int | None = None,
    debug_disable_coupling: bool = False,
) -> tuple[Vector2, float, str]:
    """Build affine row for CLF constraint g_i^{clf}(u_i) <= 0."""
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
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")

    debug_on = (step_idx is not None) and (step_idx >= 100) and (i in {0, 1, 3})
    x_i = all_agent_states[i].position
    v_i = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
    h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
    if h_i == 0.0:
        a: Vector2 = (0.0, 0.0)
        b_val = 0.0
        if debug_on:
            # print(
            #     f"[DEBUG clf decomposition] step={step_idx}, agent={i}, "
            #     f"a=({a[0]:.6f}, {a[1]:.6f}), b_local={0.0:.6f}, "
            #     f"coupling={0.0:.6f}, b_val={b_val:.6f}"
            # )
            # print(
            #     f"[DEBUG clf terms] step={step_idx}, agent={i}, "
            #     f"V={v_i:.6f}, W={0.0:.6f}, gammaW={0.0:.6f}, "
            #     f"coupling={0.0:.6f}, b_val={b_val:.6f}"
            # )
            print(
                f"[DEBUG clf decomposition] step={step_idx}, agent={i}, "
                f"gamma3W={0.0:.6e}, alpha_clf={alpha_clf:.6e}, "
                f"raw_clf_coupling_sum={0.0:.6e}, scaled_clf_coupling_sum={0.0:.6e}, "
                f"final_A=({a[0]:.6e}, {a[1]:.6e}), final_b={b_val:.6e}"
            )
            print(
                f"[DEBUG clf terms] step={step_idx}, agent={i}, "
                f"V={v_i:.6e}, W={0.0:.6e}, gammaW={0.0:.6e}, "
                f"coupling={0.0:.6e}, b_val={b_val:.6e}"
            )
        return a, b_val, f"clf(i={i},inactive)"

    z_i_clf = state.z_clf
    if debug_on:
        print(f"[DEBUG clf coupling] step={step_idx}, agent={i}, z_i_clf={z_i_clf:.6f}")

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
        contribution = h * h_i * h_j * (z_i_clf - fast_j.z_clf)
        coupling_sum += contribution
        if debug_on:
            dxj0 = x_j[0] - target_center[0]
            dxj1 = x_j[1] - target_center[1]
            sigma_dxj0 = sigma[0][0] * dxj0 + sigma[0][1] * dxj1
            sigma_dxj1 = sigma[1][0] * dxj0 + sigma[1][1] * dxj1
            phi_j = dxj0 * sigma_dxj0 + dxj1 * sigma_dxj1 - 1.0
            inside = phi_j <= 0.0
            print(
                f"[DEBUG clf coupling term] step={step_idx}, agent={i}, neighbor={j}, "
                f"z_j_clf={fast_j.z_clf:.6f}, h_ij={h:.6f}, contribution={contribution:.6f}, "
                f"neighbor_inside_target={inside}"
            )
    if debug_on:
        print(f"[DEBUG clf coupling total] step={step_idx}, agent={i}, total={coupling_sum:.6f}")

    exp_factor = math.exp(-1.0 / v_i)
    alpha = 2.0 * exp_factor * (1.0 + 1.0 / v_i)
    w_i = exp_factor * v_i

    # (x_i-c)^T Sigma u_i = ((x_i-c)^T Sigma) u_i
    dx0 = x_i[0] - target_center[0]
    dx1 = x_i[1] - target_center[1]
    a_u0 = dx0 * sigma[0][0] + dx1 * sigma[1][0]
    a_u1 = dx0 * sigma[0][1] + dx1 * sigma[1][1]
    a = (alpha * a_u0, alpha * a_u1)

    b_local = gamma_3 * w_i
    coupling_sum_effective = alpha_clf * coupling_sum
    const_term = b_local + coupling_sum_effective
    b_val = -const_term
    if debug_on:
        # print(
        #     f"[DEBUG clf decomposition] step={step_idx}, agent={i}, "
        #     f"a=({a[0]:.6f}, {a[1]:.6f}), b_local={b_local:.6f}, "
        #     f"coupling={coupling_sum:.6f}, b_val={b_val:.6f}"
        # )
        # print(
        #     f"[DEBUG clf terms] step={step_idx}, agent={i}, "
        #     f"V={v_i:.6f}, W={w_i:.6f}, gammaW={b_local:.6f}, "
        #     f"coupling={coupling_sum:.6f}, b_val={b_val:.6f}"
        # )
        print(
            f"[DEBUG clf decomposition] step={step_idx}, agent={i}, "
            f"gamma3W={b_local:.6e}, "
            f"raw_clf_coupling_sum={coupling_sum:.6e}, "
            f"alpha_clf={alpha_clf:.6e}, "
            f"scaled_clf_coupling_sum={coupling_sum_effective:.6e}, "
            f"final_A=({a[0]:.6e}, {a[1]:.6e}), "
            f"final_b={b_val:.6e}"
        )
        print(
            f"[DEBUG clf terms] step={step_idx}, agent={i}, "
            f"V={v_i:.6e}, W={w_i:.6e}, gammaW={b_local:.6e}, "
            f"coupling={coupling_sum:.6e}, coupling_effective={coupling_sum_effective:.6e}, b_val={b_val:.6e}"
        )
    return a, b_val, f"clf(i={i},active)"


def build_actuator_box_rows(u_max: float) -> list[tuple[Vector2, float, str]]:
    """Build rows for box constraint |u_x| <= u_max, |u_y| <= u_max."""
    if u_max < 0.0:
        raise ValueError(f"u_max must be >= 0, got {u_max}.")
    return [
        ((1.0, 0.0), u_max, "actuator(ux<=umax)"),
        ((-1.0, 0.0), u_max, "actuator(-ux<=umax)"),
        ((0.0, 1.0), u_max, "actuator(uy<=umax)"),
        ((0.0, -1.0), u_max, "actuator(-uy<=umax)"),
    ]


def build_local_qp_data(
    i: int,
    state: FastState,
    all_fast_states: dict[int, FastState],
    all_agent_states: dict[int, AgentState],
    obstacles: list[CircularObstacle],
    u_max: float,
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
    step_idx: int | None = None,
    enable_connectivity_constraint: bool = True,
    alpha_clf: float = 1.0,
    alpha_connectivity: float = 1.0,
    debug_disable_clf_coupling: bool = False,
    lambda_2: float | None = None,
    v2: dict[int, float] | None = None,
    n_agents: int | None = None,
) -> LocalQPData:
    """Build local QP data for one agent i using current fast mismatch states."""
    # ###### check 
    # print("[DEBUG build_local_qp_data] enable_connectivity_constraint =", 
    #   enable_connectivity_constraint)
    # ######
    if i not in all_agent_states:
        raise KeyError(f"Missing agent state for i={i}.")
    if i not in all_fast_states:
        raise KeyError(f"Missing fast state for i={i}.")
    if n_agents is None:
        n_agents = len(all_fast_states)
    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}.")
    if enable_connectivity_constraint and lambda_2 is None:
        raise ValueError("Runtime lambda_2 is required when connectivity constraint is enabled.")
    if enable_connectivity_constraint and v2 is None:
        raise ValueError("Runtime v2 is required when connectivity constraint is enabled.")

    x_i = all_agent_states[i].position
    P = [[1.0, 0.0], [0.0, 1.0]]
    q = [0.0, 0.0]
    A: list[list[float]] = []
    b: list[float] = []
    row_labels: list[str] = []
    A_raw: list[list[float]] = []
    b_raw: list[float] = []

    for j, z_edge in state.z_pair.items():
        if j not in all_agent_states:
            raise KeyError(f"Missing agent state for j={j}.")
        if j not in all_fast_states:
            raise KeyError(f"Missing fast state for j={j}.")
        if i not in all_fast_states[j].z_pair:
            raise KeyError(f"Missing mirrored z-pair entry for edge (j={j}, i={i}).")
        x_j = all_agent_states[j].position
        z_j_ij_live = all_fast_states[j].z_pair[i].self_component
        # if step_idx == 4:
        #     print("check")
        a_row, b_val, label = build_pairwise_constraint_row(
            i=i,
            j=j,
            x_i=x_i,
            x_j=x_j,
            z_i_ij=z_edge.self_component,
            z_j_ij=z_j_ij_live,
            gamma_1=gamma_1,
            d_0=d_0,
            d_c=d_c,
            y_i_ij=state.y_pair[j].self_component,
            y_j_ij=all_fast_states[j].y_pair[i].self_component,
            step_idx=step_idx,
            w_i=state.w,
        )
        _append_row(A, b, row_labels, a_row, b_val, label, A_raw, b_raw)

    for k, obstacle in enumerate(obstacles):
        a_row, b_val, label = build_obstacle_constraint_row(
            i=i,
            k=k,
            x_i=x_i,
            obstacle=obstacle,
            gamma_2=gamma_2,
        )
        _append_row(A, b, row_labels, a_row, b_val, label, A_raw, b_raw)

    if enable_connectivity_constraint:
        a_row, b_val, label = build_connectivity_constraint_row(
            i=i,
            state=state,
            all_fast_states=all_fast_states,
            all_agent_states=all_agent_states,
            eta=eta,
            lambda_2=lambda_2,
            alpha_connectivity=alpha_connectivity,
            d_c=d_c,
            epsilon_c=epsilon_c,
            v2=v2,
            n_agents=n_agents,
        )
        _append_row(A, b, row_labels, a_row, b_val, label, A_raw, b_raw)

    a_row, b_val, label = build_clf_constraint_row(
        i=i,
        state=state,
        all_fast_states=all_fast_states,
        all_agent_states=all_agent_states,
        gamma_3=gamma_3,
        d_c=d_c,
        target_center=target_center,
        sigma=sigma,
        epsilon=epsilon,
        alpha_clf=alpha_clf,
        step_idx=step_idx,
        debug_disable_coupling=debug_disable_clf_coupling,
    )
    _append_row(A, b, row_labels, a_row, b_val, label, A_raw, b_raw)

    for a_row, b_val, label in build_actuator_box_rows(u_max=u_max):
        _append_row(A, b, row_labels, a_row, b_val, label, A_raw, b_raw)

    return LocalQPData(P=P, q=q, A=A, b=b, row_labels=row_labels, A_raw=A_raw, b_raw=b_raw)

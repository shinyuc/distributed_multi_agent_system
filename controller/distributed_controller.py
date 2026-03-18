"""Distributed controller skeleton (no optimization yet)."""

from __future__ import annotations

from config import ControllerConfig
from controller.base_controller import Controller
from controller.constraints import obstacle_constraint_coefficients
from controller.fast_dynamics import compute_V_i, compute_beta_i, compute_clf_gradient_term, h_ij, pairwise_constraint_value
from controller.local_qp import build_local_qp_data, solve_local_qp_with_status
from models.agent import AgentState, Vector2
from models.environment import Environment
from models.fast_state import FastState
import math

AffineConstraint = tuple[Vector2, float]


class LocalQPInfeasibleError(RuntimeError):
    """Raised when the local QP solve fails for an agent at a simulation step."""

    def __init__(self, step_idx: int | None, agent_id: int, diagnostics: dict[str, object]):
        self.step_idx = step_idx
        self.agent_id = agent_id
        self.diagnostics = diagnostics
        super().__init__(f"Local QP infeasible at step={step_idx}, agent={agent_id}.")


class DistributedController(Controller):
    """Placeholder distributed controller.

    Current behavior uses a nominal target-pointing input only. CLF/CBF-QP terms
    are intentionally left as TODO blocks for later implementation.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.qp_debug_records: dict[tuple[int, int], dict[str, object]] = {}
        # ######### check
        # self.enable_connectivity_constraint = config.enable_connectivity_constraint
        # print("[DEBUG DistributedController.__init__] enable_connectivity_constraint ="
        #       ,self.enable_connectivity_constraint,)

    def build_obstacle_constraints(self, agent: AgentState, env: Environment) -> list[AffineConstraint]:
        """Build affine obstacle constraints a^T u <= b for one agent."""
        constraints: list[AffineConstraint] = []
        for obstacle in env.obstacles:
            a, b = obstacle_constraint_coefficients(
                agent_position=agent.position,
                obstacle_center=obstacle.center,
                obstacle_radius=obstacle.radius,
                gamma_2=self.config.cbf_rate,
            )
            constraints.append((a, b))
        return constraints

    @staticmethod
    def _final_row_for_label(
        row_labels: list[str],
        A: list[list[float]],
        b: list[float],
        target_label: str,
    ) -> tuple[list[float] | None, float | None]:
        for idx, label in enumerate(row_labels):
            if label == target_label:
                return A[idx], float(b[idx])
        return None, None

    def _build_infeasibility_row_decomposition(
        self,
        *,
        agent_id: int,
        state: FastState,
        all_fast_states: dict[int, FastState],
        all_agent_states: dict[int, AgentState],
        qp_data_A: list[list[float]],
        qp_data_b: list[float],
        qp_data_labels: list[str],
        d_c: float,
        gamma_1: float,
        d_0: float,
        gamma_3: float,
        target_center: Vector2,
        sigma: list[list[float]],
        epsilon: float,
        eta: float,
        lambda_2: float,
        alpha_connectivity: float,
        epsilon_c: float,
        v2: dict[int, float] | None,
        n_agents: int,
    ) -> dict[str, object]:
        x_i = all_agent_states[agent_id].position
        out: dict[str, object] = {}

        pairwise_rows: list[dict[str, object]] = []
        for j, z_edge in state.z_pair.items():
            x_j = all_agent_states[j].position
            h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
            raw_mismatch = z_edge.self_component - z_edge.neighbor_component
            weighted_mismatch = h * raw_mismatch
            dx = x_i[0] - x_j[0]
            dy = x_i[1] - x_j[1]
            dist_sq = dx * dx + dy * dy
            geometric_term = 0.5 * gamma_1 * (dist_sq - d_0 * d_0)
            weighted_geometric = h * geometric_term
            label = f"pairwise(i={agent_id},j={j})"
            final_A, final_b = self._final_row_for_label(qp_data_labels, qp_data_A, qp_data_b, label)
            dominant = "weighted_geometric" if abs(weighted_geometric) >= abs(weighted_mismatch) else "weighted_mismatch"
            pairwise_rows.append(
                {
                    "label": label,
                    "neighbor": j,
                    "h_ij": float(h),
                    "raw_mismatch": float(raw_mismatch),
                    "weighted_mismatch": float(weighted_mismatch),
                    "geometric_term": float(geometric_term),
                    "weighted_geometric_term": float(weighted_geometric),
                    "final_A": final_A,
                    "final_b": final_b,
                    "dominant_term": dominant,
                }
            )
        out["pairwise"] = pairwise_rows

        v_i = compute_V_i(x_i=x_i, target_center=target_center, sigma=sigma, epsilon=epsilon)
        h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
        if v_i > 0.0:
            exp_factor = math.exp(-1.0 / v_i)
            alpha = 2.0 * exp_factor * (1.0 + 1.0 / v_i)
            w_i = exp_factor * v_i
            dx0 = x_i[0] - target_center[0]
            dx1 = x_i[1] - target_center[1]
            a_u0 = dx0 * sigma[0][0] + dx1 * sigma[1][0]
            a_u1 = dx0 * sigma[0][1] + dx1 * sigma[1][1]
            gradient_term = (float(alpha * a_u0), float(alpha * a_u1))
            gamma_w = float(gamma_3 * w_i)
            coupling_terms: list[dict[str, object]] = []
            coupling_sum = 0.0
            for j, fast_j in all_fast_states.items():
                if j == agent_id:
                    continue
                x_j = all_agent_states[j].position
                h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
                v_j = compute_V_i(x_i=x_j, target_center=target_center, sigma=sigma, epsilon=epsilon)
                h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
                raw = state.z_clf - fast_j.z_clf
                weighted = h * h_i * h_j * raw
                coupling_sum += weighted
                coupling_terms.append(
                    {
                        "neighbor": j,
                        "h_ij": float(h),
                        "h_i": float(h_i),
                        "h_j": float(h_j),
                        "raw_mismatch": float(raw),
                        "weighted_term": float(weighted),
                    }
                )
            max_coupling = max((abs(float(t["weighted_term"])) for t in coupling_terms), default=0.0)
            dominant = "gamma3W" if abs(gamma_w) >= max_coupling else "weighted_clf_coupling"
        else:
            w_i = 0.0
            gradient_term = (0.0, 0.0)
            gamma_w = 0.0
            coupling_terms = []
            coupling_sum = 0.0
            dominant = "inactive"
        clf_label = f"clf(i={agent_id},{'active' if v_i > 0.0 else 'inactive'})"
        clf_A, clf_b = self._final_row_for_label(qp_data_labels, qp_data_A, qp_data_b, clf_label)
        out["clf"] = {
            "label": clf_label,
            "V_i": float(v_i),
            "h_i": float(h_i),
            "W_i": float(w_i),
            "gradient_term_for_A": gradient_term,
            "gamma3W": float(gamma_w),
            "coupling_terms": coupling_terms,
            "coupling_sum": float(coupling_sum),
            "final_A": clf_A,
            "final_b": clf_b,
            "dominant_term": dominant,
        }

        beta_i = compute_beta_i(
            i=agent_id,
            state=state,
            all_fast_states=all_fast_states,
            all_agent_states=all_agent_states,
            d_c=d_c,
            epsilon_c=epsilon_c,
            v2=v2,
        )
        conn_terms: list[dict[str, object]] = []
        conn_sum_raw = 0.0
        for j, fast_j in all_fast_states.items():
            if j == agent_id:
                continue
            x_j = all_agent_states[j].position
            h = h_ij(x_i=x_i, x_j=x_j, d_c=d_c)
            raw = state.z_c - fast_j.z_c
            weighted = h * raw
            conn_sum_raw += weighted
            conn_terms.append(
                {
                    "neighbor": j,
                    "h_ij": float(h),
                    "raw_mismatch": float(raw),
                    "weighted_term": float(weighted),
                }
            )
        conn_sum_scaled = alpha_connectivity * conn_sum_raw
        lambda_term = float(lambda_2 / n_agents)
        eta_term = float(eta / n_agents)
        max_conn = max((abs(alpha_connectivity * float(t["weighted_term"])) for t in conn_terms), default=0.0)
        dominant_conn = "lambda2_over_N"
        largest = abs(-lambda_term)
        if abs(eta_term) > largest:
            dominant_conn = "eta_over_N"
            largest = abs(eta_term)
        if max_conn > largest:
            dominant_conn = "weighted_connectivity_coupling"
        conn_label = f"connectivity(i={agent_id})"
        conn_A, conn_b = self._final_row_for_label(qp_data_labels, qp_data_A, qp_data_b, conn_label)
        out["connectivity"] = {
            "label": conn_label,
            "beta_i": (float(beta_i[0]), float(beta_i[1])),
            "lambda2_over_N": lambda_term,
            "eta_over_N": eta_term,
            "coupling_terms": conn_terms,
            "raw_connectivity_coupling_sum": float(conn_sum_raw),
            "alpha_connectivity": float(alpha_connectivity),
            "scaled_connectivity_coupling_sum": float(conn_sum_scaled),
            "final_A": conn_A,
            "final_b": conn_b,
            "dominant_term": dominant_conn,
        }
        return out

    def compute_control(
        self,
        agent: AgentState,
        env: Environment,
        **kwargs: object,
    ) -> Vector2:
        # 1) Collect obstacle affine constraints a^T u <= b for this agent.
        obstacle_constraints = self.build_obstacle_constraints(agent, env)
        _ = obstacle_constraints  # Kept for compatibility/inspection.

        fast_state = kwargs.get("fast_state")
        all_fast_states = kwargs.get("all_fast_states")
        all_agent_states = kwargs.get("all_agent_states")
        d_c = kwargs.get("d_c")
        step_idx = kwargs.get("step_idx")
        debug_qp = isinstance(step_idx, int) and step_idx >= 100 and agent.agent_id == 3
        debug_pair_12 = isinstance(step_idx, int) and step_idx in {207, 208} and agent.agent_id in {1, 2}
        debug_full_207 = isinstance(step_idx, int) and step_idx == 207 and agent.agent_id in {1, 2}

        if (
            not isinstance(fast_state, FastState)
            or not isinstance(all_fast_states, dict)
            or not isinstance(all_agent_states, dict)
            or not isinstance(d_c, (int, float))
        ):
            raise ValueError(
                "compute_control requires fast_state/all_fast_states/all_agent_states/d_c."
            )

        if self.config.enable_connectivity_constraint:
            if "lambda_2" not in kwargs or "v2" not in kwargs:
                raise ValueError(
                    "Connectivity is enabled, but runtime lambda_2/v2 were not provided."
                )
            if not isinstance(kwargs["lambda_2"], (int, float)):
                raise ValueError("Runtime lambda_2 must be a float.")
            if not isinstance(kwargs["v2"], dict):
                raise ValueError("Runtime v2 must be a dict[int, float].")

        qp_data = build_local_qp_data(
            i=agent.agent_id,
            state=fast_state,
            all_fast_states=all_fast_states,
            all_agent_states=all_agent_states,
            obstacles=env.obstacles,
            u_max=self.config.u_max,
            gamma_1=float(kwargs.get("gamma_1", self.config.cbf_rate)),
            d_0=float(kwargs.get("d_0", self.config.safety_distance)),
            gamma_2=self.config.cbf_rate,
            eta=float(kwargs.get("eta", 0.0)),
            d_c=float(d_c),
            epsilon_c=float(kwargs.get("epsilon_c", 0.5)),
            gamma_3=float(kwargs.get("gamma_3", self.config.clf_rate)),
            target_center=kwargs.get("target_center", env.target.center),  # type: ignore[arg-type]
            sigma=kwargs.get("sigma", [[1.0, 0.0], [0.0, 1.0]]),  # type: ignore[arg-type]
            epsilon=float(kwargs.get("epsilon", 1e-2)),
            step_idx=(int(step_idx) if isinstance(step_idx, int) else None),
            enable_connectivity_constraint=self.config.enable_connectivity_constraint,
            alpha_clf=self.config.alpha_clf,
            alpha_connectivity=self.config.alpha_connectivity,
            debug_disable_clf_coupling=self.config.debug_disable_clf_coupling,
            lambda_2=float(kwargs["lambda_2"]) if "lambda_2" in kwargs else None,
            v2=kwargs["v2"] if "v2" in kwargs else None,  # type: ignore[arg-type]
            n_agents=int(kwargs.get("n_agents", len(all_agent_states))),
        )
        if (
            agent.agent_id in {0, 3}
            and isinstance(step_idx, int)
            and step_idx >= 100
        ):
            for label, a_row, b_val in zip(qp_data.row_labels, qp_data.A, qp_data.b):
                if label.startswith("clf("):
                    residual_zero = -float(b_val)
                    print(
                        f"[DEBUG clf row] step={step_idx}, agent={agent.agent_id}, "
                        f"a_row=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), "
                        f"b_val={float(b_val):.6f}, residual_zero={residual_zero:.6f}"
                    )
                    break
        if debug_qp:
            for label, a_row, b_val in zip(qp_data.row_labels, qp_data.A, qp_data.b):
                residual_zero = -float(b_val)
                print(
                    f"[DEBUG qp row] step={step_idx}, agent=3, label={label}, "
                    f"a=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), "
                    f"b={float(b_val):.6f}, residual_zero={residual_zero:.6f}"
                )
        if debug_full_207:
            print(f"[DEBUG full qp] step=207, agent={agent.agent_id}, n_rows={len(qp_data.row_labels)}")
            for label, a_row, b_val in zip(qp_data.row_labels, qp_data.A, qp_data.b):
                residual_zero = -float(b_val)
                active_or_conflicting = residual_zero > 0.0
                print(
                    f"[DEBUG full qp row] step=207, agent={agent.agent_id}, "
                    f"label={label}, "
                    f"A=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), "
                    f"b={float(b_val):.6f}, residual_zero={residual_zero:.6f}, "
                    f"active_or_conflicting_at_u0={active_or_conflicting}"
                )

        u_star, solve_ok, reason = solve_local_qp_with_status(qp_data)
        z_pair_snapshot: dict[int, tuple[float, float]] = {
            j: (edge.self_component, edge.neighbor_component)
            for j, edge in fast_state.z_pair.items()
        }
        if isinstance(step_idx, int):
            self.qp_debug_records[(step_idx, agent.agent_id)] = {
                "row_labels": list(qp_data.row_labels),
                "A": [list(row) for row in qp_data.A],
                "b": list(qp_data.b),
                "solve_ok": solve_ok,
                "solve_reason": reason,
                "u_solver": (float(u_star[0]), float(u_star[1])),
                "fallback_used": (not solve_ok),
                "z_pair_snapshot": z_pair_snapshot,
            }
        if debug_pair_12:
            other = 2 if agent.agent_id == 1 else 1
            pair_label = f"pairwise(i={agent.agent_id},j={other})"
            row_idx = None
            for idx, label in enumerate(qp_data.row_labels):
                if label == pair_label:
                    row_idx = idx
                    break

            x_i = all_agent_states[agent.agent_id].position
            x_j = all_agent_states[other].position
            dx = x_i[0] - x_j[0]
            dy = x_i[1] - x_j[1]
            d0 = float(kwargs.get("d_0", self.config.safety_distance))
            h12_safety = dx * dx + dy * dy - d0 * d0
            h12_comm = h_ij(x_i=x_i, x_j=x_j, d_c=float(d_c))
            print(
                f"[DEBUG pair12 cbf] step={step_idx}, agent={agent.agent_id}, "
                f"pair_label={pair_label}, included={row_idx is not None}, "
                f"h12_safety={h12_safety:.6f}, h12_comm={h12_comm:.6f}"
            )
            if row_idx is not None:
                a_row = qp_data.A[row_idx]
                b_val = qp_data.b[row_idx]
                residual = float(a_row[0]) * u_star[0] + float(a_row[1]) * u_star[1] - float(b_val)
                print(
                    f"[DEBUG pair12 cbf row] step={step_idx}, agent={agent.agent_id}, "
                    f"A=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), b={float(b_val):.6f}, "
                    f"residual={residual:.6f}, solve_ok={solve_ok}, reason={reason}"
                )
            else:
                print(
                    f"[DEBUG pair12 cbf row] step={step_idx}, agent={agent.agent_id}, "
                    f"row_missing=True, solve_ok={solve_ok}, reason={reason}"
                )
        if debug_qp:
            print(
                f"[DEBUG qp solve] step={step_idx}, agent={agent.agent_id}, "
                f"success={solve_ok}, solver_status_or_reason={reason}"
            )
            print(f"[DEBUG qp solve] step={step_idx}, agent={agent.agent_id}, row_labels={qp_data.row_labels}")
            print(f"[DEBUG qp solve] step={step_idx}, agent={agent.agent_id}, returned_control={u_star}")
        if debug_full_207:
            print(
                f"[DEBUG full qp solve] step=207, agent={agent.agent_id}, "
                f"success={solve_ok}, solver_status_or_reason={reason}"
            )
            print(
                f"[DEBUG full qp solve] step=207, agent={agent.agent_id}, "
                f"solver_returned_control={u_star}"
            )

        if not solve_ok:
            target_center = kwargs.get("target_center", env.target.center)  # type: ignore[assignment]
            sigma = kwargs.get("sigma", [[1.0, 0.0], [0.0, 1.0]])  # type: ignore[assignment]
            epsilon = float(kwargs.get("epsilon", 1e-2))
            gamma_3 = float(kwargs.get("gamma_3", self.config.clf_rate))
            print(
                f"[DEBUG root-cause summary] step={step_idx}, failed_agent={agent.agent_id}, "
                f"first_trigger=gamma3W, step0_clf_coupling_active=no, "
                f"step1_pairwise_active=no, pairwise_active_after_w_growth=yes"
            )
            beta_i = compute_beta_i(
                i=agent.agent_id,
                state=fast_state,
                all_fast_states=all_fast_states,
                all_agent_states=all_agent_states,
                d_c=float(d_c),
                epsilon_c=float(kwargs.get("epsilon_c", 0.5)),
                v2=kwargs.get("v2") if isinstance(kwargs.get("v2"), dict) else None,  # type: ignore[arg-type]
            )
            for i_id, fs in all_fast_states.items():
                x_i = all_agent_states[i_id].position
                pairwise_sum_x = 0.0
                pairwise_sum_y = 0.0
                for j_id, z_edge in fs.z_pair.items():
                    if j_id not in fs.y_pair:
                        continue
                    x_j = all_agent_states[j_id].position
                    y_edge = fs.y_pair[j_id]
                    z_j_ij_live = all_fast_states[j_id].z_pair[i_id].self_component
                    dx = x_i[0] - x_j[0]
                    dy = x_i[1] - x_j[1]
                    h = h_ij(x_i=x_i, x_j=x_j, d_c=float(d_c))
                    velocity_term = -2.0 * (dx * fs.w[0] + dy * fs.w[1])
                    barrier_term = -0.5 * float(kwargs.get("gamma_1", self.config.cbf_rate)) * (
                        (dx * dx + dy * dy) - float(kwargs.get("d_0", self.config.safety_distance)) ** 2
                    )
                    mismatch_term = z_edge.self_component - z_j_ij_live
                    g_ij = pairwise_constraint_value(
                        x_i=x_i,
                        x_j=x_j,
                        w_i=fs.w,
                        z_i_ij=z_edge.self_component,
                        z_j_ij=z_j_ij_live,
                        gamma_1=float(kwargs.get("gamma_1", self.config.cbf_rate)),
                        d_0=float(kwargs.get("d_0", self.config.safety_distance)),
                        d_c=float(d_c),
                    )
                    coeff = 2.0 * h * y_edge.self_component
                    contribution_x = coeff * dx
                    contribution_y = coeff * dy
                    contribution_norm = math.sqrt(
                        contribution_x * contribution_x + contribution_y * contribution_y
                    )
                    pairwise_sum_x += contribution_x
                    pairwise_sum_y += contribution_y
                    print(
                        f"[DEBUG pairwise g infeasible] step={step_idx}, failed_agent={agent.agent_id}, "
                        f"agent={i_id}, neighbor={j_id}, "
                        f"z_i_ij={z_edge.self_component:.12f}, y_i_ij={y_edge.self_component:.12f}, "
                        f"z_j_ij_live={z_j_ij_live:.12f}, w_i=({fs.w[0]:.12f}, {fs.w[1]:.12f}), "
                        f"velocity_term={velocity_term:.12f}, barrier_term={barrier_term:.12f}, "
                        f"mismatch_term={mismatch_term:.12f}, h_ij={h:.12f}, g_ij={g_ij:.12f}"
                    )
                    print(
                        f"[DEBUG pairwise w term infeasible] step={step_idx}, failed_agent={agent.agent_id}, "
                        f"agent={i_id}, neighbor={j_id}, "
                        f"contribution=({contribution_x:.12f}, {contribution_y:.12f}), "
                        f"contribution_norm={contribution_norm:.12f}"
                    )
                pairwise_sum_norm = math.sqrt(pairwise_sum_x * pairwise_sum_x + pairwise_sum_y * pairwise_sum_y)
                print(
                    f"[DEBUG pairwise w total infeasible] step={step_idx}, failed_agent={agent.agent_id}, "
                    f"agent={i_id}, pair_sum=({pairwise_sum_x:.12f}, {pairwise_sum_y:.12f}), "
                    f"pair_sum_norm={pairwise_sum_norm:.12f}"
                )
                beta_i_all = compute_beta_i(
                    i=i_id,
                    state=fs,
                    all_fast_states=all_fast_states,
                    all_agent_states=all_agent_states,
                    d_c=float(d_c),
                    epsilon_c=float(kwargs.get("epsilon_c", 0.5)),
                    v2=kwargs.get("v2") if isinstance(kwargs.get("v2"), dict) else None,  # type: ignore[arg-type]
                )
                connectivity_term = (
                    beta_i_all[0] * fs.y_c,
                    beta_i_all[1] * fs.y_c,
                )
                v_i = compute_V_i(
                    x_i=x_i,
                    target_center=target_center,  # type: ignore[arg-type]
                    sigma=sigma,  # type: ignore[arg-type]
                    epsilon=epsilon,
                )
                clf_term = compute_clf_gradient_term(
                    x_i=x_i,
                    target_center=target_center,  # type: ignore[arg-type]
                    sigma=sigma,  # type: ignore[arg-type]
                    y_i_clf=fs.y_clf,
                    v_i=v_i,
                )
                h_i_clf = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
                exp_factor_clf = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
                w_i_scalar = exp_factor_clf * v_i if v_i > 0.0 else 0.0
                gamma_w_term = gamma_3 * w_i_scalar
                clf_coupling_sum = 0.0
                clf_drift_term = 0.0
                clf_coupling_contributions: list[tuple[int, float, float, float, float]] = []
                if v_i > 0.0:
                    dx0 = x_i[0] - target_center[0]  # type: ignore[index]
                    dx1 = x_i[1] - target_center[1]  # type: ignore[index]
                    s00, s01 = sigma[0]  # type: ignore[index]
                    s10, s11 = sigma[1]  # type: ignore[index]
                    sigma_w0 = s00 * fs.w[0] + s01 * fs.w[1]
                    sigma_w1 = s10 * fs.w[0] + s11 * fs.w[1]
                    x_sigma_w = dx0 * sigma_w0 + dx1 * sigma_w1
                    clf_drift_term = 2.0 * exp_factor_clf * (1.0 + 1.0 / v_i) * x_sigma_w
                if h_i_clf != 0.0:
                    for j_id, fast_j in all_fast_states.items():
                        if j_id == i_id:
                            continue
                        x_j = all_agent_states[j_id].position
                        h = h_ij(x_i=x_i, x_j=x_j, d_c=float(d_c))
                        v_j = compute_V_i(
                            x_i=x_j,
                            target_center=target_center,  # type: ignore[arg-type]
                            sigma=sigma,  # type: ignore[arg-type]
                            epsilon=epsilon,
                        )
                        h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
                        contribution = h * h_i_clf * h_j * (fs.z_clf - fast_j.z_clf)
                        clf_coupling_sum += contribution
                        clf_coupling_contributions.append((j_id, h, v_j, h_j, contribution))
                obstacle_term_x = 0.0
                obstacle_term_y = 0.0
                for obstacle, y_obs in zip(env.obstacles, fs.y_obs):
                    dx_obs = x_i[0] - obstacle.center[0]
                    dy_obs = x_i[1] - obstacle.center[1]
                    obstacle_term_x += 2.0 * dx_obs * y_obs
                    obstacle_term_y += 2.0 * dy_obs * y_obs
                obstacle_term = (obstacle_term_x, obstacle_term_y)
                input_bound_term = (-fs.y_u[0] + fs.y_u[2], -fs.y_u[1] + fs.y_u[3])
                damping_term = (-fs.w[0], -fs.w[1])
                total_rhs = (
                    damping_term[0] + pairwise_sum_x + connectivity_term[0] + clf_term[0] + obstacle_term[0] + input_bound_term[0],
                    damping_term[1] + pairwise_sum_y + connectivity_term[1] + clf_term[1] + obstacle_term[1] + input_bound_term[1],
                )
                damping_norm = math.sqrt(damping_term[0] * damping_term[0] + damping_term[1] * damping_term[1])
                connectivity_norm = math.sqrt(
                    connectivity_term[0] * connectivity_term[0] + connectivity_term[1] * connectivity_term[1]
                )
                clf_term_norm = math.sqrt(clf_term[0] * clf_term[0] + clf_term[1] * clf_term[1])
                obstacle_term_norm = math.sqrt(
                    obstacle_term[0] * obstacle_term[0] + obstacle_term[1] * obstacle_term[1]
                )
                input_term_norm = math.sqrt(
                    input_bound_term[0] * input_bound_term[0] + input_bound_term[1] * input_bound_term[1]
                )
                rhs_norm = math.sqrt(total_rhs[0] * total_rhs[0] + total_rhs[1] * total_rhs[1])
                print(
                    f"[DEBUG update_w rhs infeasible] step={step_idx}, failed_agent={agent.agent_id}, agent={i_id}"
                )
                print(f"  w_old=({fs.w[0]:.12f}, {fs.w[1]:.12f})")
                print(f"  y_c={fs.y_c:.12f}")
                print(f"  y_clf={fs.y_clf:.12f}")
                print(f"  y_obs={[float(value) for value in fs.y_obs]}")
                print(f"  y_u={[float(value) for value in fs.y_u]}")
                print(
                    f"  damping_term=({damping_term[0]:.12f}, {damping_term[1]:.12f}), "
                    f"norm={damping_norm:.12f}"
                )
                print(
                    f"  pairwise_term=({pairwise_sum_x:.12f}, {pairwise_sum_y:.12f}), "
                    f"norm={pairwise_sum_norm:.12f}"
                )
                print(f"  beta_i=({beta_i_all[0]:.12f}, {beta_i_all[1]:.12f})")
                print(
                    f"  connectivity_term=({connectivity_term[0]:.12f}, {connectivity_term[1]:.12f}), "
                    f"norm={connectivity_norm:.12f}"
                )
                print(f"  v_i={v_i:.12f}")
                print(f"  z_clf={fs.z_clf:.12f}")
                print(f"  y_clf={fs.y_clf:.12f}")
                print(f"  W_i={w_i_scalar:.12f}")
                print(f"  gamma3W={gamma_w_term:.12f}")
                print(f"  clf_drift={clf_drift_term:.12f}")
                for j_id, h, v_j, h_j, contribution in clf_coupling_contributions:
                    print(
                        f"  clf_neighbor={j_id}, h_ij={h:.12f}, V_j={v_j:.12f}, h_j={h_j:.12f}, "
                        f"clf_coupling_contribution={contribution:.12f}"
                    )
                print(f"  clf_coupling_sum={clf_coupling_sum:.12f}")
                print(
                    f"  g_clf={clf_drift_term + gamma_w_term + clf_coupling_sum:.12f}"
                )
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
            positions_snapshot = {
                i_id: {
                    "position": (
                        float(agent_state.position[0]),
                        float(agent_state.position[1]),
                    )
                }
                for i_id, agent_state in all_agent_states.items()
            }
            pairwise_distances: list[dict[str, object]] = []
            ids = sorted(all_agent_states.keys())
            for idx, i_id in enumerate(ids):
                xi, yi = all_agent_states[i_id].position
                for j_id in ids[idx + 1:]:
                    xj, yj = all_agent_states[j_id].position
                    dx = xi - xj
                    dy = yi - yj
                    dist = (dx * dx + dy * dy) ** 0.5
                    pairwise_distances.append({"pair": (i_id, j_id), "distance": float(dist)})
            fast_states_snapshot = {
                i_id: {
                    "w": (float(fs.w[0]), float(fs.w[1])),
                    "z_c": float(fs.z_c),
                    "z_clf": float(fs.z_clf),
                    "y_c": float(fs.y_c),
                    "y_clf": float(fs.y_clf),
                    "y_obs": [float(value) for value in fs.y_obs],
                    "y_u": [float(value) for value in fs.y_u],
                    "z_pair": {
                        j_id: {
                            "self_component": float(edge.self_component),
                            "neighbor_component": float(edge.neighbor_component),
                        }
                        for j_id, edge in fs.z_pair.items()
                    },
                    "y_pair": {
                        j_id: {
                            "self_component": float(edge.self_component),
                            "neighbor_component": float(edge.neighbor_component),
                        }
                        for j_id, edge in fs.y_pair.items()
                    },
                }
                for i_id, fs in all_fast_states.items()
            }
            diagnostics: dict[str, object] = {
                "step_idx": step_idx,
                "agent_id": agent.agent_id,
                "solver_status": reason,
                "row_labels": list(qp_data.row_labels),
                "A": [list(row) for row in qp_data.A],
                "b": list(qp_data.b),
                "positions": positions_snapshot,
                "pairwise_distances": pairwise_distances,
                "lambda_2": float(kwargs["lambda_2"]) if "lambda_2" in kwargs else None,
                "v2": (
                    {int(k): float(v) for k, v in kwargs["v2"].items()}
                    if "v2" in kwargs and isinstance(kwargs["v2"], dict)
                    else None
                ),
                "beta_i": (float(beta_i[0]), float(beta_i[1])),
                "fast_states": fast_states_snapshot,
                "row_decomposition": self._build_infeasibility_row_decomposition(
                    agent_id=agent.agent_id,
                    state=fast_state,
                    all_fast_states=all_fast_states,
                    all_agent_states=all_agent_states,
                    qp_data_A=qp_data.A,
                    qp_data_b=qp_data.b,
                    qp_data_labels=qp_data.row_labels,
                    d_c=float(d_c),
                    gamma_1=float(kwargs.get("gamma_1", self.config.cbf_rate)),
                    d_0=float(kwargs.get("d_0", self.config.safety_distance)),
                    gamma_3=float(kwargs.get("gamma_3", self.config.clf_rate)),
                    target_center=kwargs.get("target_center", env.target.center),  # type: ignore[arg-type]
                    sigma=kwargs.get("sigma", [[1.0, 0.0], [0.0, 1.0]]),  # type: ignore[arg-type]
                    epsilon=float(kwargs.get("epsilon", 1e-2)),
                    eta=float(kwargs.get("eta", 0.0)),
                    lambda_2=float(kwargs["lambda_2"]) if "lambda_2" in kwargs else 0.0,
                    alpha_connectivity=float(self.config.alpha_connectivity),
                    epsilon_c=float(kwargs.get("epsilon_c", 0.5)),
                    v2=kwargs.get("v2") if isinstance(kwargs.get("v2"), dict) else None,  # type: ignore[arg-type]
                    n_agents=int(kwargs.get("n_agents", len(all_agent_states))),
                ),
            }
            if isinstance(step_idx, int):
                self.qp_debug_records[(step_idx, agent.agent_id)]["u_returned"] = (
                    float("nan"),
                    float("nan"),
                )
                self.qp_debug_records[(step_idx, agent.agent_id)]["infeasible_diagnostics"] = diagnostics
            raise LocalQPInfeasibleError(
                step_idx=(int(step_idx) if isinstance(step_idx, int) else None),
                agent_id=agent.agent_id,
                diagnostics=diagnostics,
            )

        if isinstance(step_idx, int):
            self.qp_debug_records[(step_idx, agent.agent_id)]["u_returned"] = (
                float(u_star[0]),
                float(u_star[1]),
            )
        return (float(u_star[0]), float(u_star[1]))

"""Simple simulation loop for multi-agent dynamics."""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from controller.base_controller import Controller
from controller.centralized_controller import CentralizedQPInfeasibleError
from controller.distributed_controller import LocalQPInfeasibleError
from controller.fast_dynamics import alpha_ij_connectivity, connectivity_sigma, update_fast_state
from models.agent import AgentState, Vector2
from models.environment import Environment
from models.fast_state import FastState, init_fast_state_complete_graph
import numpy as np


TERMINATION_REASONS = (
    "completed",               # every configured step was applied
    "local_qp_failure",        # >= 1 local QP solve failed (see local_qp_failure_kind)
    "centralized_qp_failure",  # the joint QP failed (centralized mode)
    "numerical_guard",         # fast-state explosion guard tripped
    "exception",               # any other unexpected exception
)
SOLVER_INFEASIBLE_STATUSES = {"infeasible", "infeasible_inaccurate"}


class NumericalGuardError(RuntimeError):
    """Raised by the fast-state explosion guard (a RuntimeError, as before)."""


def _solver_status_from_reason(reason: object) -> str:
    """Recover the raw solver status from the controller's failure reason string."""
    text = str(reason)
    if text.startswith("solver status="):
        return text[len("solver status="):]
    if text.startswith("exception"):
        return "exception"
    return "other"


@dataclass
class SimulationResult:
    """Container for trajectories and safety-related metric time-series."""

    trajectories: list[list[Vector2]]
    min_obstacle_clearance: list[float | None]
    min_inter_agent_distance: list[float | None]
    first_target_entry_step: dict[int, int | None]
    control_history: dict[int, list[Vector2]]
    fast_w_norm: dict[int, list[float]]
    fast_z_c: dict[int, list[float]]
    fast_z_clf: dict[int, list[float]]
    fast_y_c: dict[int, list[float]]
    fast_y_clf: dict[int, list[float]]
    fast_max_abs_z_pair_self: dict[int, list[float]]
    fast_max_abs_y_pair_self: dict[int, list[float]]
    lambda2_comm: list[float]
    lambda2_cons: list[float]
    stopped_early: bool
    infeasible_step: int | None
    infeasible_agent_id: int | None
    infeasible_diagnostics: dict[str, object] | None
    # All agents whose local QP failed at infeasible_step (all agents are solved
    # before the simulation stops).
    infeasible_agent_ids: list[int] = field(default_factory=list)
    # Termination bookkeeping. control_history holds applied inputs only; inputs
    # computed at the terminating step but never applied are in unapplied_controls.
    termination_reason: str = "completed"
    termination_step: int | None = None
    steps_configured: int = 0
    steps_completed: int = 0
    local_qp_failure_kind: str | None = None
    qp_failures: list[dict[str, Any]] = field(default_factory=list)
    qp_failure_diagnostics: dict[int, dict[str, object]] = field(default_factory=dict)
    unapplied_controls: dict[int, Vector2] = field(default_factory=dict)
    termination_exception: dict[str, str] | None = None

    @property
    def horizon_completed(self) -> bool:
        return self.termination_reason == "completed" and self.steps_completed == self.steps_configured


class Simulator:
    """Discrete-time simulator for a fixed environment and controller."""

    def __init__(
        self,
        env: Environment,
        controller: Controller,
        dt: float,
        steps: int,
        fast_tau: float = 0.1,
        fast_xi: float = 1.0,
        fast_d_c: float = 2.0,
        fast_target_center: Vector2 | None = None,
        fast_sigma: list[list[float]] | None = None,
        fast_epsilon: float = 0.01,
        fast_gamma_1: float = 1.0,
        fast_d_0: float = 0.35,
        fast_eta: float = 0.1,
        fast_epsilon_c: float = 0.5,
        fast_lambda_2: float = 0.0,
        fast_v2: list[float] | None = None,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}.")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}.")
        if fast_tau <= 0:
            raise ValueError(f"fast_tau must be > 0, got {fast_tau}.")
        if fast_xi < 0:
            raise ValueError(f"fast_xi must be >= 0, got {fast_xi}.")
        if fast_d_c <= 0:
            raise ValueError(f"fast_d_c must be > 0, got {fast_d_c}.")
        if fast_target_center is None:
            fast_target_center = env.target.center
        if fast_sigma is None:
            fast_sigma = [[1.0, 0.0], [0.0, 1.0]]
        if (
            len(fast_sigma) != 2
            or any(len(row) != 2 for row in fast_sigma)
        ):
            raise ValueError("fast_sigma must be a 2x2 matrix.")
        if fast_epsilon <= 0:
            raise ValueError(f"fast_epsilon must be > 0, got {fast_epsilon}.")
        if fast_gamma_1 <= 0:
            raise ValueError(f"fast_gamma_1 must be > 0, got {fast_gamma_1}.")
        if fast_d_0 <= 0:
            raise ValueError(f"fast_d_0 must be > 0, got {fast_d_0}.")
        if fast_eta < 0:
            raise ValueError(f"fast_eta must be >= 0, got {fast_eta}.")
        if fast_epsilon_c <= 0:
            raise ValueError(f"fast_epsilon_c must be > 0, got {fast_epsilon_c}.")
        if fast_epsilon_c >= fast_d_c:
            raise ValueError("fast_epsilon_c must satisfy fast_epsilon_c < fast_d_c.")
        if fast_v2 is not None and len(fast_v2) != len(env.agents):
            raise ValueError(
                "fast_v2 must have one scalar per agent: "
                f"len(fast_v2)={len(fast_v2)} != num_agents={len(env.agents)}"
            )

        self.env = env
        self.agents: list[AgentState] = env.agents
        self.fast_states: dict[int, FastState] = self.initialize_fast_states()
        self.controller = controller
        self.dt = dt
        self.steps = steps
        self.fast_tau = fast_tau
        self.fast_xi = fast_xi
        self.fast_d_c = fast_d_c
        self.fast_target_center = fast_target_center
        self.fast_sigma = fast_sigma
        self.fast_epsilon = fast_epsilon
        self.fast_gamma_1 = fast_gamma_1
        self.fast_d_0 = fast_d_0
        self.fast_eta = fast_eta
        self.fast_epsilon_c = fast_epsilon_c
        self.fast_lambda_2 = fast_lambda_2
        self.fast_v2 = fast_v2
        self.fast_v2_map: dict[int, float] | None = None
        if fast_v2 is not None:
            self.fast_v2_map = {agent.agent_id: float(fast_v2[agent.agent_id]) for agent in env.agents}

    def initialize_fast_states(self) -> dict[int, FastState]:
        """Initialize one FastState per agent using a complete graph."""
        n_agents = len(self.agents)
        n_obstacles = len(self.env.obstacles)
        return {
            agent.agent_id: init_fast_state_complete_graph(
                agent.agent_id,
                n_agents,
                n_obstacles,
            )
            for agent in self.agents
        }

    def get_fast_state(self, agent_id: int) -> FastState:
        """Return fast state for the requested agent_id."""
        try:
            return self.fast_states[agent_id]
        except KeyError as exc:
            raise KeyError(f"No fast state found for agent_id={agent_id}.") from exc

    def _agent_state_map(self) -> dict[int, AgentState]:
        return {agent.agent_id: agent for agent in self.agents}

    def _compute_connectivity_runtime_spectrum(
        self,
        all_agent_states: dict[int, AgentState],
    ) -> tuple[np.ndarray, np.ndarray, float, dict[int, float]]:
        """Compute connectivity adjacency/Laplacian and runtime lambda_2, v2."""
        agent_ids = [agent.agent_id for agent in self.agents]
        n_agents = len(agent_ids)
        adjacency = np.zeros((n_agents, n_agents), dtype=float)
        sigma_conn = connectivity_sigma(d_c=self.fast_d_c, epsilon_c=self.fast_epsilon_c)
        for a in range(n_agents):
            i = agent_ids[a]
            x_i = all_agent_states[i].position
            for b in range(a + 1, n_agents):
                j = agent_ids[b]
                x_j = all_agent_states[j].position
                weight = alpha_ij_connectivity(
                    x_i=x_i,
                    x_j=x_j,
                    d_c=self.fast_d_c,
                    epsilon_c=self.fast_epsilon_c,
                    sigma=sigma_conn,
                )
                adjacency[a, b] = weight
                adjacency[b, a] = weight

        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        if n_agents >= 2:
            lambda_2 = float(eigvals[1])
            v2_vec = eigvecs[:, 1].copy()
        else:
            lambda_2 = 0.0
            v2_vec = np.ones((1,), dtype=float)

        for idx in range(v2_vec.shape[0]):
            if abs(float(v2_vec[idx])) > 1e-12:
                if v2_vec[idx] < 0.0:
                    v2_vec = -v2_vec
                break

        v2_map = {agent_ids[idx]: float(v2_vec[idx]) for idx in range(n_agents)}
        return adjacency, laplacian, lambda_2, v2_map

    def _check_fast_state_explosion(
        self,
        step_idx: int,
        agent_id: int,
        state: FastState,
        threshold: float = 1e6,
    ) -> None:
        def _raise(var_name: str, value: float) -> None:
            print(f"[NUMERICAL_GUARD] step={step_idx} agent={agent_id} variable={var_name} value={value}")
            raise NumericalGuardError(
                f"Numerical explosion detected at step={step_idx}, agent={agent_id}, "
                f"variable={var_name}, value={value}, threshold={threshold}"
            )

        wx, wy = state.w
        w_norm = sqrt(wx * wx + wy * wy)
        if abs(w_norm) > threshold:
            _raise("w_norm", w_norm)
        if abs(state.z_c) > threshold:
            _raise("z_c", state.z_c)
        if abs(state.z_clf) > threshold:
            _raise("z_clf", state.z_clf)
        if abs(state.y_c) > threshold:
            _raise("y_c", state.y_c)
        if abs(state.y_clf) > threshold:
            _raise("y_clf", state.y_clf)

        for j, edge in state.z_pair.items():
            if abs(edge.self_component) > threshold:
                _raise(f"z_pair[{j}].self_component", edge.self_component)
            if abs(edge.neighbor_component) > threshold:
                _raise(f"z_pair[{j}].neighbor_component", edge.neighbor_component)
        for j, edge in state.y_pair.items():
            if abs(edge.self_component) > threshold:
                _raise(f"y_pair[{j}].self_component", edge.self_component)
            if abs(edge.neighbor_component) > threshold:
                _raise(f"y_pair[{j}].neighbor_component", edge.neighbor_component)

    def _is_in_target_region(self, position: Vector2) -> bool:
        dx = position[0] - self.fast_target_center[0]
        dy = position[1] - self.fast_target_center[1]
        sigma = self.fast_sigma
        sigma_dx0 = sigma[0][0] * dx + sigma[0][1] * dy
        sigma_dx1 = sigma[1][0] * dx + sigma[1][1] * dy
        value = dx * sigma_dx0 + dy * sigma_dx1
        return value <= 1.0

    def _update_first_target_entry(
        self,
        first_entry: dict[int, int | None],
        step_idx: int,
    ) -> None:
        for agent in self.agents:
            i = agent.agent_id
            if first_entry[i] is None and self._is_in_target_region(agent.position):
                first_entry[i] = step_idx

    def _compute_target_phi(self, position: Vector2) -> float:
        dx = position[0] - self.fast_target_center[0]
        dy = position[1] - self.fast_target_center[1]
        sigma = self.fast_sigma
        sigma_dx0 = sigma[0][0] * dx + sigma[0][1] * dy
        sigma_dx1 = sigma[1][0] * dx + sigma[1][1] * dy
        return dx * sigma_dx0 + dy * sigma_dx1 - 1.0

    def _debug_print_target_phi(self, step_idx: int) -> None:
        phi_values: list[float] = []
        for agent in self.agents:
            phi_i = self._compute_target_phi(agent.position)
            phi_values.append(phi_i)
            print(f"[DEBUG target metric] step={step_idx}, agent={agent.agent_id}, phi={phi_i:.6f}")
        summary = ", ".join(f"{v:.6f}" for v in phi_values)
        print(f"[DEBUG target summary] step={step_idx}, phi=[{summary}]")

    def _init_control_history(self) -> dict[int, list[Vector2]]:
        return {agent.agent_id: [] for agent in self.agents}

    def _init_fast_history(self) -> dict[str, dict[int, list[float]]]:
        agent_ids = [agent.agent_id for agent in self.agents]
        keys = [
            "w_norm",
            "z_c",
            "z_clf",
            "y_c",
            "y_clf",
            "max_abs_z_pair_self",
            "max_abs_y_pair_self",
        ]
        return {k: {i: [] for i in agent_ids} for k in keys}

    def _record_fast_history_snapshot(self, history: dict[str, dict[int, list[float]]]) -> None:
        for agent in self.agents:
            i = agent.agent_id
            state = self.fast_states[i]
            wx, wy = state.w
            history["w_norm"][i].append(sqrt(wx * wx + wy * wy))
            history["z_c"][i].append(state.z_c)
            history["z_clf"][i].append(state.z_clf)
            history["y_c"][i].append(state.y_c)
            history["y_clf"][i].append(state.y_clf)
            max_abs_z = max((abs(edge.self_component) for edge in state.z_pair.values()), default=0.0)
            max_abs_y = max((abs(edge.self_component) for edge in state.y_pair.values()), default=0.0)
            history["max_abs_z_pair_self"][i].append(max_abs_z)
            history["max_abs_y_pair_self"][i].append(max_abs_y)

    def _min_obstacle_clearance(self, positions: list[Vector2]) -> float | None:
        if not self.env.obstacles:
            return None
        min_clearance = float("inf")
        for x, y in positions:
            for obs in self.env.obstacles:
                cx, cy = obs.center
                clearance = sqrt((x - cx) ** 2 + (y - cy) ** 2) - obs.radius
                if clearance < min_clearance:
                    min_clearance = clearance
        return min_clearance

    def _min_inter_agent_distance(self, positions: list[Vector2]) -> float | None:
        if len(positions) < 2:
            return None
        min_dist = float("inf")
        for i in range(len(positions)):
            xi, yi = positions[i]
            for j in range(i + 1, len(positions)):
                xj, yj = positions[j]
                d = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _evaluate_binary_connectivity(
        self,
        positions: list[Vector2],
    ) -> tuple[float, float]:
        """Evaluate lambda_2 for two binary graphs (communication and conservative)."""
        n_agents = len(positions)
        if n_agents < 2:
            return 0.0, 0.0

        adj_comm = np.zeros((n_agents, n_agents), dtype=float)
        adj_cons = np.zeros((n_agents, n_agents), dtype=float)
        d_cons = self.fast_d_c - self.fast_epsilon_c
        for i in range(n_agents):
            xi, yi = positions[i]
            for j in range(i + 1, n_agents):
                xj, yj = positions[j]
                dist = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < self.fast_d_c:
                    adj_comm[i, j] = 1.0
                    adj_comm[j, i] = 1.0
                if dist <= d_cons:
                    adj_cons[i, j] = 1.0
                    adj_cons[j, i] = 1.0

        def _lambda2_from_adj(adj: np.ndarray) -> float:
            lap = np.diag(adj.sum(axis=1)) - adj
            eigvals = np.linalg.eigvalsh(lap)
            return float(eigvals[1]) if eigvals.shape[0] >= 2 else 0.0

        return _lambda2_from_adj(adj_comm), _lambda2_from_adj(adj_cons)

    def _print_infeasibility_diagnostics(self, diagnostics: dict[str, object]) -> None:
        step = diagnostics.get("step_idx")
        agent_id = diagnostics.get("agent_id")
        print(f"[QP INFEASIBLE] first infeasible step={step}, agent={agent_id}")
        print(f"[QP INFEASIBLE] solver_status={diagnostics.get('solver_status')}")
        positions = diagnostics.get("positions")
        if isinstance(positions, dict):
            print(f"[QP INFEASIBLE] positions={positions}")
        pairwise_distances = diagnostics.get("pairwise_distances")
        if isinstance(pairwise_distances, list):
            print(f"[QP INFEASIBLE] pairwise_distances={pairwise_distances}")
        print(f"[QP INFEASIBLE] lambda2={diagnostics.get('lambda_2')}")
        print(f"[QP INFEASIBLE] v2={diagnostics.get('v2')}")
        print(f"[QP INFEASIBLE] decision_dim={diagnostics.get('decision_dim')}")
        constraint_counts = diagnostics.get("constraint_counts")
        if isinstance(constraint_counts, dict):
            print(f"[QP INFEASIBLE] constraint_counts={constraint_counts}")
        print(f"[QP INFEASIBLE] beta_i={diagnostics.get('beta_i')}")
        fast_states = diagnostics.get("fast_states")
        if isinstance(fast_states, dict):
            print(f"[QP INFEASIBLE] fast_states={fast_states}")
        block_feasibility = diagnostics.get("block_feasibility")
        if isinstance(block_feasibility, dict):
            print(f"[QP INFEASIBLE] block_feasibility={block_feasibility}")
        row_labels = diagnostics.get("row_labels")
        A = diagnostics.get("A")
        b = diagnostics.get("b")
        if isinstance(row_labels, list) and isinstance(A, list) and isinstance(b, list):
            for label, a_row, b_val in zip(row_labels, A, b):
                print(
                    f"[QP INFEASIBLE ROW] label={label}, "
                    f"A={a_row}, b={b_val}"
                )
        row_decomposition = diagnostics.get("row_decomposition")
        if isinstance(row_decomposition, dict):
            pairwise = row_decomposition.get("pairwise")
            if isinstance(pairwise, list):
                for row in pairwise:
                    if not isinstance(row, dict):
                        continue
                    print(
                        "[QP INFEASIBLE PAIRWISE] "
                        f"label={row.get('label')}, h_ij={row.get('h_ij')}, "
                        f"raw_mismatch={row.get('raw_mismatch')}, "
                        f"weighted_mismatch={row.get('weighted_mismatch')}, "
                        f"geometric_term={row.get('geometric_term')}, "
                        f"weighted_geometric={row.get('weighted_geometric_term')}, "
                        f"final_A={row.get('final_A')}, final_b={row.get('final_b')}, "
                        f"dominant={row.get('dominant_term')}"
                    )
            clf = row_decomposition.get("clf")
            if isinstance(clf, dict):
                print(
                    "[QP INFEASIBLE CLF] "
                    f"label={clf.get('label')}, V_i={clf.get('V_i')}, h_i={clf.get('h_i')}, "
                    f"W_i={clf.get('W_i')}, gradient_term={clf.get('gradient_term_for_A')}, "
                    f"gamma3W={clf.get('gamma3W')}, coupling_sum={clf.get('coupling_sum')}, "
                    f"final_A={clf.get('final_A')}, final_b={clf.get('final_b')}, "
                    f"dominant={clf.get('dominant_term')}"
                )
                coupling_terms = clf.get("coupling_terms")
                if isinstance(coupling_terms, list):
                    for term in coupling_terms:
                        if not isinstance(term, dict):
                            continue
                        print(
                            "[QP INFEASIBLE CLF TERM] "
                            f"neighbor={term.get('neighbor')}, h_ij={term.get('h_ij')}, "
                            f"h_i={term.get('h_i')}, h_j={term.get('h_j')}, "
                            f"raw_mismatch={term.get('raw_mismatch')}, "
                            f"weighted_term={term.get('weighted_term')}"
                        )
            connectivity = row_decomposition.get("connectivity")
            if isinstance(connectivity, dict):
                print(
                    "[QP INFEASIBLE CONNECTIVITY] "
                    f"label={connectivity.get('label')}, beta_i={connectivity.get('beta_i')}, "
                    f"lambda2_over_N={connectivity.get('lambda2_over_N')}, "
                    f"eta_over_N={connectivity.get('eta_over_N')}, "
                    f"raw_connectivity_coupling_sum={connectivity.get('raw_connectivity_coupling_sum')}, "
                    f"alpha_connectivity={connectivity.get('alpha_connectivity')}, "
                    f"scaled_connectivity_coupling_sum={connectivity.get('scaled_connectivity_coupling_sum')}, "
                    f"final_A={connectivity.get('final_A')}, final_b={connectivity.get('final_b')}, "
                    f"dominant={connectivity.get('dominant_term')}"
                )
                conn_terms = connectivity.get("coupling_terms")
                if isinstance(conn_terms, list):
                    for term in conn_terms:
                        if not isinstance(term, dict):
                            continue
                        print(
                            "[QP INFEASIBLE CONNECTIVITY TERM] "
                            f"neighbor={term.get('neighbor')}, h_ij={term.get('h_ij')}, "
                            f"raw_mismatch={term.get('raw_mismatch')}, "
                            f"weighted_term={term.get('weighted_term')}"
                        )

    def run(self) -> SimulationResult:
        """Run the simulation and return trajectories plus metric time-series."""

        trajectories: list[list[Vector2]] = []
        min_obstacle_clearance: list[float | None] = []
        min_inter_agent_distance: list[float | None] = []
        first_target_entry_step: dict[int, int | None] = {
            agent.agent_id: None for agent in self.agents
        }
        control_history = self._init_control_history()
        fast_history = self._init_fast_history()
        lambda2_comm_history: list[float] = []
        lambda2_cons_history: list[float] = []
        stopped_early = False
        infeasible_step: int | None = None
        infeasible_agent_id: int | None = None
        infeasible_diagnostics: dict[str, object] | None = None
        infeasible_agent_ids: list[int] = []
        termination_reason = "completed"
        termination_step: int | None = None
        local_qp_failure_kind: str | None = None
        qp_failures: list[dict[str, Any]] = []
        qp_failure_diagnostics: dict[int, dict[str, object]] = {}
        unapplied_controls: dict[int, Vector2] = {}
        termination_exception: dict[str, str] | None = None
        controls: dict[int, Vector2] = {}
        agents_updated = 0
        fast_states_step_4_snapshot: dict[int, FastState] | None = None

        positions = self.env.agent_positions()
        trajectories.append(positions)
        lambda2_comm_0, lambda2_cons_0 = self._evaluate_binary_connectivity(positions)
        lambda2_comm_history.append(lambda2_comm_0)
        lambda2_cons_history.append(lambda2_cons_0)
        min_obstacle_clearance.append(self._min_obstacle_clearance(positions))
        min_inter_agent_distance.append(self._min_inter_agent_distance(positions))
        self._update_first_target_entry(first_target_entry_step, step_idx=0)
        self._debug_print_target_phi(step_idx=0)
        self._record_fast_history_snapshot(fast_history)

        step_idx = -1
        try:
            for step_idx in range(self.steps):
                # a) Compute all control inputs for the current slow state.
                controls = {}
                agents_updated = 0
                all_agent_states = self._agent_state_map()
                prev_fast_states = self.fast_states
                _, _, lambda_2_runtime, v2_runtime = self._compute_connectivity_runtime_spectrum(
                    all_agent_states
                )
                print(
                    f"[DEBUG spectrum] step={step_idx}, lambda2={lambda_2_runtime:.6f}, "
                    f"v2={[(i, v2_runtime[i]) for i in sorted(v2_runtime)]}"
                    )
                compute_controls = getattr(self.controller, "compute_controls", None)
                if callable(compute_controls):
                    try:
                        controls = compute_controls(
                            self.env,
                            step_idx=step_idx,
                            all_fast_states=prev_fast_states,
                            all_agent_states=all_agent_states,
                            d_c=self.fast_d_c,
                            gamma_1=self.fast_gamma_1,
                            d_0=self.fast_d_0,
                            target_center=self.fast_target_center,
                            sigma=self.fast_sigma,
                            epsilon=self.fast_epsilon,
                            n_agents=len(self.agents),
                            eta=self.fast_eta,
                            epsilon_c=self.fast_epsilon_c,
                            lambda_2=lambda_2_runtime,
                            v2=v2_runtime,
                            xi=self.fast_xi,
                        )
                    except CentralizedQPInfeasibleError as exc:
                        stopped_early = True
                        infeasible_step = step_idx if exc.step_idx is None else int(exc.step_idx)
                        infeasible_agent_id = None
                        infeasible_diagnostics = exc.diagnostics
                        termination_reason = "centralized_qp_failure"
                        termination_step = step_idx
                        qp_failures = [{
                            "agent_id": None,
                            "step": step_idx,
                            "solver_status": _solver_status_from_reason(exc.diagnostics.get("solver_status")),
                            "solver_reason": str(exc.diagnostics.get("solver_status")),
                        }]
                        self._print_infeasibility_diagnostics(exc.diagnostics)
                        break
                    for agent in self.agents:
                        ux, uy = controls[agent.agent_id]
                        u_norm = sqrt(ux * ux + uy * uy)
                        print(
                            f"[DEBUG control] step={step_idx}, agent={agent.agent_id}, "
                            f"u=({ux:.6f}, {uy:.6f}), norm={u_norm:.6f}"
                        )
                else:
                    # Solve every agent's local QP at this step, then stop if any failed.
                    # No fallback input is applied and the state is not advanced.
                    step_failures: list[LocalQPInfeasibleError] = []
                    for agent in self.agents:
                        try:
                            controls[agent.agent_id] = self.controller.compute_control(
                                agent,
                                self.env,
                                step_idx=step_idx,
                                fast_state=prev_fast_states[agent.agent_id],
                                all_fast_states=prev_fast_states,
                                all_agent_states=all_agent_states,
                                d_c=self.fast_d_c,
                                gamma_1=self.fast_gamma_1,
                                d_0=self.fast_d_0,
                                target_center=self.fast_target_center,
                                sigma=self.fast_sigma,
                                epsilon=self.fast_epsilon,
                                n_agents=len(self.agents),
                                eta=self.fast_eta,
                                epsilon_c=self.fast_epsilon_c,
                                lambda_2=lambda_2_runtime,
                                v2=v2_runtime,
                            )
                        except LocalQPInfeasibleError as exc:
                            step_failures.append(exc)
                            self._print_infeasibility_diagnostics(exc.diagnostics)
                            continue
                        ux, uy = controls[agent.agent_id]
                        u_norm = sqrt(ux * ux + uy * uy)
                        print(
                            f"[DEBUG control] step={step_idx}, agent={agent.agent_id}, "
                            f"u=({ux:.6f}, {uy:.6f}), norm={u_norm:.6f}"
                        )
                    if step_failures:
                        first = step_failures[0]
                        stopped_early = True
                        infeasible_step = step_idx if first.step_idx is None else int(first.step_idx)
                        infeasible_agent_id = int(first.agent_id)
                        infeasible_agent_ids = [int(exc.agent_id) for exc in step_failures]
                        infeasible_diagnostics = first.diagnostics
                        termination_reason = "local_qp_failure"
                        termination_step = step_idx
                        for exc in step_failures:
                            status = _solver_status_from_reason(exc.diagnostics.get("solver_status"))
                            qp_failures.append({
                                "agent_id": int(exc.agent_id),
                                "step": step_idx,
                                "solver_status": status,
                                "solver_reason": str(exc.diagnostics.get("solver_status")),
                                "failure_kind": (
                                    "solver_reported_infeasible"
                                    if status in SOLVER_INFEASIBLE_STATUSES
                                    else "solver_numerical_failure"
                                ),
                            })
                            qp_failure_diagnostics[int(exc.agent_id)] = exc.diagnostics
                        kinds = {f["failure_kind"] for f in qp_failures}
                        local_qp_failure_kind = kinds.pop() if len(kinds) == 1 else "mixed"
                if stopped_early:
                    break

                # b) Update all fast states (synchronously).
                if getattr(self.controller, "uses_fast_dynamics", True):
                    updated_fast_states: dict[int, FastState] = {}
                    for agent in self.agents:
                        updated_fast_states[agent.agent_id] = update_fast_state(
                            i=agent.agent_id,
                            state=prev_fast_states[agent.agent_id],
                            step_idx=step_idx,
                            all_fast_states=prev_fast_states,
                            all_agent_states=all_agent_states,
                            dt=self.dt,
                            tau=self.fast_tau,
                            xi=self.fast_xi,
                            d_c=self.fast_d_c,
                            target_center=self.fast_target_center,
                            sigma=self.fast_sigma,
                            epsilon=self.fast_epsilon,
                            gamma_1=self.fast_gamma_1,
                            gamma_2=self.fast_gamma_1,
                            gamma_3=getattr(self.controller, "config").clf_rate,
                            d_0=self.fast_d_0,
                            alpha_clf=getattr(self.controller, "config").alpha_clf,
                            x_i=all_agent_states[agent.agent_id].position,
                            u_i=controls[agent.agent_id],
                            obstacles=self.env.obstacles,
                            u_max=getattr(self.controller, "config").u_max,
                            eta=self.fast_eta,
                            epsilon_c=self.fast_epsilon_c,
                            lambda_2=lambda_2_runtime,
                            v2=v2_runtime,
                            n_agents=len(self.agents),
                        )
                    for agent in self.agents:
                        self._check_fast_state_explosion(
                            step_idx=step_idx,
                            agent_id=agent.agent_id,
                            state=updated_fast_states[agent.agent_id],
                        )
                    self.fast_states = updated_fast_states
                    if step_idx == 4:
                        fast_states_step_4_snapshot = copy.deepcopy(self.fast_states)
                        sample_agent_id = self.agents[0].agent_id if self.agents else None
                        print("[DEBUG fast compare] stored deepcopy snapshot at step=4")
                        print(
                            f"[DEBUG fast compare] step=4 snapshot_id={id(fast_states_step_4_snapshot)}, "
                            f"current_id={id(self.fast_states)}"
                        )
                        if sample_agent_id is not None:
                            snapshot_agent = fast_states_step_4_snapshot[sample_agent_id]
                            current_agent = self.fast_states[sample_agent_id]
                            print(
                                f"[DEBUG fast compare] step=4 sample_agent={sample_agent_id}, "
                                f"snapshot_agent_id={id(snapshot_agent)}, current_agent_id={id(current_agent)}"
                            )
                            if snapshot_agent.z_pair:
                                sample_neighbor_id = next(iter(snapshot_agent.z_pair))
                                print(
                                    f"[DEBUG fast compare] step=4 sample_agent={sample_agent_id}, "
                                    f"sample_neighbor={sample_neighbor_id}, "
                                    f"snapshot_z_pair_id={id(snapshot_agent.z_pair[sample_neighbor_id])}, "
                                    f"current_z_pair_id={id(current_agent.z_pair[sample_neighbor_id])}, "
                                    f"snapshot_y_pair_id={id(snapshot_agent.y_pair[sample_neighbor_id])}, "
                                    f"current_y_pair_id={id(current_agent.y_pair[sample_neighbor_id])}"
                                )
                    elif step_idx == 5 and fast_states_step_4_snapshot is not None:
                        print("[DEBUG fast compare] comparing step=4 snapshot against current step=5 fast states")
                        print(
                            f"[DEBUG fast compare] step=5 snapshot_id={id(fast_states_step_4_snapshot)}, "
                            f"current_id={id(self.fast_states)}"
                        )
                        sample_agent_id = self.agents[0].agent_id if self.agents else None
                        if sample_agent_id is not None:
                            snapshot_agent = fast_states_step_4_snapshot[sample_agent_id]
                            current_agent = self.fast_states[sample_agent_id]
                            print(
                                f"[DEBUG fast compare] step=5 sample_agent={sample_agent_id}, "
                                f"snapshot_agent_id={id(snapshot_agent)}, current_agent_id={id(current_agent)}"
                            )
                            if snapshot_agent.z_pair:
                                sample_neighbor_id = next(iter(snapshot_agent.z_pair))
                                print(
                                    f"[DEBUG fast compare] step=5 sample_agent={sample_agent_id}, "
                                    f"sample_neighbor={sample_neighbor_id}, "
                                    f"snapshot_z_pair_id={id(snapshot_agent.z_pair[sample_neighbor_id])}, "
                                    f"current_z_pair_id={id(current_agent.z_pair[sample_neighbor_id])}, "
                                    f"snapshot_y_pair_id={id(snapshot_agent.y_pair[sample_neighbor_id])}, "
                                    f"current_y_pair_id={id(current_agent.y_pair[sample_neighbor_id])}"
                                )

                        for agent in self.agents:
                            agent_id = agent.agent_id
                            prev_state = fast_states_step_4_snapshot[agent_id]
                            curr_state = self.fast_states[agent_id]
                            w_delta = (
                                curr_state.w[0] - prev_state.w[0],
                                curr_state.w[1] - prev_state.w[1],
                            )
                            print(f"[DEBUG fast compare] agent={agent_id}")
                            print(
                                f"  w: step4={prev_state.w}, step5={curr_state.w}, delta={w_delta}"
                            )
                            print(
                                f"  z_c: step4={prev_state.z_c:.12f}, "
                                f"step5={curr_state.z_c:.12f}, "
                                f"delta={curr_state.z_c - prev_state.z_c:.12f}"
                            )
                            print(
                                f"  z_clf: step4={prev_state.z_clf:.12f}, "
                                f"step5={curr_state.z_clf:.12f}, "
                                f"delta={curr_state.z_clf - prev_state.z_clf:.12f}"
                            )
                            print(
                                f"  y_c: step4={prev_state.y_c:.12f}, "
                                f"step5={curr_state.y_c:.12f}, "
                                f"delta={curr_state.y_c - prev_state.y_c:.12f}"
                            )
                            print(
                                f"  y_clf: step4={prev_state.y_clf:.12f}, "
                                f"step5={curr_state.y_clf:.12f}, "
                                f"delta={curr_state.y_clf - prev_state.y_clf:.12f}"
                            )

                            for neighbor_id in sorted(prev_state.z_pair):
                                prev_z_edge = prev_state.z_pair[neighbor_id]
                                curr_z_edge = curr_state.z_pair[neighbor_id]
                                prev_y_edge = prev_state.y_pair[neighbor_id]
                                curr_y_edge = curr_state.y_pair[neighbor_id]
                                print(f"  neighbor={neighbor_id}")
                                print(
                                    f"    z_pair[{neighbor_id}].self_component: "
                                    f"step4={prev_z_edge.self_component:.12f}, "
                                    f"step5={curr_z_edge.self_component:.12f}, "
                                    f"delta={curr_z_edge.self_component - prev_z_edge.self_component:.12f}"
                                )
                                print(
                                    f"    z_pair[{neighbor_id}].neighbor_component: "
                                    f"step4={prev_z_edge.neighbor_component:.12f}, "
                                    f"step5={curr_z_edge.neighbor_component:.12f}, "
                                    f"delta={curr_z_edge.neighbor_component - prev_z_edge.neighbor_component:.12f}"
                                )
                                print(
                                    f"    y_pair[{neighbor_id}].self_component: "
                                    f"step4={prev_y_edge.self_component:.12f}, "
                                    f"step5={curr_y_edge.self_component:.12f}, "
                                    f"delta={curr_y_edge.self_component - prev_y_edge.self_component:.12f}"
                                )
                                print(
                                    f"    y_pair[{neighbor_id}].neighbor_component: "
                                    f"step4={prev_y_edge.neighbor_component:.12f}, "
                                    f"step5={curr_y_edge.neighbor_component:.12f}, "
                                    f"delta={curr_y_edge.neighbor_component - prev_y_edge.neighbor_component:.12f}"
                                )
                self._record_fast_history_snapshot(fast_history)

                # c) Update all agent states.
                for agent in self.agents:
                    agent.euler_step(u_k=controls[agent.agent_id], dt=self.dt)
                    control_history[agent.agent_id].append(controls[agent.agent_id])
                    agents_updated += 1

                positions = self.env.agent_positions()
                trajectories.append(positions)
                lambda2_comm_k, lambda2_cons_k = self._evaluate_binary_connectivity(positions)
                lambda2_comm_history.append(lambda2_comm_k)
                lambda2_cons_history.append(lambda2_cons_k)
                min_obstacle_clearance.append(self._min_obstacle_clearance(positions))
                min_inter_agent_distance.append(self._min_inter_agent_distance(positions))
                self._update_first_target_entry(first_target_entry_step, step_idx=step_idx + 1)
                self._debug_print_target_phi(step_idx=step_idx + 1)

        except Exception as exc:  # noqa: BLE001 - recorded, partial results returned
            # The state was not advanced at step_idx; everything recorded so far is real.
            stopped_early = True
            termination_reason = "numerical_guard" if isinstance(exc, NumericalGuardError) else "exception"
            termination_step = step_idx
            termination_exception = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                # Agents whose state was already advanced at termination_step when the
                # exception occurred (0 unless it was raised during/after the Euler update).
                "agents_state_advanced_at_step": str(agents_updated),
            }
            print(f"[SIMULATION ABORTED] step={step_idx}, reason={termination_reason}: {type(exc).__name__}: {exc}")

        steps_completed = len(trajectories) - 1
        if termination_reason != "completed":
            applied_now = {a.agent_id for a in self.agents[:agents_updated]}
            unapplied_controls = {i: controls[i] for i in sorted(controls) if i not in applied_now}

        return SimulationResult(
            trajectories=trajectories,
            min_obstacle_clearance=min_obstacle_clearance,
            min_inter_agent_distance=min_inter_agent_distance,
            first_target_entry_step=first_target_entry_step,
            control_history=control_history,
            fast_w_norm=fast_history["w_norm"],
            fast_z_c=fast_history["z_c"],
            fast_z_clf=fast_history["z_clf"],
            fast_y_c=fast_history["y_c"],
            fast_y_clf=fast_history["y_clf"],
            fast_max_abs_z_pair_self=fast_history["max_abs_z_pair_self"],
            fast_max_abs_y_pair_self=fast_history["max_abs_y_pair_self"],
            lambda2_comm=lambda2_comm_history,
            lambda2_cons=lambda2_cons_history,
            stopped_early=stopped_early,
            infeasible_step=infeasible_step,
            infeasible_agent_id=infeasible_agent_id,
            infeasible_diagnostics=infeasible_diagnostics,
            infeasible_agent_ids=infeasible_agent_ids,
            termination_reason=termination_reason,
            termination_step=termination_step,
            steps_configured=self.steps,
            steps_completed=steps_completed,
            local_qp_failure_kind=local_qp_failure_kind,
            qp_failures=qp_failures,
            qp_failure_diagnostics=qp_failure_diagnostics,
            unapplied_controls=unapplied_controls,
            termination_exception=termination_exception,
        )

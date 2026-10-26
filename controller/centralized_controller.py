"""Centralized controller that solves one joint CLF-CBF-QP for all agents."""

from __future__ import annotations

from config import ControllerConfig
from controller.base_controller import Controller
from controller.centralized_qp import (
    analyze_block_feasibility,
    build_centralized_qp_data,
    solve_centralized_qp_with_status,
)
from models.agent import AgentState, Vector2
from models.environment import Environment
from models.fast_state import FastState


class CentralizedQPInfeasibleError(RuntimeError):
    """Raised when the centralized QP fails."""

    def __init__(self, step_idx: int | None, diagnostics: dict[str, object]):
        self.step_idx = step_idx
        self.diagnostics = diagnostics
        super().__init__(f"Centralized QP infeasible at step={step_idx}.")


class CentralizedController(Controller):
    """Joint controller over all controls and mismatch variables."""

    uses_fast_dynamics = False

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.centralized_debug_records: dict[int, dict[str, object]] = {}

    def compute_control(self, agent: AgentState, env: Environment, **kwargs: object) -> Vector2:
        raise NotImplementedError(
            "CentralizedController uses compute_controls(...) and should be called by the simulator in joint mode."
        )

    def compute_controls(self, env: Environment, **kwargs: object) -> dict[int, Vector2]:
        all_agent_states = kwargs.get("all_agent_states")
        all_fast_states = kwargs.get("all_fast_states")
        d_c = kwargs.get("d_c")
        step_idx = kwargs.get("step_idx")
        if not isinstance(all_agent_states, dict) or not isinstance(all_fast_states, dict):
            raise ValueError("compute_controls requires all_agent_states and all_fast_states.")
        if not isinstance(d_c, (int, float)):
            raise ValueError("compute_controls requires d_c.")

        qp_data = build_centralized_qp_data(
            all_agent_states=all_agent_states,
            all_fast_states=all_fast_states,
            obstacles=env.obstacles,
            u_max=self.config.u_max,
            xi=float(kwargs.get("xi", 1.0)),
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
            enable_connectivity_constraint=self.config.enable_connectivity_constraint,
            alpha_clf=self.config.alpha_clf,
            alpha_connectivity=self.config.alpha_connectivity,
            lambda_2=float(kwargs["lambda_2"]) if "lambda_2" in kwargs else None,
            v2=kwargs["v2"] if "v2" in kwargs else None,  # type: ignore[arg-type]
            clf_coupling=self.config.clf_coupling,
        )

        decision, solve_ok, reason = solve_centralized_qp_with_status(qp_data)
        debug_record = {
            "solve_ok": solve_ok,
            "solver_status": reason,
            "decision_dim": len(qp_data.q),
            "control_dim": qp_data.control_dim,
            "z_dim": qp_data.z_dim,
            "constraint_counts": dict(qp_data.block_counts),
        }
        if isinstance(step_idx, int):
            self.centralized_debug_records[step_idx] = debug_record

        if not solve_ok:
            diagnostics: dict[str, object] = {
                "step_idx": step_idx,
                "controller_mode": "centralized",
                "solver_status": reason,
                "decision_dim": len(qp_data.q),
                "control_dim": qp_data.control_dim,
                "z_dim": qp_data.z_dim,
                "constraint_counts": dict(qp_data.block_counts),
                "block_feasibility": analyze_block_feasibility(qp_data),
                "row_labels": list(qp_data.row_labels),
                "row_blocks": list(qp_data.row_blocks),
                "A": [list(row) for row in qp_data.A],
                "b": list(qp_data.b),
                "decision_labels": list(qp_data.decision_labels),
                "positions": {
                    int(agent_id): {
                        "position": (
                            float(agent_state.position[0]),
                            float(agent_state.position[1]),
                        )
                    }
                    for agent_id, agent_state in all_agent_states.items()
                },
                "lambda_2": float(kwargs["lambda_2"]) if "lambda_2" in kwargs else None,
                "v2": (
                    {int(k): float(v) for k, v in kwargs["v2"].items()}
                    if "v2" in kwargs and isinstance(kwargs["v2"], dict)
                    else None
                ),
            }
            if isinstance(step_idx, int):
                self.centralized_debug_records[step_idx] = {
                    **debug_record,
                    "infeasible_diagnostics": diagnostics,
                }
            raise CentralizedQPInfeasibleError(
                step_idx=(int(step_idx) if isinstance(step_idx, int) else None),
                diagnostics=diagnostics,
            )

        controls: dict[int, Vector2] = {}
        agent_ids = sorted(all_agent_states.keys())
        for offset, agent_id in enumerate(agent_ids):
            ux = float(decision[2 * offset])
            uy = float(decision[2 * offset + 1])
            controls[agent_id] = (ux, uy)

        if isinstance(step_idx, int):
            z_solution = decision[qp_data.control_dim:].tolist()
            self.centralized_debug_records[step_idx] = {
                **debug_record,
                "decision_preview": {
                    "u": {int(i): (float(controls[i][0]), float(controls[i][1])) for i in controls},
                    "z_norm_inf": max((abs(float(v)) for v in z_solution), default=0.0),
                },
            }
        return controls

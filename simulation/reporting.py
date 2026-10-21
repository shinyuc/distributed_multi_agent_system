"""Run-level reporting: horizon completion, termination, and interval-qualified statements.

Every statement about safety, connectivity, or the task is evaluated only on the
states that were actually simulated, ``trajectories[0..steps_completed]``, and is
tagged with that interval and with whether it covers the configured horizon.
"""

from __future__ import annotations

from math import dist
from typing import Any

import numpy as np

from simulation.simulator import SimulationResult


def _inside_target(p, center, sigma) -> bool:
    dx, dy = p[0] - center[0], p[1] - center[1]
    v = dx * (sigma[0][0] * dx + sigma[0][1] * dy) + dy * (sigma[1][0] * dx + sigma[1][1] * dy)
    return v <= 1.0


def _connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if n <= 1:
        return True
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adj[i]):
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == n


def build_run_report(
    result: SimulationResult,
    *,
    dt: float,
    safety_distance: float,
    d_c: float,
    epsilon_c: float,
    target_center,
    target_sigma,
) -> dict[str, Any]:
    traj = result.trajectories
    steps_completed = len(traj) - 1
    t_sim = steps_completed * dt
    t_cfg = result.steps_configured * dt
    full = result.horizon_completed
    interval = {"t_start": 0.0, "t_end": t_sim, "covers_configured_horizon": full}

    min_d, min_d_step, first_violation = np.inf, None, None
    min_clear = np.inf
    comm_all, cons_all = True, True
    for k, pos in enumerate(traj):
        n = len(pos)
        adj_comm = np.zeros((n, n), dtype=bool)
        adj_cons = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(pos[i], pos[j])
                if d < min_d:
                    min_d, min_d_step = d, k
                if d < safety_distance and first_violation is None:
                    first_violation = {"step": k, "t": k * dt, "pair": [i, j], "distance": d}
                adj_comm[i, j] = adj_comm[j, i] = d < d_c
                adj_cons[i, j] = adj_cons[j, i] = d <= d_c - epsilon_c
        comm_all &= _connected(adj_comm)
        cons_all &= _connected(adj_cons)
    for v in result.min_obstacle_clearance:
        if v is not None:
            min_clear = min(min_clear, v)

    final = traj[-1]
    inside_final = {i: _inside_target(p, target_center, target_sigma) for i, p in enumerate(final)}
    entered = {int(i): s for i, s in result.first_target_entry_step.items()}

    if full:
        verdict = f"COMPLETED full horizon: {result.steps_configured} steps (t = {t_cfg:g} s)"
    else:
        reason = result.termination_reason
        if reason == "local_qp_failure":
            agents = [f["agent_id"] for f in result.qp_failures]
            statuses = sorted({f["solver_status"] for f in result.qp_failures})
            reason = f"local QP failure ({result.local_qp_failure_kind}; agents {agents}; status {', '.join(statuses)})"
        elif result.termination_exception:
            reason = f"{reason}: {result.termination_exception['type']}"
        verdict = (
            f"TERMINATED EARLY at step {result.termination_step} (t = {t_sim:g} s of {t_cfg:g} s; "
            f"{steps_completed}/{result.steps_configured} steps applied) - {reason}"
        )

    return {
        "verdict": verdict,
        "horizon": {
            "horizon_completed": full,
            "steps_configured": result.steps_configured,
            "steps_completed": steps_completed,
            "t_configured": t_cfg,
            "t_simulated": t_sim,
        },
        "termination": {
            "reason": result.termination_reason,
            "step": result.termination_step,
            "t": None if result.termination_step is None else result.termination_step * dt,
            "local_qp_failure_kind": result.local_qp_failure_kind,
            "qp_failures": result.qp_failures,
            "unapplied_controls": {str(k): list(v) for k, v in result.unapplied_controls.items()},
            "exception": result.termination_exception,
        },
        # All statements below hold only on `interval`.
        "simulated_interval": interval,
        "safety": {
            "interval": interval,
            "min_inter_agent_distance": None if not np.isfinite(min_d) else float(min_d),
            "min_inter_agent_distance_step": min_d_step,
            "safety_distance": safety_distance,
            "inter_agent_violation": first_violation is not None,
            "first_inter_agent_violation": first_violation,
            "min_obstacle_clearance": None if not np.isfinite(min_clear) else float(min_clear),
            "obstacle_violation": bool(np.isfinite(min_clear) and min_clear < 0.0),
        },
        "connectivity": {
            "interval": interval,
            "comm_graph_connected_at_all_simulated_samples": comm_all,
            "conservative_graph_connected_at_all_simulated_samples": cons_all,
            "min_lambda2_comm": float(min(result.lambda2_comm)) if result.lambda2_comm else None,
            "min_lambda2_cons": float(min(result.lambda2_cons)) if result.lambda2_cons else None,
        },
        "task": {
            "interval": interval,
            "first_target_entry_step": entered,
            "all_agents_entered_target_at_some_time": all(v is not None for v in entered.values()),
            "inside_target_at_final_simulated_state": inside_final,
            "all_agents_inside_target_at_final_simulated_state": all(inside_final.values()),
            # Task success requires the full horizon to have been simulated.
            "task_completed_and_horizon_completed": full and all(inside_final.values()),
        },
    }


def interval_note(report: dict[str, Any]) -> str:
    h = report["horizon"]
    note = f"over simulated [0, {h['t_simulated']:g}] s"
    if not h["horizon_completed"]:
        note += f" of configured [0, {h['t_configured']:g}] s - HORIZON NOT COMPLETED"
    return note


def short_status(report: dict[str, Any]) -> str:
    """One line suitable for a plot annotation."""
    h, term = report["horizon"], report["termination"]
    if h["horizon_completed"]:
        return f"Completed full horizon (t = {h['t_configured']:g} s)"
    label = term["reason"]
    if label == "local_qp_failure":
        agents = [f["agent_id"] for f in term["qp_failures"]]
        label = f"local QP failure: {term['local_qp_failure_kind']}, agents {agents}"
    return f"TERMINATED at t = {h['t_simulated']:g} s of {h['t_configured']:g} s ({label})"

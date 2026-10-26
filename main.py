"""Main entry point for running simulations and saving artifacts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from config import load_config
from controller.controller_factory import build_controller
from controller.local_qp import local_qp_solver_metadata
from models.agent import AgentState
from models.environment import CircularObstacle, Environment, TargetRegion
from simulation.reporting import build_run_report, interval_note, short_status
from simulation.simulator import Simulator
from utils.run_dirs import OutputDirExistsError, make_unique_run_dir, prepare_explicit_dir
from visualization.plotter import plot_trajectories


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def _overall_min(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None]
    return min(finite) if finite else None


def _find_global_min_inter_agent_distance(
    trajectories: list[list[tuple[float, float]]],
) -> tuple[float | None, int | None, tuple[int, int] | None]:
    min_dist: float | None = None
    min_step: int | None = None
    min_pair: tuple[int, int] | None = None
    for step_idx, positions in enumerate(trajectories):
        n = len(positions)
        for i in range(n):
            xi, yi = positions[i]
            for j in range(i + 1, n):
                xj, yj = positions[j]
                d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if min_dist is None or d < min_dist:
                    min_dist = d
                    min_step = step_idx
                    min_pair = (i, j)
    return min_dist, min_step, min_pair


def _print_pair_distance_window(
    trajectories: list[list[tuple[float, float]]],
    agent_i: int,
    agent_j: int,
    step_start: int,
    step_end: int,
) -> None:
    if not trajectories:
        print("pair distance window: trajectories are empty")
        return

    lo = max(0, step_start)
    hi = min(len(trajectories) - 1, step_end)
    print(
        f"pair distance window: agent pair=({agent_i}, {agent_j}), "
        f"steps={lo}..{hi}"
    )
    for step_idx in range(lo, hi + 1):
        positions = trajectories[step_idx]
        if agent_i >= len(positions) or agent_j >= len(positions):
            raise ValueError(
                f"Invalid agent indices for step {step_idx}: "
                f"len(positions)={len(positions)}, pair=({agent_i},{agent_j})"
            )
        xi, yi = positions[agent_i]
        xj, yj = positions[agent_j]
        d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
        print(f"step={step_idx}, pair=({agent_i},{agent_j}), distance={d:.6f}")


def _find_first_pair_violation(
    trajectories: list[list[tuple[float, float]]],
    agent_i: int,
    agent_j: int,
    safety_distance: float,
) -> tuple[int | None, float | None, float | None]:
    for step_idx, positions in enumerate(trajectories):
        if agent_i >= len(positions) or agent_j >= len(positions):
            raise ValueError(
                f"Invalid agent indices for step {step_idx}: "
                f"len(positions)={len(positions)}, pair=({agent_i},{agent_j})"
            )
        xi, yi = positions[agent_i]
        xj, yj = positions[agent_j]
        d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
        if d < safety_distance:
            prev_d: float | None = None
            if step_idx > 0:
                pxi, pyi = trajectories[step_idx - 1][agent_i]
                pxj, pyj = trajectories[step_idx - 1][agent_j]
                prev_d = ((pxi - pxj) ** 2 + (pyi - pyj) ** 2) ** 0.5
            return step_idx, d, prev_d
    return None, None, None


def _find_first_violation_any_pair(
    trajectories: list[list[tuple[float, float]]],
    safety_distance: float,
) -> tuple[int | None, tuple[int, int] | None, float | None]:
    for step_idx, positions in enumerate(trajectories):
        n = len(positions)
        for i in range(n):
            xi, yi = positions[i]
            for j in range(i + 1, n):
                xj, yj = positions[j]
                d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if d < safety_distance:
                    return step_idx, (i, j), d
    return None, None, None


def _print_pair_safety_diagnostics_window(
    *,
    trajectories: list[list[tuple[float, float]]],
    control_history: dict[int, list[tuple[float, float]]],
    qp_debug_records: dict[tuple[int, int], dict[str, object]],
    safety_distance: float,
    pair: tuple[int, int],
    center_step: int,
) -> None:
    i, j = pair
    lo = max(0, center_step - 5)
    hi = min(len(trajectories) - 1, center_step + 5)
    print(
        f"[DEBUG safety window] pair=({i},{j}), safety_distance={safety_distance:.6f}, "
        f"steps={lo}..{hi}"
    )
    for step in range(lo, hi + 1):
        xi, yi = trajectories[step][i]
        xj, yj = trajectories[step][j]
        dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
        h_safety = dist * dist - safety_distance * safety_distance
        print(
            f"[DEBUG safety] step={step}, pair=({i},{j}), "
            f"distance={dist:.6f}, h_safety={h_safety:.6f}"
        )

        for agent, other in ((i, j), (j, i)):
            rec = qp_debug_records.get((step, agent))
            label = f"pairwise(i={agent},j={other})"
            if rec is None:
                print(
                    f"[DEBUG safety qp] step={step}, agent={agent}, "
                    "record_missing=True"
                )
                continue

            row_labels = rec.get("row_labels", [])
            A = rec.get("A", [])
            b = rec.get("b", [])
            include = label in row_labels
            print(
                f"[DEBUG safety qp] step={step}, agent={agent}, label={label}, "
                f"included={include}, solve_ok={rec.get('solve_ok')}, "
                f"fallback_used={rec.get('fallback_used')}, "
                f"u_applied={rec.get('u_returned')}"
            )
            if include:
                idx = row_labels.index(label)
                a_row = A[idx]
                b_val = b[idx]
                u_ret = rec.get("u_returned")
                if isinstance(u_ret, tuple) and len(u_ret) == 2:
                    residual = float(a_row[0]) * float(u_ret[0]) + float(a_row[1]) * float(u_ret[1]) - float(b_val)
                else:
                    residual = float("nan")
                print(
                    f"[DEBUG safety qp row] step={step}, agent={agent}, "
                    f"A=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), "
                    f"b={float(b_val):.6f}, residual={residual:.6f}, "
                    f"solve_reason={rec.get('solve_reason')}"
                )

            z_pair = rec.get("z_pair_snapshot", {})
            if isinstance(z_pair, dict) and other in z_pair:
                z_self, z_neighbor = z_pair[other]
                print(
                    f"[DEBUG safety z] step={step}, agent={agent}, edge=({agent},{other}), "
                    f"z_self={float(z_self):.6f}, z_neighbor={float(z_neighbor):.6f}"
                )

        ui = control_history.get(i, [])
        uj = control_history.get(j, [])
        u_i = ui[step] if step < len(ui) else None
        u_j = uj[step] if step < len(uj) else None
        print(f"[DEBUG safety control] step={step}, u_{i}={u_i}, u_{j}={u_j}")


def _print_infeasible_rows_near_violation(
    *,
    qp_debug_records: dict[tuple[int, int], dict[str, object]],
    pair: tuple[int, int],
    violation_step: int,
) -> None:
    i, j = pair
    for step in range(0, violation_step + 1):
        for agent in (i, j):
            rec = qp_debug_records.get((step, agent))
            if rec is None:
                continue
            if rec.get("solve_ok"):
                continue
            print(
                f"[DEBUG infeasible qp] first at step={step}, agent={agent}, "
                f"reason={rec.get('solve_reason')}"
            )
            row_labels = rec.get("row_labels", [])
            A = rec.get("A", [])
            b = rec.get("b", [])
            for label, a_row, b_val in zip(row_labels, A, b):
                residual_zero = -float(b_val)
                print(
                    f"[DEBUG infeasible row] step={step}, agent={agent}, label={label}, "
                    f"A=({float(a_row[0]):.6f}, {float(a_row[1]):.6f}), "
                    f"b={float(b_val):.6f}, residual_zero={residual_zero:.6f}"
                )
            return


def _plot_connectivity_series(
    values: list[float],
    dt: float,
    title: str,
    ylabel: str,
    save_path: Path,
    t_configured: float | None = None,
    status_text: str | None = None,
    terminated_early: bool = False,
) -> None:
    times = [k * dt for k in range(len(values))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, values, linewidth=2, marker="o" if len(values) < 30 else None)
    if t_configured is not None and t_configured > 0:
        ax.set_xlim(0.0, t_configured)
        if terminated_early:
            t_end = times[-1] if times else 0.0
            ax.axvline(t_end, color="tab:red", linewidth=1.2, linestyle="--")
            ax.axvspan(t_end, t_configured, color="0.85", alpha=0.6, linewidth=0)
            ax.text(t_end, ax.get_ylim()[1], " terminated", color="tab:red", fontsize=8, va="top")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title + ("" if status_text is None else f"\n{status_text}"), fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _write_run_meta(out_dir: Path, meta: dict) -> None:
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-root", default="outputs",
        help="parent directory; each run gets a new <timestamp>_<config>_p<pid>_<rand> directory inside it",
    )
    parser.add_argument(
        "--out", default=None,
        help="explicit run directory (must be new or empty unless --overwrite)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="with --out: move an existing non-empty directory aside (renamed, never deleted)",
    )
    args = parser.parse_args()

    # Create the run directory before anything else so results can never land in a
    # directory that holds another run's files.
    moved_to = None
    try:
        if args.out is not None:
            out_dir, moved_to = prepare_explicit_dir(args.out, overwrite=args.overwrite)
        else:
            out_dir = make_unique_run_dir(args.output_root, Path(args.config).stem)
    except OutputDirExistsError as exc:
        print(f"[OUTPUT ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    run_meta = {
        "status": "running",
        "config_path": args.config,
        "output_dir": str(out_dir),
        "previous_contents_moved_to": None if moved_to is None else str(moved_to),
        "started_at": datetime.now().isoformat(timespec="microseconds"),
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "local_qp_solver": local_qp_solver_metadata(),
    }
    _write_run_meta(out_dir, run_meta)
    print(f"outputs: {out_dir}")

    cfg = load_config(args.config)
    if cfg.agents.initialization_log_path:
        shutil.copy2(cfg.agents.initialization_log_path, out_dir / "initial_conditions.json")
    # ######## check
    # print("[DEBUG main.py] enable_connectivity_constraint =", cfg.controller.enable_connectivity_constraint)
    # ########
    _set_seed(cfg.simulation.seed)

    agents = [
        AgentState(agent_id=i, position=tuple(pos), velocity=(0.0, 0.0))
        for i, pos in enumerate(cfg.agents.initial_positions)
    ]
    obstacles = [
        CircularObstacle(center=tuple(obs["center"]), radius=float(obs["radius"]))
        for obs in cfg.obstacles.obstacles
    ]
    target = TargetRegion(
        center=tuple(cfg.target.center),
        radius=cfg.target.radius,
        Sigma=cfg.target.Sigma,
    )
    env = Environment(agents=agents, obstacles=obstacles, target=target)

    controller = build_controller(cfg.controller)
    simulator = Simulator(
        env=env,
        controller=controller,
        dt=cfg.simulation.dt,
        steps=cfg.simulation.steps,
        fast_tau=cfg.fast_dynamics.tau,
        fast_substeps=cfg.fast_dynamics.substeps,
        fast_xi=cfg.fast_dynamics.xi,
        fast_d_c=cfg.fast_dynamics.d_c,
        fast_target_center=tuple(cfg.target.center),
        fast_sigma=cfg.target.Sigma,
        fast_epsilon=cfg.target.epsilon,
        fast_gamma_1=cfg.controller.cbf_rate,
        fast_d_0=cfg.controller.safety_distance,
        fast_eta=cfg.fast_dynamics.eta,
        fast_epsilon_c=cfg.fast_dynamics.epsilon_c,
    )

    result = simulator.run()
    report = build_run_report(
        result,
        dt=cfg.simulation.dt,
        safety_distance=cfg.controller.safety_distance,
        d_c=cfg.fast_dynamics.d_c,
        epsilon_c=cfg.fast_dynamics.epsilon_c,
        target_center=cfg.target.center,
        target_sigma=cfg.target.Sigma,
    )
    early = not report["horizon"]["horizon_completed"]
    note = interval_note(report)
    status = short_status(report)
    print("=" * 80)
    print(f"[RUN STATUS] {report['verdict']}")
    for failure in result.qp_failures:
        print(
            f"[RUN STATUS] failed QP: step={failure['step']}, agent={failure['agent_id']}, "
            f"solver_status={failure['solver_status']}, kind={failure.get('failure_kind')}"
        )
    if result.termination_exception is not None:
        print(f"[RUN STATUS] exception: {result.termination_exception['type']}: "
              f"{result.termination_exception['message']}")
    print("=" * 80)
    min_dist, min_step, min_pair = _find_global_min_inter_agent_distance(result.trajectories)
    safety_distance = cfg.controller.safety_distance
    violation = (min_dist is not None) and (min_dist < safety_distance)
    print(f"minimum inter-agent distance ({note}):", f"{min_dist:.6f}" if min_dist is not None else "N/A")
    print("safety distance:", f"{safety_distance:.6f}")
    print(f"violation happened ({note}):", violation)
    print("time step:", min_step)
    print("agent pair:", min_pair)
    first_any_step, first_any_pair, first_any_dist = _find_first_violation_any_pair(
        trajectories=result.trajectories,
        safety_distance=safety_distance,
    )
    print(f"first violation (any pair) step ({note}):", first_any_step)
    print("first violation (any pair) pair:", first_any_pair)
    print(
        "first violation (any pair) distance:",
        f"{first_any_dist:.6f}" if first_any_dist is not None else "N/A",
    )
    if (
        first_any_step is not None
        and first_any_pair is not None
        and hasattr(controller, "qp_debug_records")
    ):
        _print_pair_distance_window(
            trajectories=result.trajectories,
            agent_i=first_any_pair[0],
            agent_j=first_any_pair[1],
            step_start=first_any_step - 5,
            step_end=first_any_step + 5,
        )
        _print_pair_safety_diagnostics_window(
            trajectories=result.trajectories,
            control_history=result.control_history,
            qp_debug_records=controller.qp_debug_records,  # type: ignore[arg-type]
            safety_distance=safety_distance,
            pair=first_any_pair,
            center_step=first_any_step,
        )
        _print_infeasible_rows_near_violation(
            qp_debug_records=controller.qp_debug_records,  # type: ignore[arg-type]
            pair=first_any_pair,
            violation_step=first_any_step,
        )

    with (out_dir / "trajectories.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": args.config,
                "controller_mode": cfg.controller.mode,
                "initialization_log_path": cfg.agents.initialization_log_path,
                "dt": cfg.simulation.dt,
                "steps": cfg.simulation.steps,
                "fast_substeps": cfg.fast_dynamics.substeps,
                "steps_configured": result.steps_configured,
                "steps_completed": result.steps_completed,
                "horizon_completed": result.horizon_completed,
                "termination_reason": result.termination_reason,
                "termination_step": result.termination_step,
                "local_qp_failure_kind": result.local_qp_failure_kind,
                "qp_failures": result.qp_failures,
                "termination_exception": result.termination_exception,
                "stopped_early": result.stopped_early,
                "infeasible_step": result.infeasible_step,
                "infeasible_agent_id": result.infeasible_agent_id,
                "infeasible_agent_ids": result.infeasible_agent_ids,
                # control_history: applied inputs only (len == steps_completed per agent).
                "unapplied_controls_at_termination": {
                    str(k): list(v) for k, v in result.unapplied_controls.items()
                },
                "trajectories": result.trajectories,
                "min_obstacle_clearance": result.min_obstacle_clearance,
                "min_inter_agent_distance": result.min_inter_agent_distance,
                "control_history": result.control_history,
                "fast_w_norm": result.fast_w_norm,
                "fast_z_c": result.fast_z_c,
                "fast_z_clf": result.fast_z_clf,
                "fast_y_c": result.fast_y_c,
                "fast_y_clf": result.fast_y_clf,
                "fast_max_abs_z_pair_self": result.fast_max_abs_z_pair_self,
                "fast_max_abs_y_pair_self": result.fast_max_abs_y_pair_self,
                "lambda2_comm": result.lambda2_comm,
                "lambda2_cons": result.lambda2_cons,
            },
            f,
            indent=2,
        )
    with (out_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    if result.infeasible_diagnostics is not None:
        with (out_dir / "infeasible_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(result.infeasible_diagnostics, f, indent=2)
    if result.qp_failure_diagnostics:
        with (out_dir / "infeasible_diagnostics_all_agents.json").open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in result.qp_failure_diagnostics.items()}, f, indent=2, default=str)
    centralized_debug_records = getattr(controller, "centralized_debug_records", None)
    if isinstance(centralized_debug_records, dict):
        with (out_dir / "centralized_debug_records.json").open("w", encoding="utf-8") as f:
            json.dump(centralized_debug_records, f, indent=2)

    init_region_to_plot = None
    if cfg.agents.init_mode in {"random_region", "random_connected"}:
        init_region_to_plot = cfg.agents.init_region

    plot_trajectories(
        env,
        result.trajectories,
        init_region=init_region_to_plot,
        save_path=out_dir / "trajectories.png",
        show=False,
        status_text=status,
        terminated_early=early,
    )

    overall_obstacle = _overall_min(result.min_obstacle_clearance)
    overall_inter_agent = _overall_min(result.min_inter_agent_distance)
    print(
        f"overall minimum obstacle clearance ({note}):",
        f"{overall_obstacle:.6f}" if overall_obstacle is not None else "N/A",
    )
    print(
        f"overall minimum inter-agent distance ({note}):",
        f"{overall_inter_agent:.6f}" if overall_inter_agent is not None else "N/A",
    )
    tol_conn = 1e-8
    min_lambda2_comm = min(result.lambda2_comm) if result.lambda2_comm else 0.0
    min_lambda2_cons = min(result.lambda2_cons) if result.lambda2_cons else 0.0
    comm_connected_all_time = all(v > tol_conn for v in result.lambda2_comm)
    cons_connected_all_time = all(v > tol_conn for v in result.lambda2_cons)
    print(f"minimum lambda2_comm ({note}):", f"{min_lambda2_comm:.6e}")
    print(f"minimum lambda2_cons ({note}):", f"{min_lambda2_cons:.6e}")
    print(f"Graph A (||x_i-x_j|| < d_c) connected at all sampled times ({note}):", comm_connected_all_time)
    print(
        f"Graph B (||x_i-x_j|| <= d_c-epsilon_c) connected at all sampled times ({note}):",
        cons_connected_all_time,
    )

    _plot_connectivity_series(
        values=result.lambda2_comm,
        dt=cfg.simulation.dt,
        title="Algebraic Connectivity (Communication Graph)",
        ylabel="lambda2_comm",
        save_path=out_dir / "lambda2_comm.png",
        t_configured=report["horizon"]["t_configured"],
        status_text=status,
        terminated_early=early,
    )
    _plot_connectivity_series(
        values=result.lambda2_cons,
        dt=cfg.simulation.dt,
        title="Algebraic Connectivity (Conservative Graph)",
        ylabel="lambda2_cons",
        save_path=out_dir / "lambda2_cons.png",
        t_configured=report["horizon"]["t_configured"],
        status_text=status,
        terminated_early=early,
    )
    for agent_id in sorted(result.first_target_entry_step):
        first_step = result.first_target_entry_step[agent_id]
        entered = first_step is not None
        print(
            f"agent {agent_id} entered target region ({note}): {entered}; "
            f"first entry step: {first_step}"
        )
    task = report["task"]
    print(
        f"all agents inside target at final simulated state ({note}): "
        f"{task['all_agents_inside_target_at_final_simulated_state']}"
    )
    print(f"task completed AND full horizon completed: {task['task_completed_and_horizon_completed']}")
    print(f"[RUN STATUS] {report['verdict']}")
    print(f"outputs: {out_dir}")
    run_meta.update(
        status="finished",
        finished_at=datetime.now().isoformat(timespec="microseconds"),
        verdict=report["verdict"],
        horizon_completed=report["horizon"]["horizon_completed"],
        termination_reason=result.termination_reason,
    )
    _write_run_meta(out_dir, run_meta)
    if early:
        sys.exit(1)


if __name__ == "__main__":
    main()

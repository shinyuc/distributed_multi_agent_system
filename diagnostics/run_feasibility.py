"""Run a config with per-update local-QP feasibility diagnostics.

Usage:
    python -m diagnostics.run_feasibility --config configs/default.yaml --out outputs/feasibility/default

The simulation is built exactly as in main.py (same parameters, same controller,
same simulator). The only differences are observational: the recorder is attached
to ``controller.qp_observer``, the unbounded in-memory ``qp_debug_records`` is
disabled, and the existing per-step debug prints go to /dev/null (or to a gzip
file with ``--keep-stdout``).

``--max-steps`` truncates the horizon and is meant for smoke tests only.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cvxpy
import osqp
import scipy

from config import load_config
from controller.controller_factory import build_controller
from controller.distributed_controller import DistributedController
from diagnostics.feasibility import Tolerances
from diagnostics.recorder import FeasibilityRecorder
from main import _set_seed
from models.agent import AgentState
from models.environment import CircularObstacle, Environment, TargetRegion
from simulation.simulator import Simulator


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build(cfg):
    """Same construction as main.main()."""
    _set_seed(cfg.simulation.seed)
    agents = [
        AgentState(agent_id=i, position=tuple(pos), velocity=(0.0, 0.0))
        for i, pos in enumerate(cfg.agents.initial_positions)
    ]
    obstacles = [
        CircularObstacle(center=tuple(obs["center"]), radius=float(obs["radius"]))
        for obs in cfg.obstacles.obstacles
    ]
    target = TargetRegion(center=tuple(cfg.target.center), radius=cfg.target.radius, Sigma=cfg.target.Sigma)
    env = Environment(agents=agents, obstacles=obstacles, target=target)
    controller = build_controller(cfg.controller)
    simulator = Simulator(
        env=env,
        controller=controller,
        dt=cfg.simulation.dt,
        steps=cfg.simulation.steps,
        fast_tau=cfg.fast_dynamics.tau,
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
    return env, controller, simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    parser.add_argument("--keep-stdout", action="store_true", help="gzip the simulator's debug prints")
    parser.add_argument("--feas-tol", type=float, default=Tolerances.feas_tol)
    parser.add_argument("--viol-tol", type=float, default=Tolerances.viol_tol)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.simulation.steps = min(cfg.simulation.steps, args.max_steps)
    env, controller, simulator = build(cfg)
    if not isinstance(controller, DistributedController):
        raise SystemExit("Feasibility diagnostics apply to controller.mode == 'distributed' only.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tol = Tolerances(feas_tol=args.feas_tol, viol_tol=args.viol_tol)
    recorder = FeasibilityRecorder(
        out_dir,
        agent_ids=[a.agent_id for a in env.agents],
        n_steps=cfg.simulation.steps,
        dt=cfg.simulation.dt,
        u_max=cfg.controller.u_max,
        d_c=cfg.fast_dynamics.d_c,
        epsilon_c=cfg.fast_dynamics.epsilon_c,
        xi=cfg.fast_dynamics.xi,
        tol=tol,
        progress_every=args.progress_every,
    )
    controller.store_qp_debug_records = False
    controller.qp_observer = recorder

    meta = {
        "config_path": args.config,
        "steps_configured": cfg.simulation.steps,
        "max_steps_override": args.max_steps,
        "dt": cfg.simulation.dt,
        "n_agents": len(env.agents),
        "initial_positions": cfg.agents.initial_positions,
        "u_max": cfg.controller.u_max,
        "eta": cfg.fast_dynamics.eta,
        "d_c": cfg.fast_dynamics.d_c,
        "epsilon_c": cfg.fast_dynamics.epsilon_c,
        "tau": cfg.fast_dynamics.tau,
        "xi": cfg.fast_dynamics.xi,
        "git_commit": _git_commit(),
        "versions": {"python": platform.python_version(), "cvxpy": cvxpy.__version__,
                     "osqp": osqp.__version__, "scipy": scipy.__version__},
        "qp_solver_call": "cvxpy OSQP warm_start=True polish=False (cvxpy defaults eps_abs=eps_rel=1e-5, max_iter=10000)",
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.keep_stdout:
        sink = gzip.open(out_dir / "simulator_stdout.log.gz", "wt")
    else:
        sink = open(os.devnull, "w")
    t0 = time.time()
    result = None
    run_error = None
    try:
        with contextlib.redirect_stdout(sink):
            result = simulator.run()
    except Exception as exc:  # e.g. the simulator's numerical-explosion guard
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        sink.close()
        extra = {
            "run_meta": meta,
            "run_error": run_error,
            "stopped_early": None if result is None else result.stopped_early,
            "simulator_infeasible_step": None if result is None else result.infeasible_step,
            "simulator_infeasible_agent_ids": None if result is None else result.infeasible_agent_ids,
            "steps_completed": None if result is None else len(result.trajectories) - 1,
            "first_target_entry_step": None if result is None else result.first_target_entry_step,
            "run_wall_time_s": time.time() - t0,
        }
        summary = recorder.close(extra)
    if result is not None:
        with (out_dir / "trajectories.json").open("w", encoding="utf-8") as f:
            json.dump({"dt": cfg.simulation.dt, "trajectories": result.trajectories,
                       "lambda2_comm": result.lambda2_comm, "lambda2_cons": result.lambda2_cons}, f)
    print(json.dumps({k: summary[k] for k in ("steps_recorded", "qp_solves", "class_counts", "first_non_solved",
                                               "run_error", "stopped_early")}, indent=2, default=str),
          file=sys.stderr)


if __name__ == "__main__":
    main()

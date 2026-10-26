"""Run a config with per-update local-QP feasibility diagnostics.

Usage:
    python -m diagnostics.run_feasibility --config configs/default_5_agents.yaml
        -> outputs/feasibility/<YYYYmmdd_HHMMSS_ffffff>_default_5_agents_p<pid>_<rand>/

    --output-root DIR   parent of the unique run directory (default outputs/feasibility)
    --out DIR           explicit run directory; must be new or empty unless --overwrite
    --overwrite         with --out: move existing contents to DIR.replaced_<timestamp> (never deleted)

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
import shutil
import subprocess
import sys
import time
import traceback
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
from simulation.reporting import build_run_report
from simulation.simulator import Simulator
from utils.run_dirs import OutputDirExistsError, make_unique_run_dir, prepare_explicit_dir


class _Tee(io.TextIOBase):
    """Write to the original stream and to a log file in the run directory."""

    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close_log(self) -> None:
        self._file.close()


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
    return env, controller, simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None,
                        help="explicit run directory (must be new or empty unless --overwrite)")
    parser.add_argument("--output-root", default="outputs/feasibility",
                        help="parent of the unique timestamped run directory used when --out is omitted")
    parser.add_argument("--overwrite", action="store_true",
                        help="with --out: move an existing non-empty directory aside (renamed, never deleted)")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    parser.add_argument("--keep-stdout", action="store_true", help="gzip the simulator's debug prints")
    parser.add_argument("--feas-tol", type=float, default=Tolerances.feas_tol)
    parser.add_argument("--viol-tol", type=float, default=Tolerances.viol_tol)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    # Claim a fresh run directory before doing anything else.
    moved_to = None
    try:
        if args.out is not None:
            out_dir, moved_to = prepare_explicit_dir(args.out, overwrite=args.overwrite)
        else:
            out_dir = make_unique_run_dir(args.output_root, Path(args.config).stem)
    except OutputDirExistsError as exc:
        print(f"[OUTPUT ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    tee = _Tee(sys.stderr, out_dir / "progress.log")
    sys.stderr = tee
    print(f"[feasibility] run directory: {out_dir}", file=sys.stderr, flush=True)
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"status": "running", "config_path": args.config, "output_dir": str(out_dir),
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv}, f, indent=2)

    cfg = load_config(args.config)
    if cfg.agents.initialization_log_path:
        shutil.copy2(cfg.agents.initialization_log_path, out_dir / "initial_conditions.json")
    if args.max_steps is not None:
        cfg.simulation.steps = min(cfg.simulation.steps, args.max_steps)
    env, controller, simulator = build(cfg)
    if not isinstance(controller, DistributedController):
        raise SystemExit("Feasibility diagnostics apply to controller.mode == 'distributed' only.")

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
        "status": "running",
        "config_path": args.config,
        "output_dir": str(out_dir),
        "previous_contents_moved_to": None if moved_to is None else str(moved_to),
        "argv": sys.argv,
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
        "fast_substeps": cfg.fast_dynamics.substeps,
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
            # Simulator.run() returns partial results itself on numerical-guard and
            # unexpected exceptions; anything escaping here is outside the step loop.
            result = simulator.run()
    except BaseException as exc:
        run_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if not isinstance(exc, Exception):
            raise  # KeyboardInterrupt etc.: still re-raise after the finally block
    finally:
        sink.close()
        report = None
        if result is not None:
            report = build_run_report(
                result,
                dt=cfg.simulation.dt,
                safety_distance=cfg.controller.safety_distance,
                d_c=cfg.fast_dynamics.d_c,
                epsilon_c=cfg.fast_dynamics.epsilon_c,
                target_center=cfg.target.center,
                target_sigma=cfg.target.Sigma,
            )
        extra = {
            "run_meta": meta,
            "run_error": run_error,
            "horizon_completed": False if result is None else result.horizon_completed,
            "verdict": (report or {}).get("verdict", "NO RESULT: exception outside the simulation loop"),
            "termination_reason": "exception" if result is None else result.termination_reason,
            "termination_step": None if result is None else result.termination_step,
            "local_qp_failure_kind": None if result is None else result.local_qp_failure_kind,
            "qp_failures": None if result is None else result.qp_failures,
            "termination_exception": run_error if result is None else result.termination_exception,
            "steps_configured": cfg.simulation.steps,
            "steps_completed": None if result is None else result.steps_completed,
            "stopped_early": None if result is None else result.stopped_early,
            "simulator_infeasible_step": None if result is None else result.infeasible_step,
            "simulator_infeasible_agent_ids": None if result is None else result.infeasible_agent_ids,
            "first_target_entry_step": None if result is None else result.first_target_entry_step,
            "run_wall_time_s": time.time() - t0,
        }
        summary = recorder.close(extra, steps_completed=None if result is None else result.steps_completed)
        if report is not None:
            with (out_dir / "run_report.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        if result is not None:
            with (out_dir / "trajectories.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "dt": cfg.simulation.dt,
                    "steps_configured": result.steps_configured,
                    "steps_completed": result.steps_completed,
                    "horizon_completed": result.horizon_completed,
                    "termination_reason": result.termination_reason,
                    "trajectories": result.trajectories,
                    "control_history_applied": {str(k): v for k, v in result.control_history.items()},
                    "unapplied_controls_at_termination": {str(k): list(v) for k, v in result.unapplied_controls.items()},
                    "lambda2_comm": result.lambda2_comm,
                    "lambda2_cons": result.lambda2_cons,
                }, f)
    print(json.dumps({k: summary.get(k) for k in ("verdict", "steps_recorded", "qp_solves", "class_counts",
                                                   "first_terminating_failure", "termination_reason",
                                                   "horizon_completed", "run_error")}, indent=2, default=str),
          file=sys.stderr)
    meta.update(status="finished", verdict=summary.get("verdict"),
                horizon_completed=summary.get("horizon_completed"),
                termination_reason=summary.get("termination_reason"))
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[feasibility] run directory: {out_dir}", file=sys.stderr, flush=True)
    sys.stderr = tee._stream
    tee.close_log()
    if not summary.get("horizon_completed"):
        sys.exit(1)

if __name__ == "__main__":
    main()

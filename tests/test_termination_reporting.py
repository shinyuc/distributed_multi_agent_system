"""Termination, applied-control bookkeeping, and diagnostic-reporting tests (short runs)."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import controller.distributed_controller as dc
from config import load_config
from controller.local_qp import LocalQPSolveResult
from diagnostics.recorder import FeasibilityRecorder
from diagnostics.run_feasibility import build
from models.agent import AgentState
from simulation.reporting import build_run_report
from simulation.simulator import NumericalGuardError, Simulator

REPO = Path(__file__).resolve().parents[1]
STEP0 = "configs/repro/step0_failure_2026-03-25.yaml"
FEASIBLE = "configs/default_5_agents.yaml"


def _setup(cfg_path: str, steps: int, tmp_path: Path | None = None):
    cfg = load_config(str(REPO / cfg_path))
    cfg.simulation.steps = steps
    env, ctrl, sim = build(cfg)
    rec = None
    if tmp_path is not None:
        rec = FeasibilityRecorder(
            tmp_path, agent_ids=[a.agent_id for a in env.agents], n_steps=steps, dt=cfg.simulation.dt,
            u_max=cfg.controller.u_max, d_c=cfg.fast_dynamics.d_c, epsilon_c=cfg.fast_dynamics.epsilon_c,
            xi=cfg.fast_dynamics.xi, progress_every=0,
        )
        ctrl.qp_observer = rec
        ctrl.store_qp_debug_records = False
    return cfg, env, ctrl, sim, rec


@pytest.fixture
def euler_calls(monkeypatch):
    calls: list[tuple[int, tuple[float, float]]] = []
    original = AgentState.euler_step

    def counting(self, u_k, dt):
        calls.append((self.agent_id, u_k))
        return original(self, u_k, dt)

    monkeypatch.setattr(AgentState, "euler_step", counting)
    return calls


def _patch_solver(monkeypatch, n_agents: int, hook):
    """hook(step, agent, real_result) -> result. QPs are solved in agent order each step."""
    original = dc.solve_local_qp_detailed
    counter = {"k": 0}

    def patched(qp_data):
        k = counter["k"]
        counter["k"] += 1
        return hook(k // n_agents, k % n_agents, original(qp_data))

    monkeypatch.setattr(dc, "solve_local_qp_detailed", patched)


# 1-4: reproduced step-0 infeasibility ------------------------------------------------
def test_step0_infeasible_terminates_before_any_update(euler_calls, capsys):
    cfg, env, ctrl, sim, _ = _setup(STEP0, 50)
    x0 = [a.position for a in env.agents]
    fs0 = copy.deepcopy(sim.fast_states)
    res = sim.run()

    assert res.termination_reason == "local_qp_failure"
    assert res.local_qp_failure_kind == "solver_reported_infeasible"
    assert not res.horizon_completed and res.stopped_early
    assert res.termination_step == 0 and res.steps_completed == 0 and res.steps_configured == 50
    assert [(f["agent_id"], f["solver_status"]) for f in res.qp_failures] == [(2, "infeasible"), (4, "infeasible")]
    assert set(res.qp_failure_diagnostics) == {2, 4}
    # no control of any kind (no fallback) was applied, state and fast state untouched
    assert euler_calls == []
    assert [a.position for a in env.agents] == x0 and sim.fast_states == fs0
    assert len(res.trajectories) == 1
    # attempted-but-unapplied inputs are kept separately, never in the applied history
    assert all(len(h) == 0 for h in res.control_history.values())
    assert set(res.unapplied_controls) == {0, 1, 3}


def test_numerical_solver_failure_mid_run(monkeypatch, euler_calls, capsys):
    cfg, env, ctrl, sim, _ = _setup(FEASIBLE, 10)
    solved: dict[tuple[int, int], tuple[float, float]] = {}

    def hook(step, agent, real):
        if (step, agent) == (2, 3):
            return LocalQPSolveResult((0.0, 0.0), False, "solver status=user_limit", "user_limit")
        solved[(step, agent)] = real.u
        return real

    _patch_solver(monkeypatch, len(env.agents), hook)
    res = sim.run()
    assert res.termination_reason == "local_qp_failure"
    assert res.local_qp_failure_kind == "solver_numerical_failure"
    assert res.qp_failures[0]["solver_status"] == "user_limit"
    assert res.steps_completed == 2 and res.termination_step == 2
    # applied history == the QP solutions of steps 0 and 1 exactly, nothing from step 2
    for i, hist in res.control_history.items():
        assert hist == [solved[(0, i)], solved[(1, i)]]
    assert euler_calls == [(i, solved[(k, i)]) for k in (0, 1) for i in range(5)]
    assert res.unapplied_controls == {i: solved[(2, i)] for i in (0, 1, 2, 4)}


def test_completed_run_is_reported_completed(capsys):
    cfg, env, ctrl, sim, _ = _setup(FEASIBLE, 5)
    res = sim.run()
    assert res.termination_reason == "completed" and res.horizon_completed
    assert res.steps_completed == res.steps_configured == 5
    assert all(len(h) == 5 for h in res.control_history.values())
    assert res.unapplied_controls == {} and res.qp_failures == []
    rep = build_run_report(res, dt=cfg.simulation.dt, safety_distance=cfg.controller.safety_distance,
                           d_c=cfg.fast_dynamics.d_c, epsilon_c=cfg.fast_dynamics.epsilon_c,
                           target_center=cfg.target.center, target_sigma=cfg.target.Sigma)
    assert rep["verdict"].startswith("COMPLETED")
    assert rep["simulated_interval"]["covers_configured_horizon"] is True
    # the task is not done after 5 steps: task completion is reported separately from the horizon
    assert rep["task"]["task_completed_and_horizon_completed"] is False


def test_early_termination_report_never_claims_success(capsys):
    cfg, env, ctrl, sim, _ = _setup(STEP0, 50)
    res = sim.run()
    rep = build_run_report(res, dt=cfg.simulation.dt, safety_distance=cfg.controller.safety_distance,
                           d_c=cfg.fast_dynamics.d_c, epsilon_c=cfg.fast_dynamics.epsilon_c,
                           target_center=cfg.target.center, target_sigma=cfg.target.Sigma)
    assert rep["verdict"].startswith("TERMINATED EARLY")
    assert rep["horizon"]["horizon_completed"] is False
    for key in ("safety", "connectivity", "task"):
        assert rep[key]["interval"]["covers_configured_horizon"] is False
        assert rep[key]["interval"]["t_end"] == 0.0
    assert rep["task"]["task_completed_and_horizon_completed"] is False


def test_numerical_guard_returns_partial_results(monkeypatch, capsys):
    cfg, env, ctrl, sim, _ = _setup(FEASIBLE, 10)
    original = Simulator._check_fast_state_explosion

    def guard(self, step_idx, agent_id, state, threshold=1e6):
        if step_idx == 3:
            raise NumericalGuardError("forced guard for test")
        return original(self, step_idx, agent_id, state, threshold)

    monkeypatch.setattr(Simulator, "_check_fast_state_explosion", guard)
    res = sim.run()
    assert res.termination_reason == "numerical_guard"
    assert res.termination_exception["type"] == "NumericalGuardError"
    assert "forced guard for test" in res.termination_exception["traceback"]
    assert res.steps_completed == 3 and len(res.trajectories) == 4
    assert all(len(h) == 3 for h in res.control_history.values())
    assert set(res.unapplied_controls) == {0, 1, 2, 3, 4}


def test_unexpected_exception_returns_partial_results(capsys):
    cfg, env, ctrl, sim, _ = _setup(FEASIBLE, 10)
    seen = {"n": 0}

    def bad_observer(**kw):
        seen["n"] += 1
        if kw["step_idx"] == 2:
            raise ValueError("observer bug")

    ctrl.qp_observer = bad_observer
    res = sim.run()
    assert res.termination_reason == "exception"
    assert res.termination_exception["type"] == "ValueError"
    assert res.steps_completed == 2 and not res.horizon_completed


# 6: OSQP / LP disagreement and LP failure ---------------------------------------------
def test_osqp_lp_disagreement_is_reported(monkeypatch, tmp_path, capsys):
    cfg, env, ctrl, sim, rec = _setup(STEP0, 2, tmp_path)

    def hook(step, agent, real):
        if not real.success:  # pretend OSQP claimed success on the infeasible QPs
            return LocalQPSolveResult((0.0, 0.0), True, "optimal", "optimal")
        return real

    _patch_solver(monkeypatch, len(env.agents), hook)
    res = sim.run()
    summ = rec.close({}, steps_completed=res.steps_completed)
    # diagnostics never intervene: the run continues on the OSQP answer
    assert res.termination_reason == "completed"
    assert summ["class_counts"]["inconsistent"] == 4
    ver = summ["verification"]
    assert ver["osqp_success_but_lp_infeasible"]["count"] == 4
    assert ver["verified_feasible_and_solved"] == 6
    assert summ["first_not_verified"]["class"] == "inconsistent"
    assert summ["first_terminating_failure"] is None
    rows = list(csv.DictReader(gzip.open(tmp_path / "qp_log.csv.gz", "rt")))
    bad = [r for r in rows if r["class"] == "inconsistent"]
    assert {(r["agent"], r["lp_verdict"], r["solver_status"], r["applied"]) for r in bad} == {
        ("2", "infeasible", "optimal", "1"), ("4", "infeasible", "optimal", "1")}
    assert len(list((tmp_path / "failures").glob("*_inconsistent.json"))) == 4


def test_lp_failure_is_not_reported_as_verified(monkeypatch, tmp_path, capsys):
    import diagnostics.recorder as recorder_mod

    cfg, env, ctrl, sim, rec = _setup(FEASIBLE, 2, tmp_path)
    real_lf = recorder_mod.local_feasibility

    def failing_lp(*args, **kwargs):
        out = real_lf(*args, **kwargs)
        out.update(s_star=math.nan, lp_status="highs_status_4: forced", binding=[])
        return out

    monkeypatch.setattr(recorder_mod, "local_feasibility", failing_lp)
    res = sim.run()
    summ = rec.close({}, steps_completed=res.steps_completed)
    assert summ["class_counts"]["lp_error"] == 10 and summ["class_counts"]["solved"] == 0
    assert summ["verification"]["verified_feasible_and_solved"] == 0
    assert summ["verification"]["lp_errors"]["count"] == 10


# 4-5: output files written by the diagnostic runner -----------------------------------
def _runner(cfg_path: str, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "diagnostics.run_feasibility", "--config", cfg_path, "--out", str(out),
         "--progress-every", "0", *extra],
        cwd=REPO, capture_output=True, text=True,
    )


def test_runner_outputs_for_step0_failure(tmp_path):
    proc = _runner(STEP0, tmp_path)
    assert proc.returncode == 1
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["horizon_completed"] is False
    assert summ["verdict"].startswith("TERMINATED EARLY")
    assert summ["termination_reason"] == "local_qp_failure"
    assert summ["local_qp_failure_kind"] == "solver_reported_infeasible"
    assert [(f["agent_id"], f["solver_status"]) for f in summ["qp_failures"]] == [(2, "infeasible"), (4, "infeasible")]
    assert summ["steps_completed"] == 0 and summ["steps_configured"] == 4000
    assert summ["first_terminating_failure"]["agent"] == 2
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert report["horizon"]["horizon_completed"] is False
    traj = json.loads((tmp_path / "trajectories.json").read_text())
    assert traj["termination_reason"] == "local_qp_failure" and len(traj["trajectories"]) == 1
    assert all(v == [] for v in traj["control_history_applied"].values())
    assert set(traj["unapplied_controls_at_termination"]) == {"0", "1", "3"}
    rows = list(csv.DictReader(gzip.open(tmp_path / "qp_log.csv.gz", "rt")))
    assert {r["applied"] for r in rows} == {"0"}


def test_runner_outputs_for_completed_run(tmp_path):
    proc = _runner(FEASIBLE, tmp_path, "--max-steps", "5")
    assert proc.returncode == 0, proc.stderr[-2000:]
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["horizon_completed"] is True and summ["termination_reason"] == "completed"
    assert summ["verdict"].startswith("COMPLETED")
    assert summ["steps_completed"] == summ["steps_configured"] == 5
    rows = list(csv.DictReader(gzip.open(tmp_path / "qp_log.csv.gz", "rt")))
    assert len(rows) == 25 and {r["applied"] for r in rows} == {"1"}

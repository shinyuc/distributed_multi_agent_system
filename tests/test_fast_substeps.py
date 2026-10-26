"""Multi-rate integration: M fast Euler substeps of dt / M per slow step."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import controller.distributed_controller as dc
import simulation.simulator as simulator_module
from config import load_config
from controller.fast_dynamics import update_fast_state
from diagnostics.run_feasibility import build
from models.agent import AgentState
from simulation.simulator import NumericalGuardError, Simulator

REPO = Path(__file__).resolve().parents[1]
FEASIBLE = "configs/default_5_agents.yaml"


def _setup(substeps: int, steps: int):
    cfg = load_config(str(REPO / FEASIBLE))
    cfg.simulation.steps = steps
    cfg.fast_dynamics.substeps = substeps
    env, ctrl, sim = build(cfg)
    return cfg, env, ctrl, sim


@pytest.fixture
def events(monkeypatch):
    """Ordered log of QP solves, fast updates and slow Euler updates."""
    log: list[tuple] = []

    original_solve = dc.solve_local_qp_detailed

    def solve(qp_data):
        log.append(("qp",))
        return original_solve(qp_data)

    def fast(state, **kwargs):
        log.append((
            "fast",
            kwargs["step_idx"],
            kwargs["i"],
            dict(kwargs),
            state,
            {j: a.position for j, a in kwargs["all_agent_states"].items()},
        ))
        return update_fast_state(state=state, **kwargs)

    original_euler = AgentState.euler_step

    def euler(self, u_k, dt):
        log.append(("euler", self.agent_id, u_k, dt))
        return original_euler(self, u_k, dt)

    monkeypatch.setattr(dc, "solve_local_qp_detailed", solve)
    monkeypatch.setattr(simulator_module, "update_fast_state", fast)
    monkeypatch.setattr(AgentState, "euler_step", euler)
    return log


@pytest.mark.parametrize("M", [1, 3])
def test_update_ordering_and_qp_count(events, M, capsys):
    steps = 3
    cfg, env, _, sim = _setup(M, steps)
    n = len(env.agents)
    res = sim.run()
    assert res.termination_reason == "completed"

    # Per outer step: N QP solves, then M*N fast updates, then N slow updates.
    kinds = [e[0] for e in events]
    assert kinds == (["qp"] * n + ["fast"] * (M * n) + ["euler"] * n) * steps
    assert kinds.count("qp") == steps * n

    fast_events = [e for e in events if e[0] == "fast"]
    for k in range(steps):
        step_calls = fast_events[k * M * n:(k + 1) * M * n]
        assert {e[1] for e in step_calls} == {k}
        x_k = {i: tuple(p) for i, p in enumerate(res.trajectories[k])}
        first_kwargs = step_calls[0][3]
        for e in step_calls:
            _, _, i, kw, _, positions = e
            # Slow state, control and x-dependent quantities held fixed.
            assert positions == x_k
            assert kw["x_i"] == x_k[i]
            assert kw["u_i"] == res.control_history[i][k]
            assert kw["lambda_2"] == first_kwargs["lambda_2"]
            assert kw["v2"] == first_kwargs["v2"]
            assert kw["dt"] == cfg.simulation.dt / M
        # Synchronous Jacobi: within a substep every agent reads the same snapshot.
        for s in range(M):
            sub = step_calls[s * n:(s + 1) * n]
            assert [e[2] for e in sub] == [a.agent_id for a in env.agents]
            snapshot = sub[0][3]["all_fast_states"]
            assert all(e[3]["all_fast_states"] is snapshot for e in sub)
            assert all(e[4] is snapshot[e[2]] for e in sub)

    # Slow update applies u_k once with the full slow step.
    for e in events:
        if e[0] == "euler":
            assert e[3] == cfg.simulation.dt


def test_m1_passes_full_dt(events, capsys):
    cfg, _, _, sim = _setup(1, 1)
    sim.run()
    assert all(e[3]["dt"] == cfg.simulation.dt for e in events if e[0] == "fast")


def test_substeps_chain_and_persist_across_outer_steps(events, capsys):
    """Replaying the recorded calls from the zero initial state reproduces the final fast state."""
    M, steps = 4, 3
    _, env, _, sim = _setup(M, steps)
    theta = copy.deepcopy(sim.fast_states)
    sim.run()

    n = len(env.agents)
    fast_events = [e for e in events if e[0] == "fast"]
    assert len(fast_events) == steps * M * n
    for start in range(0, len(fast_events), n):
        sub = fast_events[start:start + n]
        # Each substep starts from exactly the previous substep's result
        # (including across outer steps: no reinitialization).
        assert sub[0][3]["all_fast_states"] == theta
        # Recorded AgentStates are live objects; rebuild them at the recorded x_k.
        x_k = {j: AgentState(agent_id=j, position=p, velocity=(0.0, 0.0)) for j, p in sub[0][5].items()}
        theta = {
            e[2]: update_fast_state(
                **{**e[3], "state": theta[e[2]], "all_fast_states": theta, "all_agent_states": x_k}
            )
            for e in sub
        }
    assert sim.fast_states == theta


def test_substeps_change_fast_state_but_not_qp_inputs_at_step0(capsys):
    """At step 0, the QP sees the same (x_0, z_0) for any M; only the fast update differs."""
    res = {}
    states = {}
    for M in (1, 10):
        _, _, _, sim = _setup(M, 1)
        res[M] = sim.run()
        states[M] = sim.fast_states
    assert res[1].control_history == res[10].control_history
    assert res[1].trajectories == res[10].trajectories
    assert states[1] != states[10]


def test_numerical_guard_checked_every_substep(monkeypatch, capsys):
    M = 3
    _, env, _, sim = _setup(M, 5)
    n = len(env.agents)
    calls = {"k": 0}
    original_guard = sim._check_fast_state_explosion

    def guard(step_idx, agent_id, state):
        calls["k"] += 1
        if step_idx == 1 and calls["k"] == M * n + n + 1:  # step 1, substep 2, first agent
            raise NumericalGuardError("forced")
        return original_guard(step_idx=step_idx, agent_id=agent_id, state=state)

    monkeypatch.setattr(sim, "_check_fast_state_explosion", guard)
    state_after_step0: list = []
    original_record = sim._record_fast_history_snapshot

    def record(history):
        state_after_step0.append(copy.deepcopy(sim.fast_states))
        return original_record(history)

    monkeypatch.setattr(sim, "_record_fast_history_snapshot", record)
    res = sim.run()

    assert res.termination_reason == "numerical_guard" and res.termination_step == 1
    assert res.steps_completed == 1 and len(res.trajectories) == 2
    # Guard ran on every agent after every substep of step 0.
    assert calls["k"] == M * n + n + 1
    # Fast state not partially advanced at the failing step: still the end-of-step-0 state.
    assert sim.fast_states == state_after_step0[-1]
    # History recorded once per completed outer step (initial + step 0).
    assert all(len(v) == 2 for v in res.fast_z_c.values())


def test_history_recorded_once_per_outer_step(capsys):
    steps = 4
    _, _, _, sim = _setup(5, steps)
    res = sim.run()
    assert len(res.trajectories) == steps + 1
    for series in (res.fast_w_norm, res.fast_z_c, res.fast_y_c, res.fast_max_abs_z_pair_self):
        assert all(len(v) == steps + 1 for v in series.values())


# Configuration ---------------------------------------------------------------------
def _write_cfg(tmp_path: Path, substeps) -> str:
    raw = yaml.safe_load((REPO / FEASIBLE).read_text())
    if substeps is not None:
        raw["fast_dynamics"]["substeps"] = substeps
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, True, False, "2", "not-a-number"])
def test_invalid_substeps_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="substeps"):
        load_config(_write_cfg(tmp_path, bad))


def test_substeps_default_and_explicit(tmp_path):
    assert load_config(_write_cfg(tmp_path, None)).fast_dynamics.substeps == 1
    assert load_config(_write_cfg(tmp_path, 10)).fast_dynamics.substeps == 10


@pytest.mark.parametrize("bad", [0, -2, 2.0, True])
def test_simulator_rejects_invalid_substeps(bad):
    cfg = load_config(str(REPO / FEASIBLE))
    env, ctrl, _ = build(cfg)
    with pytest.raises(ValueError, match="fast_substeps"):
        Simulator(env=env, controller=ctrl, dt=cfg.simulation.dt, steps=1, fast_substeps=bad)


def test_build_uses_configured_substeps():
    cfg = load_config(str(REPO / FEASIBLE))
    cfg.fast_dynamics.substeps = 7
    _, _, sim = build(cfg)
    assert sim.fast_substeps == 7

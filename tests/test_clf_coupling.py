"""Configurable CLF mismatch coupling: gated v_ij = h_ij sigma_i sigma_j vs communication v_ij = h_ij.

alpha_clf = 1 throughout (the alpha_clf scaling of the fast CLF value is a separate, known issue).
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import controller.distributed_controller as dc
import simulation.simulator as sm
from config import load_config
from controller.centralized_qp import build_centralized_qp_data
from controller.fast_dynamics import (
    alpha_ij_connectivity,
    clf_coupling_gates,
    clf_constraint_value,
    compute_beta_i,
    compute_V_i,
    connectivity_sigma,
    h_ij,
    update_z_clf,
)
from controller.local_qp import build_clf_constraint_row, build_local_qp_data, solve_local_qp_detailed
from diagnostics.feasibility import Tolerances, centralized_feasibility, local_feasibility, lp_verdict
from diagnostics.recorder import FeasibilityRecorder
from diagnostics.run_batch import run_batch
from diagnostics.run_feasibility import build
from models.agent import AgentState
from models.environment import CircularObstacle
from models.fast_state import init_fast_state_complete_graph

REPO = Path(__file__).resolve().parents[1]
MODES = ("gated", "communication")
SIG = [[2.0, 0.0], [0.0, 0.25]]
EPS, DC, EPSC, GAMMA3 = 0.5, 0.9, 0.1, 0.01
CENTER = (0.0, 0.0)
TOL = Tolerances()


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _sigma(q: float) -> float:
    return math.exp(-1.0 / q) if q > 0.0 else 0.0


def _W(x) -> float:
    q = compute_V_i(x, CENTER, SIG, EPS)
    return q * math.exp(-1.0 / q) if q > 0.0 else 0.0


def _random_setting(seed: int):
    """Agents inside (q <= 0) and outside their inner target, some pairs beyond d_c."""
    rng = np.random.default_rng(seed)
    pos = {0: (0.0, 0.1), 1: (0.3, -0.2), 2: (-0.65, 0.1), 3: (0.7, 0.4), 4: (-0.6, 0.85), 5: (1.4, -0.3)}
    pos = {i: (p[0] + rng.normal(scale=0.03), p[1] + rng.normal(scale=0.03)) for i, p in pos.items()}
    n = len(pos)
    agents = {i: AgentState(i, p, (0.0, 0.0)) for i, p in pos.items()}
    fast = {i: init_fast_state_complete_graph(i, n, 0) for i in pos}
    for i in pos:
        fast[i].w = tuple(rng.normal(scale=0.05, size=2))
        fast[i].z_clf = float(rng.normal())
        fast[i].y_clf = float(abs(rng.normal()))
    q = {i: compute_V_i(p, CENTER, SIG, EPS) for i, p in pos.items()}
    assert any(v <= 0 for v in q.values()) and any(v > 0 for v in q.values())
    return agents, fast


def _weights(agents, mode):
    ids = sorted(agents)
    V = np.zeros((len(ids), len(ids)))
    for a in ids:
        for b in ids:
            if a != b:
                xa, xb = agents[a].position, agents[b].position
                ga, gb = clf_coupling_gates(_sigma(compute_V_i(xa, CENTER, SIG, EPS)),
                                            _sigma(compute_V_i(xb, CENTER, SIG, EPS)), mode)
                V[a, b] = h_ij(xa, xb, DC) * ga * gb
    return V


def _clf_row(i, agents, fast, mode):
    return _quiet(build_clf_constraint_row, i=i, state=fast[i], all_fast_states=fast, all_agent_states=agents,
                  gamma_3=GAMMA3, d_c=DC, target_center=CENTER, sigma=SIG, epsilon=EPS, alpha_clf=1.0,
                  clf_coupling=mode)


def _g_clf(i, agents, fast, mode, w=None):
    return clf_constraint_value(i=i, state=fast[i], all_fast_states=fast, x_i=agents[i].position,
                                all_agent_states=agents, w_i=fast[i].w if w is None else w, gamma_3=GAMMA3,
                                d_c=DC, target_center=CENTER, sigma=SIG, epsilon=EPS, clf_coupling=mode)


# ------------------------------------------------------------------ configuration
def test_config_default_and_validation(tmp_path):
    assert load_config(str(REPO / "configs/distributed_10_agents.yaml")).controller.clf_coupling == "gated"
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    for value, ok in (("communication", True), ("gated", True), ("sigma", False), ("", False)):
        raw["controller"]["clf_coupling"] = value
        p = tmp_path / f"c_{value or 'empty'}.yaml"
        p.write_text(yaml.safe_dump(raw))
        if ok:
            assert load_config(str(p)).controller.clf_coupling == value
        else:
            with pytest.raises(ValueError, match="clf_coupling"):
                load_config(str(p))


def test_coupling_gates():
    assert clf_coupling_gates(0.3, 0.0, "gated") == (0.3, 0.0)
    assert clf_coupling_gates(0.3, 0.0, "communication") == (1.0, 1.0)
    with pytest.raises(ValueError):
        clf_coupling_gates(0.3, 0.2, "other")


# ------------------------------------------------------------------ row / dynamics consistency
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(5))
def test_local_qp_row_equals_fast_clf_constraint(mode, seed):
    agents, fast = _random_setting(seed)
    for i in agents:
        a, b, label = _clf_row(i, agents, fast, mode)
        w = np.asarray(fast[i].w)
        g = _g_clf(i, agents, fast, mode)
        assert np.dot(a, w) - b == pytest.approx(g, rel=1e-12, abs=1e-15)
        inside = compute_V_i(agents[i].position, CENTER, SIG, EPS) <= 0.0
        assert label.endswith("inactive)") == inside
        if inside:
            assert a == (0.0, 0.0)
            if mode == "gated":
                assert b == 0.0 and g == 0.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(3))
def test_z_clf_dynamics_is_the_lagrangian_z_gradient(mode, seed):
    """tau z_dot = -xi z - d/dz sum_i y_i g_i^clf, with the weights of the chosen mode."""
    agents, fast = _random_setting(seed)
    xi, dt, tau = 0.5, 1e-3, 0.02
    for k in agents:
        z_new = update_z_clf(i=k, state=fast[k], all_fast_states=fast, all_agent_states=agents, dt=dt, tau=tau,
                             xi=xi, alpha_clf=1.0, d_c=DC, target_center=CENTER, sigma=SIG, epsilon=EPS,
                             clf_coupling=mode)
        rhs = (z_new - fast[k].z_clf) * tau / dt
        grad = 0.0
        step = 1e-3
        for sign in (1.0, -1.0):
            pert = copy.deepcopy(fast)
            pert[k].z_clf += sign * step
            grad += sign * sum(fast[i].y_clf * _g_clf(i, agents, pert, mode, w=fast[i].w) for i in agents)
        grad /= 2 * step
        assert rhs == pytest.approx(-xi * fast[k].z_clf - grad, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_coupling_symmetry_cancellation_and_consistency_across_modules(mode):
    agents, fast = _random_setting(7)
    V = _weights(agents, mode)
    assert np.array_equal(V, V.T)
    if mode == "communication":
        H = np.array([[h_ij(agents[a].position, agents[b].position, DC) if a != b else 0.0 for b in agents]
                      for a in agents])
        assert np.array_equal(V, H)
    z = np.array([fast[i].z_clf for i in sorted(agents)])
    coupling = {i: _g_clf(i, agents, fast, mode, w=(0.0, 0.0)) - GAMMA3 * _W(agents[i].position) for i in agents}
    for i in agents:  # Laplacian form sum_j v_ij (z_i - z_j)
        assert coupling[i] == pytest.approx(sum(V[i, j] * (z[i] - z[j]) for j in agents), rel=1e-12, abs=1e-15)
    assert abs(sum(coupling.values())) < 1e-12
    # The recorder reports the same coupling term.
    kw = dict(all_fast_states=fast, all_agent_states=agents, n_agents=len(agents), target_center=CENTER, sigma=SIG,
              epsilon=EPS, d_c=DC, epsilon_c=EPSC, enable_connectivity_constraint=False, gamma_3=GAMMA3,
              alpha_clf=1.0, alpha_connectivity=1.0, clf_coupling=mode)
    for i in agents:
        assert FeasibilityRecorder._decomposition(None, i, kw)["clf_mismatch"] == pytest.approx(coupling[i], abs=1e-15)
    # The augmented centralized CLF rows have the same coefficients (rows are positively rescaled).
    cqp = build_centralized_qp_data(
        all_agent_states=agents, all_fast_states=fast, obstacles=[], u_max=20.0, xi=0.5, gamma_1=2.0, d_0=0.1,
        gamma_2=2.0, eta=0.1, d_c=DC, epsilon_c=EPSC, gamma_3=GAMMA3, target_center=CENTER, sigma=SIG,
        epsilon=EPS, enable_connectivity_constraint=False, alpha_clf=1.0, alpha_connectivity=1.0,
        lambda_2=None, v2=None, clf_coupling=mode)
    zcols = [cqp.decision_labels.index(f"z_clf[{j}]") for j in sorted(agents)]
    for i in agents:
        row = np.asarray(cqp.A[cqp.row_labels.index(next(l for l in cqp.row_labels if l.startswith(f"clf(i={i},")))])
        expected = np.array([-V[i, j] if j != i else V[i].sum() for j in sorted(agents)])
        got = row[zcols]
        if not expected.any():
            assert not got.any()
            continue
        a_local, _, _ = _clf_row(i, agents, fast, mode)
        raw_norm = math.hypot(*a_local, *expected)  # rhs magnitude is below this norm here
        assert got == pytest.approx(expected / max(raw_norm, abs(GAMMA3 * _W(agents[i].position)), 1e-8), rel=1e-9)


# ------------------------------------------------------------------ agents inside their target
def _inside_agent_qp(mode, z_clf_inside: float):
    agents, fast = _random_setting(3)
    inside = next(i for i in agents if compute_V_i(agents[i].position, CENTER, SIG, EPS) <= 0.0)
    for i in fast:
        fast[i].z_clf = 0.0
    fast[inside].z_clf = z_clf_inside
    qp = _quiet(build_local_qp_data, i=inside, state=fast[inside], all_fast_states=fast, all_agent_states=agents,
                obstacles=[], u_max=20.0, gamma_1=2.0, d_0=0.1, gamma_2=2.0, eta=0.1, d_c=DC, epsilon_c=EPSC,
                gamma_3=GAMMA3, target_center=CENTER, sigma=SIG, epsilon=EPS,
                enable_connectivity_constraint=False, alpha_clf=1.0, clf_coupling=mode)
    k = next(n for n, l in enumerate(qp.row_labels) if l.startswith("clf("))
    return qp, k, agents, fast, inside


@pytest.mark.parametrize("z_inside,feasible", [(-0.5, True), (0.0, True), (0.5, False)])
def test_inside_target_agent_keeps_its_coupling_row_in_communication_mode(z_inside, feasible):
    qp, k, agents, fast, inside = _inside_agent_qp("communication", z_inside)
    coupling = sum(h_ij(agents[inside].position, agents[j].position, DC) * (z_inside - 0.0) for j in agents if j != inside)
    assert qp.A_raw[k] == [0.0, 0.0] and qp.b_raw[k] == pytest.approx(-coupling)
    assert _g_clf(inside, agents, fast, "communication") == pytest.approx(coupling)
    res = solve_local_qp_detailed(qp)
    lp = local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels, 20.0, TOL)
    assert res.success is feasible and lp_verdict(lp["s_star"], TOL) == ("feasible" if feasible else "infeasible")
    if not feasible:
        assert res.status == "infeasible" and lp["degenerate_violated"][0]["label"].startswith("clf(")


@pytest.mark.parametrize("z_inside", [-0.5, 0.5])
def test_inside_target_agent_has_no_coupling_in_gated_mode(z_inside):
    qp, k, agents, fast, inside = _inside_agent_qp("gated", z_inside)
    assert qp.A_raw[k] == [0.0, 0.0] and qp.b_raw[k] == 0.0
    assert _g_clf(inside, agents, fast, "gated") == 0.0
    assert solve_local_qp_detailed(qp).success


# ------------------------------------------------------------------ 3-agent counterexample
# Agents 0, 1 outside the inner target, agent 2 inside it; 0 and 1 communicate only through 2.
# Agent 0 sits delta from an obstacle lying between it and the target.
CE = dict(g1=2.0, g2=2.0, d0=0.2, chi=0.1, umax=20.0, r=0.2, delta=5e-5)
CE_X = {0: (-0.6, 0.0), 1: (0.6, 0.0), 2: (0.0, 0.0)}
CE_U = {0: np.zeros(2), 1: np.array([-1e-3, 0.0]), 2: np.zeros(2)}


def _ce_setup():
    agents = {i: AgentState(i, p, (0.0, 0.0)) for i, p in CE_X.items()}
    obst = [CircularObstacle((-0.4 + CE["delta"], 0.0), CE["r"])]
    sig = connectivity_sigma(DC, EPSC)
    A = np.zeros((3, 3))
    for a in range(3):
        for b in range(a + 1, 3):
            A[a, b] = A[b, a] = alpha_ij_connectivity(CE_X[a], CE_X[b], DC, EPSC, sig)
    w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
    v2 = {i: float(V[i, 1]) for i in range(3)}
    return agents, obst, float(w[1]), v2


def _ce_qp(i, agents, fast, obst, lam, v2, mode):
    return _quiet(build_local_qp_data, i=i, state=fast[i], all_fast_states=fast, all_agent_states=agents,
                  obstacles=obst, u_max=CE["umax"], gamma_1=CE["g1"], d_0=CE["d0"], gamma_2=CE["g2"],
                  eta=CE["chi"], d_c=DC, epsilon_c=EPSC, gamma_3=GAMMA3, target_center=CENTER, sigma=SIG,
                  epsilon=EPS, enable_connectivity_constraint=True, alpha_clf=1.0, alpha_connectivity=1.0,
                  lambda_2=lam, v2=v2, n_agents=3, clf_coupling=mode)


def _ce_allocation(agents, lam, v2):
    """Explicit mismatch allocation for u = CE_U (communication weights v_ij = h_ij)."""
    fast = {i: init_fast_state_complete_graph(i, 3, 1) for i in range(3)}
    Hm = np.array([[h_ij(CE_X[a], CE_X[b], DC) if a != b else 0.0 for b in range(3)] for a in range(3)])
    L = np.diag(Hm.sum(1)) - Hm
    for i in range(3):  # pairwise: equal split of each active pair's slack
        for j in range(i + 1, 3):
            if Hm[i, j] > 0:
                d = np.subtract(CE_X[i], CE_X[j])
                H = d @ d - CE["d0"] ** 2
                a_i = -2 * d @ CE_U[i] - 0.5 * CE["g1"] * H
                a_j = 2 * d @ CE_U[j] - 0.5 * CE["g1"] * H
                m = (a_j - a_i) / 2
                fast[i].z_pair[j].self_component, fast[j].z_pair[i].self_component = m / 2, -m / 2
    s = np.array([np.dot(compute_beta_i(i, None, {}, agents, DC, EPSC, v2), CE_U[i]) + (lam - CE["chi"]) / 3
                  for i in range(3)])
    c = []
    for i in range(3):
        q = compute_V_i(CE_X[i], CENTER, SIG, EPS)
        grad = 2 * _sigma(q) * (1 + 1 / q) * (np.array(SIG) @ CE_X[i]) if q > 0 else np.zeros(2)
        c.append(grad @ CE_U[i] + GAMMA3 * _W(CE_X[i]))
    c = np.array(c)
    zc = np.linalg.lstsq(L, s - s.mean(), rcond=None)[0]  # L z = s - mean(s) <= s
    zclf = np.linalg.lstsq(L, -c + c.mean(), rcond=None)[0]  # L z = -c + mean(c) <= -c
    for i in range(3):
        fast[i].z_c, fast[i].z_clf = float(zc[i]), float(zclf[i])
    return fast, s.sum(), c.sum()


def _margin(qp) -> float:
    return local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels, CE["umax"], TOL)["s_star"]


def _centralized_margin(agents, fast, obst, lam, v2, mode) -> float:
    cqp = _quiet(build_centralized_qp_data, all_agent_states=agents, all_fast_states=fast, obstacles=obst,
                 u_max=CE["umax"], xi=0.5, gamma_1=CE["g1"], d_0=CE["d0"], gamma_2=CE["g2"], eta=CE["chi"], d_c=DC,
                 epsilon_c=EPSC, gamma_3=GAMMA3, target_center=CENTER, sigma=SIG, epsilon=EPS,
                 enable_connectivity_constraint=True, alpha_clf=1.0, alpha_connectivity=1.0, lambda_2=lam, v2=v2,
                 clf_coupling=mode)
    return centralized_feasibility(np.asarray(cqp.A), np.asarray(cqp.b), cqp.row_labels, cqp.row_blocks,
                                   cqp.control_dim, len(cqp.q), CE["umax"], TOL)["s_star"]


def test_counterexample_is_centrally_feasible():
    agents, obst, lam, v2 = _ce_setup()
    _, g_c, g_clf = _ce_allocation(agents, lam, v2)
    assert lam - CE["chi"] > 0 and g_c > 0 and g_clf < 0  # h_c >= 0, centralized rows hold at u*
    for i, j in ((0, 2), (1, 2)):
        d = np.subtract(CE_X[i], CE_X[j])
        assert 2 * d @ (CE_U[i] - CE_U[j]) + CE["g1"] * (d @ d - CE["d0"] ** 2) >= 0
    for i in range(3):
        o = obst[0]
        d = np.subtract(CE_X[i], o.center)
        assert 2 * d @ CE_U[i] + CE["g2"] * (d @ d - o.radius ** 2) >= 0
    assert h_ij(CE_X[0], CE_X[1], DC) == 0.0 and compute_V_i(CE_X[2], CENTER, SIG, EPS) <= 0.0


def test_counterexample_gated_is_structurally_infeasible():
    agents, obst, lam, v2 = _ce_setup()
    allocation, _, _ = _ce_allocation(agents, lam, v2)
    rng = np.random.default_rng(0)
    candidates = [{i: init_fast_state_complete_graph(i, 3, 1) for i in range(3)}, allocation]
    for scale in (1e-3, 1.0, 1e3):
        f = copy.deepcopy(allocation)
        for i in f:
            f[i].z_c, f[i].z_clf = rng.normal(scale=scale), rng.normal(scale=scale)
            for j in f[i].z_pair:
                f[i].z_pair[j].self_component = rng.normal(scale=scale)
        candidates.append(f)
    clf_rows = set()
    for fast in candidates:
        qp = _ce_qp(0, agents, fast, obst, lam, v2, "gated")
        assert lp_verdict(_margin(qp), TOL) == "infeasible"
        k = next(n for n, l in enumerate(qp.row_labels) if l.startswith("clf("))
        clf_rows.add((tuple(qp.A_raw[k]), qp.b_raw[k]))
    assert len(clf_rows) == 1  # agent 0's CLF row does not depend on z at all
    # No allocation exists: the augmented centralized problem with free z is infeasible.
    assert lp_verdict(_centralized_margin(agents, allocation, obst, lam, v2, "gated"), TOL) == "infeasible"


def test_counterexample_communication_admits_a_feasible_allocation():
    agents, obst, lam, v2 = _ce_setup()
    fast, _, _ = _ce_allocation(agents, lam, v2)
    for i in range(3):
        qp = _ce_qp(i, agents, fast, obst, lam, v2, "communication")
        assert lp_verdict(_margin(qp), TOL) == "feasible"
        residual = np.asarray(qp.A_raw) @ CE_U[i] - np.asarray(qp.b_raw)
        assert residual.max() <= 1e-12  # u* itself is admissible in every local QP at this allocation
        assert solve_local_qp_detailed(qp).success
    assert lp_verdict(_centralized_margin(agents, fast, obst, lam, v2, "communication"), TOL) == "feasible"


def test_counterexample_communication_infeasible_at_prescribed_transient_z():
    agents, obst, lam, v2 = _ce_setup()
    allocation, _, _ = _ce_allocation(agents, lam, v2)
    # (a) z = 0 (the initial mismatch state): agent 0 cannot meet its own CLF decrease.
    zero = {i: init_fast_state_complete_graph(i, 3, 1) for i in range(3)}
    assert lp_verdict(_margin(_ce_qp(0, agents, zero, obst, lam, v2, "communication")), TOL) == "infeasible"
    # (b) a transient z that loads the relay agent 2 (inside its target): its z-only CLF row is violated.
    loaded = copy.deepcopy(allocation)
    loaded[2].z_clf += 0.5
    qp2 = _ce_qp(2, agents, loaded, obst, lam, v2, "communication")
    lp2 = local_feasibility(np.asarray(qp2.A_raw), np.asarray(qp2.b_raw), qp2.row_labels, CE["umax"], TOL)
    assert lp_verdict(lp2["s_star"], TOL) == "infeasible"
    assert lp2["degenerate_violated"][0]["label"] == "clf(i=2,inactive)"
    # ... while a feasible allocation exists for the same state.
    for i in range(3):
        assert lp_verdict(_margin(_ce_qp(i, agents, allocation, obst, lam, v2, "communication")), TOL) == "feasible"


# ------------------------------------------------------------------ propagation and metadata
@pytest.mark.parametrize("mode", MODES)
def test_mode_reaches_local_qps_and_fast_dynamics(monkeypatch, mode):
    seen = {"qp": set(), "fast": set()}
    original_build, original_fast = dc.build_local_qp_data, sm.update_fast_state

    def build_spy(**kw):
        seen["qp"].add(kw["clf_coupling"])
        return original_build(**kw)

    def fast_spy(state, **kw):
        seen["fast"].add(kw["clf_coupling"])
        return original_fast(state=state, **kw)

    monkeypatch.setattr(dc, "build_local_qp_data", build_spy)
    monkeypatch.setattr(sm, "update_fast_state", fast_spy)
    cfg = load_config(str(REPO / "configs/distributed_10_agents.yaml"))
    cfg.simulation.steps = 3
    cfg.controller.clf_coupling = mode
    _, _, sim = build(cfg)
    assert _quiet(sim.run).termination_reason == "completed"
    assert seen == {"qp": {mode}, "fast": {mode}}


def test_batch_records_the_mode(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({
        "base_config": "configs/distributed_10_agents.yaml",
        "overrides": {"simulation.steps": 5, "fast_dynamics.substeps": 2},
        "scenarios": [{"id": "default"}, {"id": "comm", "overrides": {"controller.clf_coupling": "communication"}}],
    }))
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = {r["scenario_id"]: r for r in run_batch(spec, batch, workers=2)}
    assert rows["default"]["clf_coupling"] == "gated" and rows["comm"]["clf_coupling"] == "communication"
    assert {r["status"] for r in rows.values()} == {"completed"}
    meta = json.loads((batch / "scenarios" / "comm" / "run" / "run_meta.json").read_text())
    assert meta["clf_coupling"] == "communication"

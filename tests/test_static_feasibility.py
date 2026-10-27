"""Unit tests of the static Level 1-3 feasibility checker (diagnostics.static_feasibility). No simulation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from config import load_config
from diagnostics.certificate import CertModel, CertTolerances, solve_kkt
from diagnostics.static_feasibility import (StaticTolerances, classify_polytope, level1_problem,
                                            static_check)

REPO = Path(__file__).resolve().parents[1]
TOL = StaticTolerances()
BOX = (-np.ones(2), np.ones(2))


def _cfg(tmp_path, positions, obstacles=((-5.0, 1.0, 0.6),), mode="gated", name="cfg", **ctl):
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    raw["simulation"].update(dt=0.001, steps=10)
    raw["agents"].update(num_agents=len(positions), initial_positions=[list(p) for p in positions], init_mode="manual")
    raw["obstacles"]["obstacles"] = [{"center": [x, y], "radius": r} for x, y, r in obstacles]
    raw["controller"].update({"clf_coupling": mode, "alpha_clf": 1.0, "alpha_connectivity": 1.0,
                              "enable_connectivity_constraint": True, **ctl})
    raw["fast_dynamics"].update(tau=0.02, substeps=10)
    path = tmp_path / f"{name}_{mode}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(str(path))


# ------------------------------------------------------------------ classify_polytope on small polytopes
def test_strictly_feasible_polytope_is_F_with_witness():
    v = classify_polytope(np.array([[1.0, 0.0]]), np.array([0.5]), *BOX, ["r"], TOL)
    assert v.verdict == "F" and v.witness_slack <= -TOL.tau_w


def test_infeasible_polytope_is_I_with_reverified_farkas_certificate():
    v = classify_polytope(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([-0.5, -0.6]), *BOX, ["a", "b"], TOL)
    c = v.certificate
    assert v.verdict == "I" and c["type"] == "farkas" and set(c["support"]) == {"a", "b"}
    assert c["lam_min"] >= 0 and c["phi"] == pytest.approx(0.55, abs=1e-9) and c["phi"] > c["err"]


def test_box_term_makes_certificates_with_nonzero_null_residual_valid():
    # x + y <= -50 with |x|, |y| <= 20: lam = 1 has c = M^T lam != 0; phi = -40/sqrt2 + 50/sqrt2 > 0
    v = classify_polytope(np.array([[1.0, 1.0]]), np.array([-50.0]), -20 * np.ones(2), 20 * np.ones(2), ["r"], TOL)
    assert v.verdict == "I" and v.certificate["null_residual_inf"] > 0.1
    assert v.certificate["phi"] == pytest.approx(10 / np.sqrt(2), rel=1e-12)


@pytest.mark.parametrize("gap", [0.0, 1e-12, -1e-12])
def test_touching_and_near_zero_margins_are_unresolved(gap):
    v = classify_polytope(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([0.5, -0.5 + gap]), *BOX, ["a", "b"], TOL)
    assert v.verdict == "U"


@pytest.mark.parametrize("beta, terms, expected", [
    (1.0, None, "F"),            # 0 <= 1
    (0.0, None, "F"),            # identically zero row
    (-1e-3, None, "I"),          # clearly violated
    (-4.314e-8, None, "I"),      # the A-communication z-only CLF row: above tau_0 = 1e-9
    (-5e-10, None, "U"),         # within the existing zero-row tolerance
    (-1e-8, 1e9, "U"),           # above tau_0 but below the cancellation floor kappa*eps*1e9 ~ 2e-5
])
def test_zero_control_rows(beta, terms, expected):
    M = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([beta, 0.5])
    zt = None if terms is None else np.array([terms, 0.0])
    v = classify_polytope(M, b, *BOX, ["zero", "r"], TOL, zero_row_terms=zt)
    assert v.verdict == expected
    assert v.zero_rows[0]["beta"] == beta
    if expected == "I":
        assert v.certificate["type"] == "zero_row" and v.certificate["label"] == "zero"


def test_unbounded_variables_need_an_exact_certificate():
    inf = np.inf
    lo, hi = np.array([-1.0, -inf]), np.array([1.0, inf])
    v = classify_polytope(np.array([[0.0, 1.0], [0.0, -1.0]]), np.array([-1.0, -1.0]), lo, hi, ["a", "b"], TOL)
    assert v.verdict == "I" and v.certificate["null_residual_inf"] == 0.0         # exact cancellation on z
    v2 = classify_polytope(np.array([[0.0, 1.0]]), np.array([-1.0]), lo, hi, ["a"], TOL)
    assert v2.verdict == "F"                                                     # z -> -inf is allowed


# ------------------------------------------------------------------ the 3-agent counterexample (test_clf_coupling)
CE = [(-0.6, 0.0), (0.6, 0.0), (0.0, 0.0)]
CE_OBS = ((-0.4 + 5e-5, 0.0, 0.2),)
CE_CTL = dict(cbf_rate=2.0, safety_distance=0.2, clf_rate=0.01, u_max=20.0)


def _ce_model(tmp_path, mode):
    cfg = _cfg(tmp_path, CE, CE_OBS, mode=mode, name="ce", **CE_CTL)
    return CertModel(cfg)


def test_counterexample_gated_is_structural_B(tmp_path):
    m = _ce_model(tmp_path, "gated")
    r = static_check(m, CE, np.zeros(m.rows(CE).n - m.rows(CE).nw))
    assert r.components["v"] == [[0], [1], [2]] and r.components["h"] == [[0, 1, 2]]
    assert r.level1.verdict == "F" and r.level2.verdict == "I" and r.category == "B"
    assert {"obs(0,0)", "clf{0}"} <= set(r.level2.certificate["support"])
    assert r.level2_augmented.verdict in ("I", "U") and r.level2_repository["verdict"] == "infeasible"
    assert r.level3_verdict == "I" and not [c for c in r.consistency if c.startswith("INCONSISTENT")]


def test_counterexample_communication_feasible_allocation_and_transients(tmp_path):
    m = _ce_model(tmp_path, "communication")
    rows = m.rows(CE)
    r0 = static_check(m, CE, np.zeros(rows.n - rows.nw))
    assert r0.components["v"] == [[0, 1, 2]]
    assert r0.level1.verdict == "F" and r0.level2.verdict == "F" and r0.level2_lifted["verified"]
    assert r0.category == "C" and r0.level3[0].verdict == "I"                     # z = 0: agent 0 blocked
    # the lifted allocation makes every local QP feasible ...
    from diagnostics.static_feasibility import level2_componentwise_problem, lift_allocation
    from scipy.optimize import linprog
    M2, b2, _, comps = level2_componentwise_problem(rows)
    nr = np.linalg.norm(M2, axis=1)
    res = linprog(np.r_[np.zeros(rows.nw), 1.0], A_ub=np.hstack([M2 / nr[:, None], -np.ones((len(b2), 1))]),
                  b_ub=b2 / nr, bounds=[(-20, 20)] * rows.nw + [(-1e9, None)], method="highs")
    z = lift_allocation(rows, res.x[:-1], comps)
    assert static_check(m, CE, z).category == "feasible"
    # ... and loading the relay agent (inside its target) violates its z-only CLF row
    z2 = z.copy()
    z2[rows.izl[2] - rows.nw] += 0.5
    r2 = static_check(m, CE, z2)
    assert r2.category == "C" and r2.level3[2].verdict == "I" and r2.level3[2].certificate["type"] == "zero_row"


# ------------------------------------------------------------------ physical infeasibility and thresholds
FAR = [(-1.25, 0.25), (-1.25, -0.25)]


def test_actuator_limited_state_is_physical_A(tmp_path):
    m = CertModel(_cfg(tmp_path, FAR, clf_rate=2.0, u_max=0.05))
    rows = m.rows(FAR)
    r = static_check(m, FAR, np.zeros(rows.n - rows.nw))
    assert r.level1.verdict == "I" and r.category == "A" and "clf(global)" in r.level1.certificate["support"]
    assert r.level2.verdict == "I" and r.level3_verdict == "I"


def test_head_on_obstacles_are_physical_A(tmp_path):
    S = np.diag([2.0, 0.25])
    obs = []
    for x in FAR:
        d = -S @ np.asarray(x)
        c = np.asarray(x) + 0.7 * d / np.linalg.norm(d)
        obs.append((float(c[0]), float(c[1]), 0.6))
    m = CertModel(_cfg(tmp_path, FAR, tuple(obs), clf_rate=1.0))
    r = static_check(m, FAR)
    assert r.level1.verdict == "I" and r.category == "A"
    assert {"obs(0,0)", "obs(1,1)", "clf(global)"} <= set(r.level1.certificate["support"])


AXIS = [(-1.25, 0.0), (-1.75, 0.0)]   # both CLF gradients along x: a common x-motion leaves pair/conn rows unchanged


def test_actuator_threshold_gives_F_U_I(tmp_path):
    m0 = CertModel(_cfg(tmp_path, AXIS, clf_rate=2.0, u_max=20.0, name="thr0"))
    M, b, lab = level1_problem(m0.rows(AXIS))
    k = lab.index("clf(global)")
    assert np.all(M[k][1::2] == 0.0)                # no y-components in the CLF row
    u_dag = -b[k] / np.abs(M[k]).sum()              # exact threshold here: CLF row vs the box
    out = {}
    for name, f in (("lo", 1 - 1e-3), ("eq", 1.0), ("hi", 1 + 1e-3)):
        m = CertModel(_cfg(tmp_path, AXIS, clf_rate=2.0, u_max=float(u_dag * f), name=f"thr_{name}"))
        out[name] = static_check(m, AXIS).level1.verdict
    assert out == {"lo": "I", "eq": "U", "hi": "F"}


def test_pairwise_row_raises_the_naive_actuator_threshold(tmp_path):
    # side-by-side agents: using the box corners would make them approach each other, so the naive
    # threshold u_dag = gamma*sum W / sum ||a_i||_1 is not sufficient (plan v2 A01/A02 must use an LP threshold)
    m0 = CertModel(_cfg(tmp_path, FAR, clf_rate=2.0, u_max=20.0, name="naive0"))
    M, b, lab = level1_problem(m0.rows(FAR))
    k = lab.index("clf(global)")
    u_dag = -b[k] / np.abs(M[k]).sum()
    m = CertModel(_cfg(tmp_path, FAR, clf_rate=2.0, u_max=float(1.001 * u_dag), name="naive1"))
    r = static_check(m, FAR)
    assert r.level1.verdict == "I" and any(s.startswith("pair") for s in r.level1.certificate["support"])


def test_no_inconsistency_on_random_mismatch(tmp_path):
    pos = [(-0.55, 1.35), (0.0, 1.25), (0.55, 1.35)]
    m = CertModel(_cfg(tmp_path, pos, clf_rate=0.25, name="rand"))
    rows = m.rows(pos)
    rng = np.random.default_rng(0)
    for scale in (0.0, 1e-3, 1e-1, 1.0):
        r = static_check(m, pos, rng.normal(scale=scale, size=rows.n - rows.nw), repository_check=False)
        assert not [c for c in r.consistency if c.startswith("INCONSISTENT")]


# ------------------------------------------------------------------ static cost at N = 8-10 (no simulation)
@pytest.mark.parametrize("cols, rows_, spacing", [(2, 4, 0.42), (2, 5, 0.42), (2, 5, 0.31)])
def test_static_cost_and_correctness_at_scale(tmp_path, cols, rows_, spacing):
    pos = [(-1.35 - c * spacing, (r - (rows_ - 1) / 2) * spacing) for c in range(cols) for r in range(rows_)]
    m = CertModel(_cfg(tmp_path, pos, name=f"lat{cols}{rows_}{int(spacing * 100)}", clf_rate=0.025))
    rows = m.rows(pos)
    kkt = solve_kkt(rows, CertTolerances())
    assert kkt.status == "ok"
    t0 = time.perf_counter()
    r = static_check(m, pos, kkt.p[rows.nw:])
    elapsed = time.perf_counter() - t0
    assert r.level1.verdict == "F" and r.level2.verdict == "F" and r.category == "feasible"
    assert not [c for c in r.consistency if c.startswith(("INCONSISTENT", "DISAGREEMENT"))]
    print(f"\nN={len(pos)} spacing {spacing}: static Levels 1-3 {elapsed * 1e3:.1f} ms "
          f"({ {k: round(v * 1e3, 1) for k, v in r.timings.items()} })")
    assert elapsed < 30.0

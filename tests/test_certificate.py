"""Unit tests of the R6.1 Method B certifier (diagnostics.certificate) on small synthetic states.

No closed-loop simulation is run. Two tests apply a few frozen-state substeps of the unchanged
update_fast_state to check the Lemma 2 inequality on the implementation itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from config import load_config
from controller.fast_dynamics import update_fast_state
from diagnostics.certificate import (CERT_NA, CERT_OK, CERT_PARTIAL, CertModel, CertTolerances, KKTPoint,
                                     _quiet, _U, c0_decision, certify, kappa_p, lemma_constants, m2_assess, m2_margin,
                                     norm_bounds, solve_kkt, spectral_upper, zero_fast_state)
from diagnostics.feasibility import Tolerances, local_feasibility, lp_verdict

REPO = Path(__file__).resolve().parents[1]
TRIPLE = [(-0.55, 1.35), (0.0, 1.25), (0.55, 1.35)]     # O1, I (inside the inner target), O2


def _cfg(tmp_path, positions, mode="gated", obstacles=((-5.0, 1.0, 0.6),), **ctl):
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    raw["simulation"].update(dt=0.001, steps=10)
    raw["agents"].update(num_agents=len(positions), initial_positions=[list(p) for p in positions], init_mode="manual")
    raw["obstacles"]["obstacles"] = [{"center": [x, y], "radius": r} for x, y, r in obstacles]
    raw["controller"].update({"clf_coupling": mode, "alpha_clf": 1.0, "alpha_connectivity": 1.0,
                              "enable_connectivity_constraint": True, **ctl})
    raw["fast_dynamics"].update(tau=0.02, substeps=10)
    path = tmp_path / f"cfg_{mode}_{len(positions)}_{abs(hash(str(ctl))) % 10**6}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(str(path))


@pytest.mark.parametrize("mode", ["gated", "communication"])
def test_rows_match_fast_dynamics_and_local_qp(tmp_path, mode):
    model = CertModel(_cfg(tmp_path, TRIPLE, mode))
    rows = model.rows(TRIPLE)
    dp, dg = rows.check_fast_dynamics(seed=3)
    assert dp < 1e-9 and dg < 1e-9
    assert rows.check_local_qp(seed=3) < 1e-12
    # gated: the inside agent's CLF row is identically zero; communication: it is a z-only row
    k = rows.labels.index("clf(1)")
    assert rows.deg[k] == (mode == "gated")
    if mode == "communication":
        assert not rows.A[k, rows.iw[1]].any() and rows.A[k, rows.nw:].any()


def test_kkt_point_satisfies_kkt_conditions(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE))
    rows = model.rows(TRIPLE)
    kkt = solve_kkt(rows, CertTolerances())
    assert kkt.status == "ok" and kkt.rank == kkt.m
    y = np.zeros(len(rows.b))
    y[kkt.active] = kkt.y_active
    g = rows.A @ kkt.p + rows.b
    assert np.max(np.abs(rows.P * kkt.p + rows.A.T @ y)) < 1e-12           # stationarity
    assert np.all(g[~rows.deg] <= 1e-12)                                     # primal feasibility
    assert np.all(kkt.y_active > 0) and np.all(np.abs(g[kkt.active]) < 1e-12)  # complementarity, strict
    assert {rows.labels[i] for i in np.flatnonzero(kkt.active)} == {"clf(0)", "clf(2)"}
    assert kkt.clarabel_diff < 1e-6


def test_communication_mode_inside_agent_has_zero_margin(tmp_path):
    # the inside agent's z-only CLF row is active at z*, so rho(z*) = 0 exactly (R6.1 F3 with a support-1 ray)
    model = CertModel(_cfg(tmp_path, TRIPLE, mode="communication", clf_rate=0.25))
    rows = model.rows(TRIPLE)
    kkt = solve_kkt(rows, CertTolerances())
    rho, per = m2_margin(rows, kkt.p[rows.nw:], CertTolerances())
    assert rows.labels.index("clf(1)") in np.flatnonzero(kkt.active) and rho <= 1e-9 and per[1] <= 1e-9
    m2 = m2_assess(rows, kkt.p[rows.nw:], CertTolerances())
    assert m2.status == "upper_bound_below_tol" and m2.lower == 0.0 and 0.0 <= m2.upper <= 1e-9 and m2.per_upper[1] <= 1e-9
    assert m2.structural_zero_rows[1] == "clf(1)"            # the z-only row of the inside agent (support-1 ray)


def test_m2_assessment_brackets_rho_and_never_reports_a_negative_margin(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE, mode="gated", clf_rate=0.25))
    rows = model.rows(TRIPLE)
    kkt = solve_kkt(rows, CertTolerances())
    zstar = kkt.p[rows.nw:]
    m2 = m2_assess(rows, zstar, CertTolerances())
    assert m2.status == "certified_positive" and 0 < m2.lower <= m2.upper < math.inf
    assert m2.lower == m2.candidate
    tol = Tolerances()
    infeasible_seen = False
    for k, ub in enumerate(m2.per_upper):        # beyond the upper bound, along some direction, Z_feas is left
        if not math.isfinite(ub):
            continue
        for t in (1.01, 1.5):
            ok = []
            for d in np.eye(zstar.size):
                for sgn in (1.0, -1.0):
                    p = kkt.p.copy()
                    p[rows.nw:] = zstar + sgn * t * max(ub, 1e-6) * d * math.sqrt(zstar.size)
                    fs = rows.from_vec(p, np.zeros(len(rows.b)))
                    qp = rows.local_qp(k, fs)
                    lp = local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels,
                                           model.cfg.controller.u_max, tol)
                    ok.append(lp_verdict(lp["s_star"], tol) == "feasible")
            infeasible_seen |= not all(ok)
    assert infeasible_seen
    # a nonpositive candidate is reported as 0, never as a negative margin
    neg = m2_assess(rows, zstar + 5.0 * np.random.default_rng(2).normal(size=zstar.size), CertTolerances())
    assert neg.candidate < 0 and neg.lower == 0.0 and neg.upper == 0.0 and neg.status == "z_outside_Zfeas"


def test_spectral_upper_bound_is_valid_and_tight():
    rng = np.random.default_rng(4)
    for shape in ((1, 5), (3, 7), (6, 30), (12, 40)):
        A = rng.normal(size=shape) * 10.0 ** rng.uniform(-3, 3, size=(shape[0], 1))
        sv = np.linalg.svd(A, compute_uv=False)
        ub = spectral_upper(A)
        assert sv[0] <= ub <= np.sqrt(np.sum(A ** 2)) * (1 + 1e-12)
        assert ub <= sv[0] * (1 + 1e-10)
        v = np.linalg.svd(A)[2][0]
        assert np.linalg.norm(A @ v) <= ub                              # a rigorous lower bound on ||A||_2
        a, a_lo, smin = norm_bounds(A, "spectral")
        assert a == ub and a_lo <= sv[0] <= a and 0 <= smin <= sv[shape[0] - 1]
        af, af_lo, sminf = norm_bounds(A, "frobenius")
        assert af == af_lo >= a * (1 - 1e-12) and sminf <= sv[shape[0] - 1] * (1 + 1e-12)
    assert norm_bounds(np.zeros((0, 4)), "spectral") == (0.0, 0.0, math.inf)


def test_c0_decision_separates_ok_fail_and_ambiguous():
    r, pmin, pmax = 0.005, 0.5, 1.0
    kap, s, _ = c0_decision(r, pmin, pmax, 5.0, 5.0)
    assert s == "ok" and kap > 0
    a0 = 10.0                                                            # kappa_p(a) crosses zero near here
    lo, hi = 1.0, 40.0
    for _ in range(200):                                                 # bisection for the root of kappa_p
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if kappa_p(r, pmin, pmax, mid)[0] > 0 else (lo, mid)
    a0 = hi
    assert c0_decision(r, pmin, pmax, a0 * 1.01, a0 * 0.99)[1] == "ambiguous"   # undecided within [a_lo, a]
    assert c0_decision(r, pmin, pmax, a0 * 1.02, a0 * 1.01)[1] == "fail"
    assert c0_decision(r, pmin, pmax, a0 * 0.99, a0 * 0.98)[1] == "ok"
    assert c0_decision(r, pmin, pmax, 1.0 / r, 1.0 / r)[1] == "fail"             # r a = 1


def test_spectral_switching_bound_satisfies_remark_6_1(tmp_path):
    seq = [[(x - 0.004 * k, y + 0.001 * k * x) for x, y in TRIPLE] for k in range(5)]
    model = CertModel(_cfg(tmp_path, TRIPLE, clf_rate=0.025))
    s = certify(model, seq, zero_fast_state(model), CertTolerances(norm="spectral")).steps
    assert len(s["a"]) == 5 and set(s["c0_status"]) == {"ok"}
    for k in range(1, 5):
        assert s["a"][k] <= s["a"][k - 1] + s["dA"][k]                  # a_{k+1} <= a_k + d_k exactly
        rows_k, rows_p = model.rows(seq[k]), model.rows(seq[k - 1])
        kk = solve_kkt(rows_k, CertTolerances())
        kp = solve_kkt(rows_p, CertTolerances())
        dA = rows_k.A[kk.active] - rows_p.A[kp.active]
        assert np.linalg.svd(dA, compute_uv=False)[0] <= s["dA"][k]     # d_k bounds ||Delta A||_2
        assert s["a"][k] >= np.linalg.svd(rows_k.A[kk.active], compute_uv=False)[0]


def test_m2_margin_is_a_valid_lower_bound_on_the_implementation(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE, mode="gated", clf_rate=0.25))
    rows = model.rows(TRIPLE)
    kkt = solve_kkt(rows, CertTolerances())
    zstar = kkt.p[rows.nw:]
    rho, _ = m2_margin(rows, zstar, CertTolerances())
    assert 0 < rho < math.inf
    rng = np.random.default_rng(0)
    tol = Tolerances()
    for _ in range(20):
        d = rng.normal(size=zstar.size)
        z = zstar + 0.99 * rho * d / np.linalg.norm(d)
        p = kkt.p.copy()
        p[rows.nw:] = z
        fs = rows.from_vec(p, np.zeros(len(rows.b)))
        for i in range(rows.N):
            qp = rows.local_qp(i, fs)
            lp = local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels,
                                   model.cfg.controller.u_max, tol)
            assert lp_verdict(lp["s_star"], tol) == "feasible"


def test_lemma2_nonincrease_on_the_implemented_fast_dynamics(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE, clf_rate=0.25))
    rows = model.rows(TRIPLE)
    tol = CertTolerances()
    kkt = solve_kkt(rows, tol)
    lc = lemma_constants(rows, kkt, model.r, tol)
    rho, _ = m2_margin(rows, kkt.p[rows.nw:], tol)
    assert lc.c0_ok
    E_adm = min(lc.C1, lc.C2, rho)
    AA = rows.A[kkt.active]
    rng = np.random.default_rng(1)
    ep = rng.normal(size=rows.n)
    ey = rng.normal(size=kkt.m)
    e_norm = math.sqrt(_U(ep, ey, AA, model.r))
    target = 0.5 * E_adm * math.sqrt(1 - model.r * lc.a)          # U <= R^2 with E(R) = E_adm / 2
    target = min(target, 0.5 * float(kkt.y_active.min()))         # keep y >= 0
    ep, ey = ep * target / e_norm, ey * target / e_norm
    y = np.zeros(len(rows.b))
    y[kkt.active] = kkt.y_active + ey
    fs = rows.from_vec(kkt.p + ep, y)
    U_prev = _U(ep, ey, AA, model.r)
    dt = model.cfg.simulation.dt / model.M
    for _ in range(3 * model.M):
        fs = {i: _quiet(update_fast_state, **rows._fast_kwargs(i, fs, dt)) for i in range(rows.N)}
        p, yv = rows.to_vec(fs)
        assert np.all(yv[kkt.inactive] == 0.0)
        U = _U(p - kkt.p, yv[kkt.active] - kkt.y_active, AA, model.r)
        assert U <= U_prev * (1 + 1e-12) + 1e-30
        U_prev = U


def test_lemma_constants_conventions_for_empty_sets_and_zero_rows():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    rows = SimpleNamespace(A=A, b=np.array([-0.5, -1.0]), P=np.ones(2), row_norm=np.linalg.norm(A, axis=1),
                           labels=["zero_inactive", "inactive"])
    kkt = KKTPoint(status="ok", p=np.zeros(2), y_active=np.zeros(0), active=np.array([False, False]),
                   inactive=np.array([True, True]))
    lc = lemma_constants(rows, kkt, 0.005, CertTolerances())
    assert lc.C2 == math.inf and lc.smin == math.inf          # A empty
    assert lc.C1 == pytest.approx(1.0) and lc.C1_label == "inactive"   # zero row contributes +inf
    kkt2 = KKTPoint(status="ok", p=np.zeros(2), y_active=np.zeros(0), active=np.array([False, False]),
                    inactive=np.array([True, False]))
    assert lemma_constants(rows, kkt2, 0.005, CertTolerances()).C1 == math.inf
    assert lc.c == pytest.approx(lc.kappa_p)                   # m = 0: rate kappa_p / (1 + r*0)


@pytest.mark.parametrize("norm", ["spectral", "frobenius"])
def test_certify_static_state_and_prop6_bound(tmp_path, norm):
    seq = [[(x - 0.002 * k, y) for x, y in TRIPLE] for k in range(6)]
    model = CertModel(_cfg(tmp_path, TRIPLE, clf_rate=0.025))
    cert = certify(model, seq, zero_fast_state(model), CertTolerances(norm=norm))
    assert cert.results["contraction"].status == CERT_OK
    s = cert.steps
    # R6.1 Prop 6 (non-increase; Remark 6.1 holds for both norm conventions): E_k >= |e_0| + sum_{j<k} delta_j
    cum = np.cumsum(s["delta"])
    assert np.all(np.asarray(s["E_ni"]) >= cert.initial["e0"] + cum - 1e-15)


def test_partition_change_is_reported(tmp_path):
    # the connector leaves the inner target (q from -0.11 to +0.06): its gated CLF row becomes nondegenerate
    seq = [[TRIPLE[0], (0.0, 1.25 + 0.05 * k), TRIPLE[2]] for k in range(6)]
    model = CertModel(_cfg(tmp_path, TRIPLE))
    cert = certify(model, seq, zero_fast_state(model))
    r = cert.results["non_increase"]
    assert r.status == CERT_PARTIAL and r.reason == "PART" and r.reason_kind == "prerequisite"
    assert r.certified_states == r.k_fail >= 1


def test_nonzero_inactive_dual_is_a_missing_prerequisite(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE))
    chi0 = zero_fast_state(model)
    chi0[0].y_obs[0] = 0.1
    r = certify(model, [TRIPLE], chi0).results["contraction"]
    assert r.status == CERT_NA and r.reason == "DUAL0"


def test_augmented_infeasibility_is_reported_as_infeasibility(tmp_path):
    pos = [(-1.25, 0.25), (-1.25, -0.25)]                     # far, actuator-limited (A01-type)
    model = CertModel(_cfg(tmp_path, pos, clf_rate=2.0, u_max=0.05))
    r = certify(model, [pos], zero_fast_state(model)).results["contraction"]
    assert r.status == CERT_NA and r.reason == "L2_INFEASIBLE" and r.reason_kind == "infeasibility"


def test_unsupported_alpha_is_refused(tmp_path):
    model = CertModel(_cfg(tmp_path, TRIPLE, alpha_clf=0.5))
    r = certify(model, [TRIPLE], zero_fast_state(model)).results["contraction"]
    assert r.status == CERT_NA and r.reason == "UNSUPPORTED"


def test_spectral_bound_is_not_larger_than_frobenius(tmp_path):
    seq = [[(x - 0.002 * k, y) for x, y in TRIPLE] for k in range(4)]
    model = CertModel(_cfg(tmp_path, TRIPLE, clf_rate=0.025))
    fro = certify(model, seq, zero_fast_state(model), CertTolerances(norm="frobenius")).steps["E_ni"]
    spe = certify(model, seq, zero_fast_state(model), CertTolerances(norm="spectral")).steps["E_ni"]
    assert np.all(np.asarray(spe) <= np.asarray(fro) * (1 + 1e-12))

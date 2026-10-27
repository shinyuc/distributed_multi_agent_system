"""Frozen pre-convergence warm start (study plan v2; the protocol of the D3 experiment, clfc/run_d3.py).

The physical state is held at x_0, no local QP is solved or applied, and the unchanged fast dynamics run from the
repository zero initialization with exactly the simulator's keyword set (M substeps of dt/M per outer-step
equivalent, synchronous Jacobi). Stop at the first outer step where
  (1) every local QP is LP-feasible (repository verifier) and OSQP-optimal at the current mismatch,
  (2) every inactive dual (rows inactive at the frozen KKT point) is exactly 0.0,
  (3) the lemma radius E = sqrt(U)/sqrt(1 - r a) is <= the threshold (a = ||A_A||_F, as in the D3 experiment;
      the stopping rule is part of the scenario definition and does not follow the certificate's default norm), and
  (4) C0 holds and E < E_adm(x_0) = min(C1, C2, rho_M2).
This is a static computation at a fixed physical state, not a closed-loop simulation.
"""

from __future__ import annotations

import math

import numpy as np

from controller.fast_dynamics import update_fast_state
from controller.local_qp import solve_local_qp_detailed
from diagnostics.certificate import (CertModel, CertTolerances, _quiet, _U, lemma_constants, m2_margin, solve_kkt,
                                     zero_fast_state)
from diagnostics.feasibility import Tolerances, local_feasibility, lp_verdict


def frozen_warmup(model: CertModel, x0, rule: dict, max_steps: int = 5000):
    """rule: {"type": "E_abs", "value": v} or {"type": "E_rel", "factor": f} (threshold f * E_adm(x_0))."""
    tol = CertTolerances(norm="frobenius")   # the D3 protocol's convention; part of the scenario definition
    rows = model.rows(x0)
    kkt = solve_kkt(rows, tol)
    if kkt.status != "ok":
        raise ValueError(f"frozen KKT point unavailable at x0: {kkt.status} {kkt.note}")
    lc = lemma_constants(rows, kkt, model.r, tol)
    rho, _ = m2_margin(rows, kkt.p[rows.nw:], tol)
    E_adm = min(lc.C1, lc.C2, rho)
    E_stop = rule["value"] if rule["type"] == "E_abs" else rule["factor"] * E_adm
    AA = rows.A[kkt.active]
    ystar = np.zeros(len(rows.b))
    ystar[kkt.active] = kkt.y_active
    u_max = model.cfg.controller.u_max
    vt = Tolerances()
    dt = model.cfg.simulation.dt / model.M
    fs = zero_fast_state(model)
    log = []
    for k in range(max_steps + 1):
        p, y = rows.to_vec(fs)
        inz = bool(np.all(y[kkt.inactive] == 0.0))
        ep, ey = p - kkt.p, y[kkt.active] - kkt.y_active
        E = math.sqrt(max(_U(ep, ey, AA, model.r), 0.0)) / math.sqrt(1 - model.r * lc.a) if inz else math.inf
        e = float(np.linalg.norm(np.r_[p - kkt.p, np.where(rows.deg, 0.0, y - ystar)]))
        c3 = E <= E_stop
        c4 = lc.c0_ok and E < E_adm
        c1 = None
        if c3 or k % 50 == 0:
            c1 = True
            for i in range(rows.N):
                qp = rows.local_qp(i, fs)
                lp = local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels, u_max, vt)
                st = solve_local_qp_detailed(qp).status
                c1 &= lp_verdict(lp["s_star"], vt) == "feasible" and st in ("solved", "optimal")
        log.append((k, e, E, inz, c1))
        if c1 and inz and c3 and c4:
            return fs, {"stop_step": k, "frozen_time_s": k * model.cfg.simulation.dt, "e": e, "E": E, "E_stop": E_stop,
                        "E_adm_x0": E_adm, "rule": rule, "inactive_duals_zero": inz, "log_every50": log[::50]}
        for _ in range(model.M):
            fs = {i: _quiet(update_fast_state, **rows._fast_kwargs(i, fs, dt)) for i in range(rows.N)}
    raise RuntimeError(f"warm-up threshold not reached within {max_steps} outer-step equivalents (E = {E:.3e})")

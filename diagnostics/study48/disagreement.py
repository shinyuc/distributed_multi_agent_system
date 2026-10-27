"""Diagnosis of Level-2 disagreements between the componentwise check (Prop 2) and the solver-based checks.

    python -m diagnostics.study48.disagreement --study STUDY_DIR --ids B04,B12

For a scenario whose componentwise Level 2 is F while the augmented LP / the repository solver / CLARABEL do not
confirm it, this separates mathematical feasibility from solver failure:

* the explicit allocation of the Prop 2 proof (lift_allocation) is re-verified in floating point with a rounding
  error bound AND in exact rational arithmetic on the floating-point row data (fractions.Fraction);
* the coupling weights (h, gated v) are listed exactly; nothing is rounded to zero;
* a rigorous lower bound on ||z||_inf over ALL feasible allocations: for any multipliers lam >= 0,
  lam^T (M_w w + M_z z) <= lam^T beta and |w| <= u_max force ||z||_inf >= phi_w / ||lam^T M_z||_1 with
  phi_w = sum_j min(c_j lo_j, c_j hi_j) - lam^T beta over the control columns; lam is taken from the LP of the
  problem with the tiny coupling entries removed, but the bound is evaluated with the true matrix;
* the solver mechanism: the normalized augmented matrix's entries below HiGHS's small_matrix_value (1e-9) are
  listed, and the margin LP with exactly those entries set to zero is compared with the repository result;
* a column-scaled augmented LP (each z column divided by its largest coefficient) solved by HiGHS, with its
  witness re-verified exactly;
* conditioning: coefficient dynamic range, condition numbers of the component Laplacians used by the lift.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from config import load_config
from diagnostics.certificate import CertModel, CertTolerances, solve_kkt
from diagnostics.static_feasibility import (EPS, graphs, level2_augmented_problem, level2_componentwise_problem,
                                            static_check)

HIGHS_SMALL = 1e-9          # HiGHS option small_matrix_value (default): smaller |entries| are dropped


def _exact_max_residual(M: np.ndarray, beta: np.ndarray, x: np.ndarray) -> tuple[float, str]:
    """max_k (M_k x - beta_k) / ||M_k|| evaluated exactly (rational) on the floating-point data; the division by
    the (float) row norm only rescales a sign-exact quantity."""
    worst, worst_k = None, -1
    xf = [Fraction(float(v)) for v in x]
    for k in range(M.shape[0]):
        acc = -Fraction(float(beta[k]))
        for j in np.flatnonzero(M[k]):
            acc += Fraction(float(M[k, j])) * xf[j]
        val = float(acc) / max(float(np.linalg.norm(M[k])), 1e-300)
        if worst is None or val > worst:
            worst, worst_k = val, k
    return worst, worst_k


def _margin_lp(Mh, bh, lo, hi, s_lower=-1e9):
    n = Mh.shape[1]
    bounds = [(None if not np.isfinite(l) else l, None if not np.isfinite(h) else h) for l, h in zip(lo, hi)]
    return linprog(np.r_[np.zeros(n), 1.0], A_ub=np.hstack([Mh, -np.ones((len(bh), 1))]), b_ub=bh,
                   bounds=bounds + [(s_lower, None)], method="highs")


def diagnose(model: CertModel, positions) -> dict:
    rows = model.rows(positions)
    u = float(model.cfg.controller.u_max)
    rep = static_check(model, positions, np.zeros(rows.n - rows.nw))
    idx = {l: k for k, l in enumerate(rows.labels)}
    out: dict = {"levels": {"componentwise": rep.level2.verdict, "augmented": rep.level2_augmented.verdict,
                            "augmented_note": rep.level2_augmented.note,
                            "augmented_certificate": rep.level2_augmented.certificate,
                            "repository": rep.level2_repository}, "graphs": graphs(rows)}
    # coupling weights, exactly as in the rows
    v = {f"{i}-{j}": float(-rows.A[idx[f"clf({i})"], rows.izl[j]]) for i in range(rows.N) for j in range(rows.N)
         if j != i and rows.A[idx[f"clf({i})"], rows.izl[j]] != 0}
    h = {f"{i}-{j}": float(w) for (i, j), w in rows.H.items() if w > 0}
    out["coupling"] = {"v": v, "h": h, "min_v": min(v.values()) if v else None, "min_h": min(h.values()) if h else None,
                       "sigma": [float(rows.sig[i]) for i in range(rows.N)], "q": [float(rows.q[i]) for i in range(rows.N)]}
    # lifted allocation
    Ma, ba, lo, hi, la = level2_augmented_problem(rows)
    lift = dict(rep.level2_lifted)
    if lift.get("z") is not None:
        x = np.r_[np.asarray(lift["u"]), np.asarray(lift["z"])]
        ex, ek = _exact_max_residual(Ma, ba, x)
        z = np.asarray(lift["z"])
        lift.update(exact_max_normalized_residual=ex, exact_binding_row=la[ek],
                    exact_verified=bool(ex <= -1e-9), z_l2_norm=float(np.linalg.norm(z)))
    out["lifted"] = lift
    # normalized matrix, entries below the HiGHS threshold, and the LP with exactly those entries removed
    nr = np.linalg.norm(Ma, axis=1)
    keep = nr > 0
    Mh, bh = Ma[keep] / nr[keep, None], ba[keep] / nr[keep]
    nz = np.abs(Mh[Mh != 0])
    small = (np.abs(Mh) < HIGHS_SMALL) & (Mh != 0)
    lab_k = [l for l, kk in zip(la, keep) if kk]
    out["conditioning"] = {"normalized_entry_min": float(nz.min()), "normalized_entry_max": float(nz.max()),
                           "dynamic_range": float(nz.max() / nz.min()),
                           "entries_below_highs_small_matrix_value": int(small.sum()),
                           "rows_with_small_entries": sorted({lab_k[k] for k in np.argwhere(small)[:, 0]})}
    Md = Mh.copy()
    Md[small] = 0.0
    r_full, r_drop = _margin_lp(Mh, bh, lo, hi), _margin_lp(Md, bh, lo, hi)
    out["solver_mechanism"] = {"lp_margin_full_matrix": float(r_full.x[-1]), "lp_margin_small_entries_removed":
                               float(r_drop.x[-1]), "repository_s_star": rep.level2_repository.get("s_star"),
                               "identical_full_vs_removed": bool(r_full.x[-1] == r_drop.x[-1])}
    # rigorous lower bound on ||z||_inf over all feasible allocations (multipliers of the decoupled LP)
    lam = np.maximum(-np.asarray(r_drop.ineqlin.marginals, float), 0.0)
    lam = lam / lam.sum() if lam.sum() > 0 else lam
    c = Mh.T @ lam
    nwc = rows.nw
    cw, cz = c[:nwc], c[nwc:]
    phi_w = float(np.sum(np.minimum(-u * cw, u * cw))) - float(lam @ bh)
    err = 4 * (Mh.shape[0] + Mh.shape[1]) * EPS * (u * float(np.abs(Mh[:, :nwc]).T @ lam @ np.ones(nwc))
                                                   + float(lam @ np.abs(bh)) + 1.0)
    czl1 = float(np.abs(cz).sum()) + 4 * Mh.shape[0] * EPS * float(np.abs(Mh[:, nwc:]).T @ lam @ np.ones(Mh.shape[1] - nwc))
    out["mismatch_lower_bound"] = {"phi_w": phi_w, "err": err, "lam_Mz_l1": czl1,
                                   "z_inf_lower_bound": (phi_w - err) / czl1 if phi_w > err and czl1 > 0 else 0.0,
                                   "support": [lab_k[k] for k in np.flatnonzero(lam > 1e-12)]}
    # column-scaled augmented LP (z columns divided by their largest |coefficient|), witness re-verified exactly
    colmax = np.max(np.abs(Mh[:, nwc:]), axis=0)
    scale = np.where(colmax > 0, colmax, 1.0)
    Ms = Mh.copy()
    Ms[:, nwc:] = Mh[:, nwc:] / scale
    rs = _margin_lp(Ms, bh, lo, hi)
    xs = rs.x[:-1].copy()
    xs[nwc:] = xs[nwc:] / scale
    xs[:nwc] = np.clip(xs[:nwc], -u, u)
    exs, eks = _exact_max_residual(Ma, ba, xs)
    out["column_scaled_lp"] = {"status": int(rs.status), "lp_margin": float(rs.x[-1]),
                               "witness_exact_max_normalized_residual": exs, "witness_verified": bool(exs <= -1e-9),
                               "witness_z_inf_norm": float(np.max(np.abs(xs[nwc:])))}
    # component Laplacians used by the lift (lstsq)
    comps = level2_componentwise_problem(rows)[3]
    conds = {}
    for kind, key, izmap in (("conn", "h", rows.izc), ("clf", "v", rows.izl)):
        for C in comps[key]:
            if len(C) >= 2:
                L = rows.A[np.ix_([idx[f"{kind}({i})"] for i in C], [izmap[i] for i in C])]
                sv = np.linalg.svd(L, compute_uv=False)
                conds[f"{kind}{C}"] = {"sigma_max": float(sv[0]), "sigma_min_nonzero": float(sv[sv > 0][-2]) if
                                       (sv > 0).sum() >= 2 else float(sv[sv > 0][-1])}
    out["component_laplacians"] = conds
    kkt = solve_kkt(rows, CertTolerances())
    out["kkt"] = {"status": kkt.status, "note": kkt.note,
                  "interpretation": ("solver failure: the augmented QP is feasible (exact witness) and strictly "
                                     "convex, so its KKT point exists; ||z*||_inf >= the mismatch lower bound")
                  if kkt.status in ("L2_INFEASIBLE", "SOLVER") and lift.get("exact_verified") else ""}
    lb, zl = out["mismatch_lower_bound"]["z_inf_lower_bound"], lift.get("z_inf_norm")
    out["classification"] = {
        "mathematical_level2": "F" if lift.get("exact_verified") else rep.level2.verdict,
        "evidence": "explicit allocation verified in exact rational arithmetic on the floating-point rows"
        if lift.get("exact_verified") else "componentwise verdict only",
        "required_mismatch_inf_norm": [lb, zl],
        "numerical_status": ("ill-conditioned (near-structural): every feasible allocation needs "
                             f"||z||_inf >= {lb:.3e}; double-precision solvers with default thresholds treat the "
                             "coupling as absent") if lb > 1e6 else "well-conditioned",
        "solver_disagreement_explained": bool(out["solver_mechanism"]["identical_full_vs_removed"]),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose Level-2 solver disagreements")
    ap.add_argument("--study", required=True)
    ap.add_argument("--ids", required=True)
    a = ap.parse_args(argv)
    study = Path(a.study)
    od = study / "diagnostics"
    od.mkdir(exist_ok=True)
    for sid in a.ids.split(","):
        cfg = load_config(str(study / "configs" / f"{sid}.yaml"))
        d = diagnose(CertModel(cfg), [list(map(float, p)) for p in cfg.agents.initial_positions])
        (od / f"{sid}_level2_disagreement.json").write_text(json.dumps(d, indent=2, default=str))
        np.savez(od / f"{sid}_lifted_allocation.npz", u=np.asarray(d["lifted"].get("u", [])),
                 z=np.asarray(d["lifted"].get("z", [])))
        c = d["classification"]
        print(f"[disagreement] {sid}: math L2 {c['mathematical_level2']}; ||z||_inf in [{c['required_mismatch_inf_norm'][0]:.3e}, "
              f"{c['required_mismatch_inf_norm'][1]:.3e}]; solver drop explained {c['solver_disagreement_explained']}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

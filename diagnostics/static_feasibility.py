"""Static feasibility checker for Levels 1-3 with verified F / I / U classifications (no simulation).

Levels (R6.1 §1, Props 1-2; study plan v2 §3):
* Level 1, the original centralized problem: summed pairwise CBF rows for h_ij > 0, obstacle rows, one global
  connectivity row, one global CLF row, actuator box. Built by summing the augmented rows of
  :class:`diagnostics.certificate.AugmentedRows` (identical to the fast-dynamics rows), so Prop 1 holds by
  construction.
* Level 2, existence of a feasible mismatch allocation. Official verdict: the componentwise problem of
  R6.1 Prop 2 (u only, bounded, so Farkas certificates can be verified robustly). Cross-checks: the augmented
  LP with z free (model rows) and the repository's ``centralized_feasibility`` (scaled rows). A feasible
  componentwise witness is lifted to an explicit allocation z (Prop 2 proof) and re-verified on the augmented rows.
* Level 3, every local QP at the supplied mismatch z, using the repository's local-QP rows (``A_raw``, ``b_raw``)
  and the existing conventions (rows normalized by their control part; actuator rows as hard bounds).

Verdicts for a polytope {v : M v <= beta, lo <= v <= hi}:
* F: an LP witness v (clipped into the bounds) whose raw residuals, recomputed, give a normalized slack
  <= -tau_w on every row with a nonzero control part, and every zero-control row satisfied (beta_r >= 0).
* I: a Farkas certificate lam >= 0 (from the LP duals, negatives clipped) with
  phi(lam) = sum_j min(c_j lo_j, c_j hi_j) - lam^T beta > tau_c + err, c = M^T lam, using the normalized rows.
  This is valid for ANY lam >= 0 (no exact null-space condition needed) because the variables are bounded; for
  unbounded variables it requires c_j == 0 exactly. err bounds the floating-point error of evaluating phi.
  A zero-control row 0 <= beta_r with beta_r < -max(tau_0, floor_r) is an I certificate by itself; tau_0 is the
  existing zero-row tolerance (1e-9) and floor_r = kappa * eps * (sum of the magnitudes of the terms that formed
  beta_r), a cancellation floor.
* U: neither could be established (small margins, zero-row values within tolerance, LP failure, unbounded
  variables without an exact certificate, ...).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from diagnostics.certificate import AugmentedRows, CertModel, _quiet
from diagnostics.feasibility import ACTUATOR_GROUPS, Tolerances, centralized_feasibility, local_feasibility, row_group

EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class StaticTolerances:
    tau_w: float = 1e-9         # witness: required normalized slack (control units) on rows with a control part
    tau_c: float = 1e-10        # certificate: required phi beyond its floating-point error bound
    tau_0: float = 1e-9         # zero-control rows: existing zero-row tolerance (diagnostics.feasibility)
    kappa: float = 100.0        # cancellation-floor multiplier
    zero_row_norm: float = 1e-150
    s_lower: float = -1e9


@dataclass
class Verdict:
    verdict: str                          # "F", "I", "U"
    lp_margin: float = math.nan           # LP margin s* (normalized rows; < 0 feasible)
    witness_slack: float = math.nan       # max normalized residual of the re-verified witness (<= -tau_w for F)
    certificate: dict | None = None       # Farkas evidence for I (support labels, lam, phi, err) or zero-row evidence
    zero_rows: list = field(default_factory=list)   # zero-control rows: label, beta, floor, status
    normalization: str = ""
    tolerances: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def classify_polytope(M: np.ndarray, beta: np.ndarray, lo: np.ndarray, hi: np.ndarray, labels: list[str],
                      tol: StaticTolerances, zero_row_terms: np.ndarray | None = None,
                      normalization: str = "rows / ||control part||_2") -> Verdict:
    M, beta = np.asarray(M, float), np.asarray(beta, float)
    n = M.shape[1] if M.ndim == 2 else len(lo)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    tdict = asdict(tol)
    norms = np.linalg.norm(M, axis=1) if M.size else np.zeros(0)
    zero = norms <= tol.zero_row_norm
    terms = np.abs(beta) if zero_row_terms is None else np.maximum(np.asarray(zero_row_terms, float), np.abs(beta))
    zinfo, zero_viol, zero_amb = [], None, []
    for k in np.flatnonzero(zero):
        floor = max(tol.tau_0, tol.kappa * EPS * float(terms[k]))
        if beta[k] >= 0:
            st = "satisfied"
        elif beta[k] < -floor:
            st = "violated"
            if zero_viol is None or beta[k] < beta[zero_viol]:
                zero_viol = k
        else:
            st = "within_tolerance"
            zero_amb.append(labels[k])
        zinfo.append({"label": labels[k], "beta": float(beta[k]), "floor": floor, "status": st})
    if zero_viol is not None:
        return Verdict("I", certificate={"type": "zero_row", "label": labels[zero_viol], "beta": float(beta[zero_viol]),
                                         "phi": float(-beta[zero_viol]),
                                         "floor": next(z["floor"] for z in zinfo if z["label"] == labels[zero_viol])},
                       zero_rows=zinfo, normalization=normalization, tolerances=tdict,
                       note="zero-control row 0 <= beta_r violated beyond max(tau_0, cancellation floor)")
    keep = ~zero
    Mh = M[keep] / norms[keep, None] if keep.any() else np.zeros((0, n))
    bh = beta[keep] / norms[keep] if keep.any() else np.zeros(0)
    klab = [labels[k] for k in np.flatnonzero(keep)]
    bounds = [(None if not math.isfinite(l) else l, None if not math.isfinite(h) else h) for l, h in zip(lo, hi)]
    if Mh.shape[0] == 0:
        v = np.clip(np.zeros(n), lo, hi)
        return Verdict("U" if zero_amb else "F", lp_margin=-math.inf, witness_slack=-math.inf, zero_rows=zinfo,
                       normalization=normalization, tolerances=tdict,
                       note="no rows with a control part" + (f"; zero rows within tolerance {zero_amb}" if zero_amb else ""))
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.hstack([Mh, -np.ones((Mh.shape[0], 1))]), b_ub=bh,
                  bounds=bounds + [(tol.s_lower, None)], method="highs")
    if res.status != 0 or res.x is None:
        return Verdict("U", zero_rows=zinfo, normalization=normalization, tolerances=tdict,
                       note=f"LP failed: status {res.status} {res.message}")
    s = float(res.x[-1])
    v = np.clip(res.x[:-1], np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf))
    resid = (M[keep] @ v - beta[keep]) / norms[keep]
    wslack = float(resid.max())
    out = Verdict("U", lp_margin=s, witness_slack=wslack, zero_rows=zinfo, normalization=normalization,
                  tolerances=tdict)
    if s < 0:
        if wslack <= -tol.tau_w and not zero_amb:
            out.verdict = "F"
            out.note = "re-verified witness"
        else:
            out.note = (f"LP margin {s:.3e} but witness slack {wslack:.3e} > -tau_w" if wslack > -tol.tau_w
                        else f"zero-control rows within tolerance: {zero_amb}")
        return out
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is None:
        out.note = "no LP duals available"
        return out
    lam = np.maximum(-np.asarray(marg, float), 0.0)
    if lam.sum() <= 0:
        out.note = "LP duals vanish"
        return out
    lam = lam / lam.sum()
    c = Mh.T @ lam
    unb = ~(np.isfinite(lo) & np.isfinite(hi))
    if np.any(c[unb] != 0.0):
        out.note = (f"LP margin {s:.3e} > 0 but the certificate has nonzero coefficients "
                    f"(max {np.max(np.abs(c[unb])):.1e}) on unbounded variables: not verifiable")
        # The same multipliers still prove infeasibility for every point whose unbounded variables are bounded by
        # Z_min = phi_b / ||c_unb||_1 (phi_b: the Farkas margin with the bounded variables' box terms only):
        # lam^T (M x - beta) <= 0 fails whenever ||x_unb||_inf < Z_min. Reported as evidence, verdict stays U.
        lo_b, hi_b = np.where(unb, 0.0, lo), np.where(unb, 0.0, hi)
        cb = np.where(unb, 0.0, c)
        phi_b = float(np.sum(np.minimum(cb * lo_b, cb * hi_b))) - float(lam @ bh)
        B = np.maximum(np.abs(lo_b), np.abs(hi_b))
        err = tol.kappa * EPS * (float(np.abs(cb) @ B) + float(lam @ np.abs(bh)) + float(B @ (np.abs(Mh).T @ lam)) + 1.0)
        cu = float(np.abs(c[unb]).sum()) * (1 + tol.kappa * EPS * Mh.shape[0])
        out.certificate = {"type": "farkas_bounded_unbounded_vars", "phi_bounded": phi_b, "err": err,
                           "c_unbounded_l1": cu, "c_unbounded_max": float(np.max(np.abs(c[unb]))),
                           "unbounded_inf_norm_lower_bound": (phi_b - err) / cu if phi_b > err else 0.0,
                           "support": [klab[k] for k in np.flatnonzero(lam > 1e-12)]}
        if phi_b > err:
            out.note += (f"; every feasible point needs ||unbounded vars||_inf >= "
                         f"{(phi_b - err) / cu:.3e} (same multipliers)")
        return out
    lo_b, hi_b = np.where(unb, 0.0, lo), np.where(unb, 0.0, hi)
    box_term = float(np.sum(np.minimum(c * lo_b, c * hi_b)))
    phi = box_term - float(lam @ bh)
    B = np.maximum(np.abs(lo_b), np.abs(hi_b))
    err = tol.kappa * EPS * (float(np.abs(c) @ B) + float(lam @ np.abs(bh)) + float(B @ (np.abs(Mh).T @ lam)) + 1.0)
    supp = [klab[k] for k in np.flatnonzero(lam > 1e-12)]
    cert = {"type": "farkas", "support": supp, "lam": {klab[k]: float(lam[k]) for k in np.flatnonzero(lam > 1e-12)},
            "phi": phi, "err": err, "box_term": box_term, "null_residual_inf": float(np.max(np.abs(c))),
            "lam_min": float(lam.min())}
    out.certificate = cert
    if phi > tol.tau_c + err and not zero_amb:
        out.verdict = "I"
        out.note = "re-verified Farkas certificate"
    elif phi > tol.tau_c + err:
        out.verdict = "I"
        out.note = f"re-verified Farkas certificate (zero rows within tolerance as well: {zero_amb})"
    else:
        out.note = f"LP margin {s:.3e} > 0 but phi = {phi:.3e} <= tau_c + err = {tol.tau_c + err:.1e}"
    return out


# ------------------------------------------------------------------------------------------- problems
def _w_cols(rows: AugmentedRows) -> list[int]:
    return list(range(rows.nw))


def level1_problem(rows: AugmentedRows):
    """Original centralized rows in u (2N): summed pairs (h > 0, divided by h), obstacles, global conn, global CLF."""
    W = _w_cols(rows)
    M, beta, lab = [], [], []
    idx = {l: k for k, l in enumerate(rows.labels)}
    for i in range(rows.N):
        for j in range(i + 1, rows.N):
            h = rows.H[(i, j)]
            if h > 0:
                r = rows.A[idx[f"pair({i},{j})"], W] + rows.A[idx[f"pair({j},{i})"], W]
                b = rows.b[idx[f"pair({i},{j})"]] + rows.b[idx[f"pair({j},{i})"]]
                M.append(r / h); beta.append(-b / h); lab.append(f"pair{{{i},{j}}}")
    for k, l in enumerate(rows.labels):
        if l.startswith("obs("):
            M.append(rows.A[k, W]); beta.append(-rows.b[k]); lab.append(l)
    for kind in ("conn", "clf"):
        ks = [idx[f"{kind}({i})"] for i in range(rows.N)]
        M.append(rows.A[ks][:, W].sum(0)); beta.append(-rows.b[ks].sum()); lab.append(f"{kind}(global)")
    return np.array(M), np.array(beta), lab


def _components(n: int, weight) -> list[list[int]]:
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i); comp.append(i)
            stack += [j for j in range(n) if j != i and j not in seen and weight(i, j) > 0]
        comps.append(sorted(comp))
    return comps


def graphs(rows: AugmentedRows) -> dict[str, list[list[int]]]:
    idx = {l: k for k, l in enumerate(rows.labels)}
    vw = lambda i, j: -rows.A[idx[f"clf({i})"], rows.izl[j]]          # v_ij from the CLF row of agent i
    return {"h": _components(rows.N, lambda i, j: rows.H[(i, j)]), "v": _components(rows.N, vw)}


def level2_componentwise_problem(rows: AugmentedRows):
    """R6.1 Prop 2: u only; summed pairs, obstacles, conn per h-component, CLF per v-component."""
    M, beta, lab = level1_problem(rows)
    keep = [k for k, l in enumerate(lab) if not l.endswith("(global)")]
    M, beta, lab = [M[k] for k in keep], [beta[k] for k in keep], [lab[k] for k in keep]
    W = _w_cols(rows)
    idx = {l: k for k, l in enumerate(rows.labels)}
    g = graphs(rows)
    for kind, comps in (("conn", g["h"]), ("clf", g["v"])):
        for C in comps:
            ks = [idx[f"{kind}({i})"] for i in C]
            r, b = rows.A[ks][:, W].sum(0), rows.b[ks].sum()
            if not r.any() and b == 0.0:
                continue                                          # identically zero (gated inside agent)
            M.append(r); beta.append(-b); lab.append(f"{kind}{{{','.join(map(str, C))}}}")
    return np.array(M), np.array(beta), lab, g


def lift_allocation(rows: AugmentedRows, u: np.ndarray, comps: dict) -> np.ndarray:
    """Explicit allocation z for a componentwise-feasible u (construction in the proof of R6.1 Prop 2)."""
    idx = {l: k for k, l in enumerate(rows.labels)}
    W = _w_cols(rows)
    z = np.zeros(rows.n - rows.nw)
    off = rows.nw
    for i in range(rows.N):
        for j in range(i + 1, rows.N):
            h = rows.H[(i, j)]
            if h > 0:
                ai = (rows.A[idx[f"pair({i},{j})"], W] @ u + rows.b[idx[f"pair({i},{j})"]]) / h
                aj = (rows.A[idx[f"pair({j},{i})"], W] @ u + rows.b[idx[f"pair({j},{i})"]]) / h
                d = (aj - ai) / 2
                z[rows.izp[(i, j)] - off], z[rows.izp[(j, i)] - off] = d / 2, -d / 2
    for kind, key, izmap in (("conn", "h", rows.izc), ("clf", "v", rows.izl)):
        for C in comps[key]:
            if len(C) < 2:
                continue
            ks = [idx[f"{kind}({i})"] for i in C]
            cvals = rows.A[ks][:, W] @ u + rows.b[ks]
            q = cvals.mean() - cvals
            L = rows.A[np.ix_(ks, [izmap[i] for i in C])]
            zc = np.linalg.lstsq(L, q, rcond=None)[0]
            for i, val in zip(C, zc):
                z[izmap[i] - off] = val
    return z


def level2_augmented_problem(rows: AugmentedRows):
    """Augmented rows over (w, z), z free; actuator rows as bounds; degenerate rows dropped."""
    keep = [k for k, l in enumerate(rows.labels) if not rows.deg[k] and not l.startswith("act(")]
    M, beta = rows.A[keep], -rows.b[keep]
    u = rows.cfg.controller.u_max
    lo = np.r_[-u * np.ones(rows.nw), -np.inf * np.ones(rows.n - rows.nw)]
    hi = np.r_[u * np.ones(rows.nw), np.inf * np.ones(rows.n - rows.nw)]
    return M, beta, lo, hi, [rows.labels[k] for k in keep]


def zero_row_terms_for_agent(rows: AugmentedRows, i: int, z: np.ndarray) -> float:
    """Magnitude of the terms forming the right-hand side of agent i's zero-control rows (cancellation floor)."""
    best = 0.0
    for k in range(len(rows.b)):
        if rows.own[k] == i and not rows.A[k, rows.iw[i]].any():
            best = max(best, abs(rows.b[k]) + float(np.abs(rows.A[k, rows.nw:]) @ np.abs(z)))
    return best


# ------------------------------------------------------------------------------------------- driver
@dataclass
class StaticReport:
    level1: Verdict
    level2: Verdict                      # official: componentwise (Prop 2)
    level2_augmented: Verdict            # cross-check: augmented LP, z free (model rows)
    level2_repository: dict              # cross-check: diagnostics.feasibility.centralized_feasibility
    level2_lifted: dict                  # explicit allocation from a componentwise witness, re-verified
    level3: list[Verdict] | None
    level3_verdict: str | None
    category: str
    consistency: list[str]
    components: dict
    timings: dict

    def as_dict(self) -> dict[str, Any]:
        d = {k: (v.as_dict() if isinstance(v, Verdict) else v) for k, v in self.__dict__.items() if k != "level3"}
        d["level3"] = [v.as_dict() for v in self.level3] if self.level3 is not None else None
        return d


def _level3(rows: AugmentedRows, z: np.ndarray, tol: StaticTolerances) -> list[Verdict]:
    p = np.zeros(rows.n)
    p[rows.nw:] = z
    fs = rows.from_vec(p, np.zeros(len(rows.b)))
    u = rows.cfg.controller.u_max
    out = []
    for i in range(rows.N):
        qp = rows.local_qp(i, fs)
        A, b, lab = np.asarray(qp.A_raw, float), np.asarray(qp.b_raw, float), list(qp.row_labels)
        keep = [k for k, l in enumerate(lab) if row_group(l) not in ACTUATOR_GROUPS]
        terms = np.full(len(keep), zero_row_terms_for_agent(rows, i, z))
        v = classify_polytope(A[keep], b[keep], -u * np.ones(2), u * np.ones(2), [lab[k] for k in keep], tol,
                              zero_row_terms=terms)
        rep = local_feasibility(A, b, lab, u, Tolerances())
        v.note += f" | repository local_feasibility s* = {rep['s_star']:.6e}"
        out.append(v)
    return out


def _combine(vs: list[str]) -> str:
    if "I" in vs:
        return "I"
    return "F" if all(v == "F" for v in vs) else "U"


def static_check(model: CertModel, positions, z: np.ndarray | None = None, tol: StaticTolerances | None = None,
                 repository_check: bool = True) -> StaticReport:
    """Levels 1-3 at the physical state ``positions``; Level 3 only if the mismatch ``z`` is given."""
    tol = tol or StaticTolerances()
    t = {}
    t0 = time.perf_counter()
    rows = model.rows(positions)
    t["rows"] = time.perf_counter() - t0
    u = model.cfg.controller.u_max
    nb = rows.nw
    t0 = time.perf_counter()
    M1, b1, l1 = level1_problem(rows)
    L1 = classify_polytope(M1, b1, -u * np.ones(nb), u * np.ones(nb), l1, tol)
    t["level1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    M2, b2, l2, comps = level2_componentwise_problem(rows)
    L2 = classify_polytope(M2, b2, -u * np.ones(nb), u * np.ones(nb), l2, tol)
    lifted: dict[str, Any] = {}
    if L2.verdict == "F":
        res = linprog(np.r_[np.zeros(nb), 1.0], A_ub=np.hstack([M2 / np.linalg.norm(M2, axis=1)[:, None],
                                                                 -np.ones((len(b2), 1))]),
                      b_ub=b2 / np.linalg.norm(M2, axis=1), bounds=[(-u, u)] * nb + [(tol.s_lower, None)],
                      method="highs")
        uu = np.clip(res.x[:-1], -u, u)
        zz = lift_allocation(rows, uu, comps)
        Ma, ba, lo, hi, la = level2_augmented_problem(rows)
        x = np.r_[uu, zz]
        nr = np.linalg.norm(Ma, axis=1)
        nrs = np.where(nr > 0, nr, 1.0)
        resid = (Ma @ x - ba) / nrs
        # rounding error of the residual evaluation (|fl(M x - beta) - (M x - beta)| <= gamma (|M||x| + |beta|))
        rerr = 2.0 * (Ma.shape[1] + 2) * EPS * (np.abs(Ma) @ np.abs(x) + np.abs(ba)) / nrs
        lifted = {"verified": bool(np.max(resid + rerr) <= -tol.tau_w and np.all(np.isfinite(zz))),
                  "max_normalized_residual": float(resid.max()), "max_residual_error_bound": float(rerr.max()),
                  "z_inf_norm": float(np.max(np.abs(zz))) if zz.size else 0.0,
                  "u": uu.tolist(), "z": zz.tolist()}
    t["level2"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ma, ba, lo, hi, la = level2_augmented_problem(rows)
    L2a = classify_polytope(Ma, ba, lo, hi, la, tol, normalization="rows / ||(w, z) part||_2")
    t["level2_augmented"] = time.perf_counter() - t0
    repo = {}
    if repository_check:
        t0 = time.perf_counter()
        repo = _repository_level2(rows)
        t["level2_repository"] = time.perf_counter() - t0
    L3 = None
    l3v = None
    if z is not None:
        t0 = time.perf_counter()
        L3 = _level3(rows, np.asarray(z, float), tol)
        l3v = _combine([v.verdict for v in L3])
        t["level3"] = time.perf_counter() - t0
    # category from confirmed verdicts (study plan v2 §3.2)
    if L1.verdict == "I":
        cat = "A"
    elif L1.verdict == "F" and L2.verdict == "I":
        cat = "B"
    elif L2.verdict == "F" and l3v == "I":
        cat = "C"
    elif l3v == "F":
        cat = "feasible"
    elif z is None and L2.verdict == "F":
        cat = "L2-feasible (Level 3 not evaluated)"
    else:
        cat = "U"
    cons = []
    if l3v == "F" and L2.verdict == "I":
        cons.append("INCONSISTENT: Level 3 F but Level 2 I")
    if L2.verdict == "F" and L1.verdict == "I":
        cons.append("INCONSISTENT: Level 2 F but Level 1 I")
    if l3v == "F" and L1.verdict == "I":
        cons.append("INCONSISTENT: Level 3 F but Level 1 I")
    if {L2.verdict, L2a.verdict} == {"F", "I"}:
        cons.append(f"DISAGREEMENT: componentwise {L2.verdict} vs augmented {L2a.verdict}")
    if repo and L2.verdict in ("F", "I") and repo.get("verdict") in ("feasible", "infeasible") and \
            (repo["verdict"] == "feasible") != (L2.verdict == "F"):
        cons.append(f"DISAGREEMENT: componentwise {L2.verdict} vs repository centralized_feasibility {repo['verdict']}")
    if L2.verdict == "F" and lifted and not lifted["verified"]:
        cons.append("NOTE: lifted allocation not numerically verifiable (ill-conditioned coupling weights)")
    return StaticReport(L1, L2, L2a, repo, lifted, L3, l3v, cat, cons, comps, t)


def _repository_level2(rows: AugmentedRows) -> dict:
    from controller.centralized_qp import build_centralized_qp_data
    cfg = rows.cfg
    c, f = cfg.controller, cfg.fast_dynamics
    fs = rows.from_vec(np.zeros(rows.n), np.zeros(len(rows.b)))
    cqp = _quiet(build_centralized_qp_data, all_agent_states=rows.agents, all_fast_states=fs,
                 obstacles=rows.model.env.obstacles, u_max=c.u_max, xi=f.xi, gamma_1=c.cbf_rate, d_0=c.safety_distance,
                 gamma_2=c.cbf_rate, eta=f.eta, d_c=f.d_c, epsilon_c=f.epsilon_c, gamma_3=c.clf_rate,
                 target_center=tuple(cfg.target.center), sigma=cfg.target.Sigma, epsilon=cfg.target.epsilon,
                 enable_connectivity_constraint=c.enable_connectivity_constraint, alpha_clf=c.alpha_clf,
                 alpha_connectivity=c.alpha_connectivity, lambda_2=rows.lam2, v2=rows.v2, clf_coupling=c.clf_coupling)
    tol = Tolerances()
    r = centralized_feasibility(np.asarray(cqp.A), np.asarray(cqp.b), cqp.row_labels, cqp.row_blocks,
                                cqp.control_dim, len(cqp.q), c.u_max, tol)
    from diagnostics.feasibility import lp_verdict
    return {"s_star": r["s_star"], "verdict": lp_verdict(r["s_star"], tol), "binding": r["binding"][:8]}

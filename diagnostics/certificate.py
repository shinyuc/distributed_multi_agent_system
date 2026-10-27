"""Finite-horizon Method B feasibility certificate (Research Note R6.1, Thm 3) for recorded physical sequences.

The certifier never runs the closed loop. Given a configuration, the physical states x_0..x_{K-1} at which the
local QPs are solved, and the fast state chi_0 at QP time 0, it evaluates at every x_k:

* the rows G(p; x) = A(x) p + b(x) exactly as the fast dynamics evaluate them (unscaled, h-weighted pairwise
  rows), see :class:`AugmentedRows`; their identity with ``update_fast_state`` and with the local QP rows can be
  checked numerically (:meth:`AugmentedRows.check_fast_dynamics`, :meth:`AugmentedRows.check_local_qp`);
* the regularized augmented KKT point (CLARABEL, then an exact equality-KKT polish on the identified active set,
  then sign verification), the partition (active / inactive / degenerate), LICQ and strict complementarity;
* a verified fixed-control lower bound on the feasible-allocation margin rho (M2);
* the Method B lemma constants C0, C1, C2 and the contraction rate, using the unscaled rows;
* the moving-target Lyapunov recursion (with and without contraction) and the first obstruction.

Degenerate rows (A_r = 0, b_r = 0) keep their dual frozen by the dynamics; the KKT selection uses
y*_D = y_D(0) throughout (R6.1 Thm 3, (D1)). Inactive duals of chi_0 must be exactly zero (D2).
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from controller.fast_dynamics import clf_coupling_gates, compute_beta_i, compute_V_i, h_ij, update_fast_state
from controller.local_qp import build_local_qp_data
from diagnostics.feasibility import Tolerances, local_feasibility, lp_verdict
from models.fast_state import EdgeDualState, EdgeMismatchState, FastState

# Outcome codes.
CERT_OK = "CERT_OK"            # every QP state 0..K-1 certified
CERT_PARTIAL = "CERT_PARTIAL"  # certified for k < k_fail, k_fail >= 1
CERT_FAIL = "CERT_FAIL"        # a sufficient condition fails at k = 0
CERT_NA = "CERT_NA"            # a prerequisite is missing (no KKT point, LICQ, nonzero inactive duals, ...)
CERT_AMBIG = "CERT_AMBIG"      # numerically ambiguous (weak activity, unverified margin, gap below tolerance)

# Reason codes, grouped by kind.
REASON_KIND = {
    "RHO": "sufficient_condition", "C1": "sufficient_condition", "C2": "sufficient_condition",
    "C0": "sufficient_condition",
    "L2_INFEASIBLE": "infeasibility",
    "PART": "prerequisite", "LICQ": "prerequisite", "DUAL0": "prerequisite", "Y0_NEG": "prerequisite",
    "MODEL": "prerequisite", "UNSUPPORTED": "prerequisite", "SOLVER": "prerequisite",
    "SC_AMBIG": "ambiguity", "WEAK_INACTIVE": "ambiguity", "M2_UNVERIFIED": "ambiguity",
    "GAP_AMBIG": "ambiguity", "KKT_RESIDUAL": "ambiguity",
}


@dataclass(frozen=True)
class CertTolerances:
    act_rel: float = 1e-9        # |g_r| <= act_rel * max(|A_r|, |b_r|): row taken as active (CLARABEL point)
    weak_rel: float = 1e-7       # inactive slack (normalized) below this: weakly inactive -> ambiguous
    y_sc_rel: float = 1e-9       # active multiplier below y_sc_rel * max(1, max y*): weakly active -> ambiguous
    licq_rel: float = 1e-10      # sigma_min / sigma_max of A_A below this: LICQ failure
    kkt_res: float = 1e-10       # admissible KKT residual (relative to the row scale)
    m2_hard_tighten: float = 1e-9
    m2_safety: float = 1e-12
    m2_hard_thn: float = 1e-14   # mismatch-coefficient norm below this: row is z-independent (hard) in M2
    m2_t_max: float = 1e3
    smin_safety: float = 1e-12
    gap_ambig: float = 1e-9      # 0 <= E_adm - E below this: ambiguous
    model_tol: float = 1e-9      # admissible relative mismatch of the model vs the implementation
    norm: str = "frobenius"      # "frobenius" (default) or "spectral" (as in the Experiment A script)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ----------------------------------------------------------------------------------------------- row model
class CertModel:
    """Holds the environment/simulator built once from a configuration; produces rows at physical states."""

    def __init__(self, cfg, build_fn=None):
        if build_fn is None:
            from diagnostics.run_feasibility import build as build_fn
        self.cfg = cfg
        self.env, self.controller, self.sim = _quiet(build_fn, cfg)
        c, f = cfg.controller, cfg.fast_dynamics
        self.unsupported = []
        if abs(c.alpha_clf - 1.0) > 0 or abs(c.alpha_connectivity - 1.0) > 0:
            self.unsupported.append("alpha_clf and alpha_connectivity must be 1 (R6.1 §1.6)")
        if not c.enable_connectivity_constraint:
            self.unsupported.append("enable_connectivity_constraint must be True")
        if getattr(c, "debug_disable_clf_coupling", False):
            self.unsupported.append("debug_disable_clf_coupling must be False")
        self.r = cfg.simulation.dt / (f.substeps * f.tau)
        self.M = int(f.substeps)

    def rows(self, positions) -> "AugmentedRows":
        return AugmentedRows(self, positions)


class AugmentedRows:
    """Rows of the augmented problem at a physical state, exactly as the fast dynamics evaluate them.

    Variables p = (w (2N), z_pair (ordered pairs), z_c (N), z_clf (N)). Row order per agent i:
    pair(i,j) for j != i, obs(i,k), conn(i), clf(i), act(i,0..3) = [w0-u, w1-u, -w0-u, -w1-u].
    """

    def __init__(self, model: CertModel, positions):
        cfg, env, sim = model.cfg, model.env, model.sim
        c, f = cfg.controller, cfg.fast_dynamics
        self.model, self.cfg = model, cfg
        for a, p in zip(env.agents, positions):
            a.position = (float(p[0]), float(p[1]))
        self.agents = sim._agent_state_map()
        ids = sorted(self.agents)
        if ids != list(range(len(ids))):
            raise ValueError("agent ids must be 0..N-1")
        self.N = N = len(ids)
        _, _, self.lam2, self.v2 = sim._compute_connectivity_runtime_spectrum(self.agents)
        X = {i: np.asarray(self.agents[i].position, float) for i in range(N)}
        self.X = X
        self.iw = {i: [2 * i, 2 * i + 1] for i in range(N)}
        idx = 2 * N
        self.izp = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    self.izp[(i, j)] = idx
                    idx += 1
        self.izc = {i: idx + i for i in range(N)}
        idx += N
        self.izl = {i: idx + i for i in range(N)}
        idx += N
        self.n, self.nw = idx, 2 * N
        Sig, cen, eps = np.asarray(cfg.target.Sigma, float), np.asarray(cfg.target.center, float), cfg.target.epsilon
        q = {i: compute_V_i(tuple(X[i]), tuple(cen), cfg.target.Sigma, eps) for i in range(N)}
        sig = {i: math.exp(-1.0 / q[i]) if q[i] > 0 else 0.0 for i in range(N)}
        self.q, self.sig = q, sig
        H = {(i, j): h_ij(tuple(X[i]), tuple(X[j]), f.d_c) for i in range(N) for j in range(N) if i != j}
        self.H = H
        rows, b, lab, own = [], [], [], []
        for i in range(N):
            for j in range(N):
                if j == i:
                    continue
                r = np.zeros(self.n)
                h, d = H[(i, j)], X[i] - X[j]
                r[self.iw[i]] = -2.0 * h * d
                r[self.izp[(i, j)]] += h
                r[self.izp[(j, i)]] -= h
                rows.append(r); b.append(-h * 0.5 * c.cbf_rate * (d @ d - c.safety_distance ** 2))
                lab.append(f"pair({i},{j})"); own.append(i)
            for k, o in enumerate(env.obstacles):
                r = np.zeros(self.n)
                d = X[i] - np.asarray(o.center, float)
                r[self.iw[i]] = -2.0 * d
                rows.append(r); b.append(-c.cbf_rate * (d @ d - o.radius ** 2))
                lab.append(f"obs({i},{k})"); own.append(i)
            r = np.zeros(self.n)
            r[self.iw[i]] = -np.asarray(_quiet(compute_beta_i, i, None, {}, self.agents, f.d_c, f.epsilon_c, self.v2))
            for j in range(N):
                if j != i:
                    r[self.izc[i]] += H[(i, j)]
                    r[self.izc[j]] -= H[(i, j)]
            rows.append(r); b.append(-self.lam2 / N + f.eta / N)
            lab.append(f"conn({i})"); own.append(i)
            r = np.zeros(self.n)
            const = 0.0
            if not (c.clf_coupling == "gated" and sig[i] == 0.0):
                if q[i] > 0:
                    r[self.iw[i]] = 2.0 * sig[i] * (1.0 + 1.0 / q[i]) * (Sig @ (X[i] - cen))
                    const = c.clf_rate * q[i] * sig[i]
                for j in range(N):
                    if j != i:
                        gi, gj = clf_coupling_gates(sig[i], sig[j], c.clf_coupling)
                        v = H[(i, j)] * gi * gj
                        r[self.izl[i]] += v
                        r[self.izl[j]] -= v
            rows.append(r); b.append(const)
            lab.append(f"clf({i})"); own.append(i)
            for comp, sgn in ((0, 1.0), (1, 1.0), (0, -1.0), (1, -1.0)):
                r = np.zeros(self.n)
                r[self.iw[i][comp]] = sgn
                rows.append(r); b.append(-c.u_max)
                lab.append(f"act({i},{sum(1 for o_, l_ in zip(own, lab) if o_ == i and l_.startswith('act('))})")
                own.append(i)
        self.A, self.b, self.labels, self.own = np.array(rows), np.array(b), lab, np.array(own)
        self.P = np.concatenate([np.ones(self.nw), f.xi * np.ones(self.n - self.nw)])
        self.deg = np.array([not self.A[k].any() and self.b[k] == 0.0 for k in range(len(self.b))])
        self.row_norm = np.linalg.norm(self.A, axis=1)

    # ---- FastState <-> vectors (self components; neighbour components are not used by the dynamics model)
    def to_vec(self, fs: dict[int, FastState]) -> tuple[np.ndarray, np.ndarray]:
        p = np.zeros(self.n)
        y = np.zeros(len(self.b))
        for i in range(self.N):
            p[self.iw[i]] = fs[i].w
            for j in range(self.N):
                if j != i:
                    p[self.izp[(i, j)]] = fs[i].z_pair[j].self_component
            p[self.izc[i]] = fs[i].z_c
            p[self.izl[i]] = fs[i].z_clf
        for k, l in enumerate(self.labels):
            kind, args = l[:-1].split("(")
            a = tuple(int(t) for t in args.split(","))
            s = fs[a[0]]
            y[k] = {"pair": lambda: s.y_pair[a[1]].self_component, "obs": lambda: s.y_obs[a[1]],
                    "conn": lambda: s.y_c, "clf": lambda: s.y_clf, "act": lambda: s.y_u[a[1]]}[kind]()
        return p, y

    def from_vec(self, p: np.ndarray, y: np.ndarray) -> dict[int, FastState]:
        N, K = self.N, len(self.model.env.obstacles)
        fs = {i: FastState(w=(0.0, 0.0), z_pair={j: EdgeMismatchState(0.0, 0.0) for j in range(N) if j != i},
                           z_c=0.0, z_clf=0.0, y_pair={j: EdgeDualState(0.0, 0.0) for j in range(N) if j != i},
                           y_c=0.0, y_clf=0.0, y_obs=[0.0] * K, y_u=[0.0] * 4) for i in range(N)}
        for i in range(N):
            fs[i].w = (float(p[self.iw[i][0]]), float(p[self.iw[i][1]]))
            for j in range(N):
                if j != i:
                    fs[i].z_pair[j].self_component = float(p[self.izp[(i, j)]])
            fs[i].z_c, fs[i].z_clf = float(p[self.izc[i]]), float(p[self.izl[i]])
        for k, l in enumerate(self.labels):
            kind, args = l[:-1].split("(")
            a = tuple(int(t) for t in args.split(","))
            v = float(y[k])
            if kind == "pair":
                fs[a[0]].y_pair[a[1]].self_component = v
            elif kind == "obs":
                fs[a[0]].y_obs[a[1]] = v
            elif kind == "conn":
                fs[a[0]].y_c = v
            elif kind == "clf":
                fs[a[0]].y_clf = v
            else:
                fs[a[0]].y_u[a[1]] = v
        return fs

    # ---- identity checks against the implementation
    def _fast_kwargs(self, i, fs, dt):
        sim, c = self.model.sim, self.cfg.controller
        return dict(i=i, state=fs[i], step_idx=-1, all_fast_states=fs, all_agent_states=self.agents, dt=dt,
                    tau=sim.fast_tau, xi=sim.fast_xi, d_c=sim.fast_d_c, target_center=sim.fast_target_center,
                    sigma=sim.fast_sigma, epsilon=sim.fast_epsilon, gamma_1=sim.fast_gamma_1, gamma_2=sim.fast_gamma_1,
                    gamma_3=c.clf_rate, d_0=sim.fast_d_0, alpha_clf=c.alpha_clf, clf_coupling=c.clf_coupling,
                    x_i=self.agents[i].position, u_i=(0.0, 0.0), obstacles=sim.env.obstacles, u_max=c.u_max,
                    eta=sim.fast_eta, epsilon_c=sim.fast_epsilon_c, lambda_2=self.lam2, v2=self.v2, n_agents=self.N)

    def check_fast_dynamics(self, seed: int = 0) -> tuple[float, float]:
        """Relative max difference between one tiny Euler step of update_fast_state and the model vector field."""
        rng = np.random.default_rng(seed)
        p = rng.normal(scale=0.1, size=self.n)
        y = rng.uniform(0.5, 1.5, size=len(self.b))
        fs = self.from_vec(p, y)
        dt = 1e-7
        rr = dt / self.model.sim.fast_tau
        new = {i: _quiet(update_fast_state, **self._fast_kwargs(i, fs, dt)) for i in range(self.N)}
        p1, y1 = self.to_vec(new)
        model_p = -self.P * p - self.A.T @ y
        model_g = self.A @ p + self.b
        dp = np.max(np.abs((p1 - p) / rr - model_p)) / (1 + np.max(np.abs(model_p)))
        dg = np.max(np.abs((y1 - y) / rr - model_g)) / (1 + np.max(np.abs(model_g)))
        return float(dp), float(dg)

    def local_qp(self, i: int, fs: dict[int, FastState]):
        c, f = self.cfg.controller, self.cfg.fast_dynamics
        return _quiet(build_local_qp_data, i=i, state=fs[i], all_fast_states=fs, all_agent_states=self.agents,
                      obstacles=self.model.env.obstacles, u_max=c.u_max, gamma_1=c.cbf_rate, d_0=c.safety_distance,
                      gamma_2=c.cbf_rate, eta=f.eta, d_c=f.d_c, epsilon_c=f.epsilon_c, gamma_3=c.clf_rate,
                      target_center=tuple(self.cfg.target.center), sigma=self.cfg.target.Sigma,
                      epsilon=self.cfg.target.epsilon, enable_connectivity_constraint=c.enable_connectivity_constraint,
                      alpha_clf=c.alpha_clf, alpha_connectivity=c.alpha_connectivity, lambda_2=self.lam2, v2=self.v2,
                      n_agents=self.N, clf_coupling=c.clf_coupling)

    def check_local_qp(self, seed: int = 0) -> float:
        """Every nonzero model row of agent i equals a repository local-QP row (a^T u <= b) up to positive
        scaling, and vice versa; returns the largest normalized difference."""
        rng = np.random.default_rng(seed)
        p = rng.normal(scale=0.05, size=self.n)
        fs = self.from_vec(p, np.zeros(len(self.b)))
        worst = 0.0
        for i in range(self.N):
            qp = self.local_qp(i, fs)
            Q = np.column_stack([np.asarray(qp.A_raw, float), np.asarray(qp.b_raw, float)])
            Qn = np.linalg.norm(Q, axis=1)
            Q = Q[Qn > 0] / Qn[Qn > 0, None]
            R = [k for k in range(len(self.b)) if self.own[k] == i]
            M_ = np.array([np.r_[self.A[k, self.iw[i]], -self.b[k] - self.A[k, self.nw:] @ p[self.nw:]] for k in R])
            Mn = np.linalg.norm(M_, axis=1)
            M_ = M_[Mn > 0] / Mn[Mn > 0, None]
            for row in M_:
                worst = max(worst, float(np.min(np.linalg.norm(Q - row, axis=1))))
            for row in Q:
                worst = max(worst, float(np.min(np.linalg.norm(M_ - row, axis=1))))
        return worst

    def local_margins(self, z: np.ndarray, tol: Tolerances | None = None) -> list[float]:
        """Independent LP margins s* of every agent's local QP at mismatch z, from the model rows."""
        tol = tol or Tolerances()
        out = []
        for i in range(self.N):
            R = [k for k in range(len(self.b)) if self.own[k] == i]
            Au = self.A[np.ix_(R, self.iw[i])]
            bb = -self.b[R] - self.A[np.ix_(R, range(self.nw, self.n))] @ z
            out.append(local_feasibility(Au, bb, [self.labels[k] for k in R], self.cfg.controller.u_max, tol)["s_star"])
        return out


# ----------------------------------------------------------------------------------------------- KKT
@dataclass
class KKTPoint:
    status: str                          # "ok" or a reason code
    p: np.ndarray | None = None
    y_active: np.ndarray | None = None
    active: np.ndarray | None = None     # boolean mask over rows
    inactive: np.ndarray | None = None
    rank: int = 0
    m: int = 0
    y_min: float = math.inf
    slack_min_norm: float = math.inf
    residual: float = math.nan
    clarabel_diff: float = math.nan
    note: str = ""


def solve_kkt(rows: AugmentedRows, tol: CertTolerances) -> KKTPoint:
    import cvxpy as cp
    nd = ~rows.deg
    A, b = rows.A, rows.b
    scale = np.maximum(rows.row_norm, np.abs(b)) + 1e-300
    p = cp.Variable(rows.n)
    cons = [A[nd] @ p + b[nd] <= 0] if nd.any() else []
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum(cp.multiply(rows.P, cp.square(p)))), cons)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13, max_iter=500)
    except cp.error.SolverError as exc:
        return KKTPoint(status="SOLVER", note=f"CLARABEL failed: {exc}")
    if prob.status in ("infeasible", "infeasible_inaccurate"):
        return KKTPoint(status="L2_INFEASIBLE", note=f"CLARABEL status {prob.status} (solver-reported, not a "
                                                    f"re-verified Farkas certificate)")
    if prob.status != "optimal":
        return KKTPoint(status="SOLVER", note=f"CLARABEL status {prob.status}")
    pc = np.asarray(p.value)
    g = A @ pc + b
    act = nd & (np.abs(g) <= tol.act_rel * scale)
    for _ in range(20):                         # small primal-dual active-set correction loop
        AA = A[act]
        m = int(act.sum())
        if m:
            sv = np.linalg.svd(AA, compute_uv=False)
            rank = int(np.sum(sv > tol.licq_rel * sv[0])) if sv[0] > 0 else 0
            if rank < m:
                return KKTPoint(status="LICQ", active=act, rank=rank, m=m,
                                note=f"numerical rank(A_A) = {rank} < {m} (sigma_min/sigma_max = "
                                     f"{sv[-1] / sv[0]:.1e} <= {tol.licq_rel:.0e}); multipliers not verifiably unique")
        else:
            rank = 0
        Kmat = np.block([[np.diag(rows.P), AA.T], [AA, np.zeros((m, m))]])
        sol = np.linalg.solve(Kmat, np.concatenate([np.zeros(rows.n), -b[act]]))
        pp, ya = sol[:rows.n], sol[rows.n:]
        gg = A @ pp + b
        viol = nd & ~act & (gg > tol.act_rel * scale)
        if m and ya.min() < 0:
            act[np.flatnonzero(act)[int(np.argmin(ya))]] = False
            continue
        if viol.any():
            act[int(np.argmax(np.where(viol, gg / scale, -np.inf)))] = True
            continue
        break
    else:
        return KKTPoint(status="SOLVER", note="active-set correction did not converge")
    inact = nd & ~act
    stat = rows.P * pp + A[act].T @ ya
    res = max(float(np.max(np.abs(stat))) if stat.size else 0.0,
              float(np.max(np.abs(gg[act]) / scale[act])) if m else 0.0)
    slack_norm = -gg[inact] / np.where(rows.row_norm[inact] > 0, rows.row_norm[inact], np.inf)
    out = KKTPoint(status="ok", p=pp, y_active=ya, active=act, inactive=inact, rank=rank, m=m,
                   y_min=float(ya.min()) if m else math.inf,
                   slack_min_norm=float(np.min(np.where(rows.row_norm[inact] > 0, slack_norm, np.inf)))
                   if inact.any() else math.inf,
                   residual=res, clarabel_diff=float(np.max(np.abs(pp - pc))))
    if res > tol.kkt_res * (1.0 + float(np.max(np.abs(b)) if b.size else 0.0)):
        out.status, out.note = "KKT_RESIDUAL", f"KKT residual {res:.2e}"
    elif m and ya.min() <= tol.y_sc_rel * max(1.0, float(ya.max())):
        out.status, out.note = "SC_AMBIG", f"weakly active row: min y*_A = {ya.min():.3e}"
    elif inact.any() and out.slack_min_norm <= tol.weak_rel:
        out.status, out.note = "WEAK_INACTIVE", f"weakly inactive row: min slack/|A_r| = {out.slack_min_norm:.3e}"
    return out


# ----------------------------------------------------------------------------------------------- margins
def m2_margin(rows: AugmentedRows, zstar: np.ndarray, tol: CertTolerances) -> tuple[float, list[float]]:
    """Fixed-control lower bound on rho(z*) (R6.1 Cor 1), with the control re-verified a posteriori.
    Returns (rho_lb, per_agent); -inf marks an agent whose bound could not be verified."""
    per = []
    for i in range(rows.N):
        R = [k for k in range(len(rows.b)) if rows.own[k] == i and not rows.deg[k]]
        Au = rows.A[np.ix_(R, rows.iw[i])]
        Th = rows.A[np.ix_(R, range(rows.nw, rows.n))]
        beta = -rows.b[R] - Th @ zstar
        thn = np.linalg.norm(Th, axis=1)
        hard = thn <= tol.m2_hard_thn
        res = linprog(c=[0, 0, -1], A_ub=np.column_stack([Au, np.where(hard, 0, thn)]),
                      b_ub=beta - tol.m2_hard_tighten * hard,
                      bounds=[(None, None), (None, None), (None, tol.m2_t_max)], method="highs")
        if res.status != 0:
            per.append(-math.inf)
            continue
        u = res.x[:2]
        slack = beta - Au @ u
        if np.any(slack[hard] <= 0):
            per.append(-math.inf)
            continue
        per.append(float(np.min(slack[~hard] / thn[~hard])) - tol.m2_safety if (~hard).any() else math.inf)
    return (min(per) if per else math.inf), per


def m1_margin(rows: AugmentedRows, zstar: np.ndarray) -> tuple[float, list[float]]:
    """Exact Farkas margin (R6.1 Prop 3(d)) by enumeration of extreme rays (supports <= 3); numerical
    estimate (SVD null-space tests). Expensive; for diagnostics only."""
    import itertools
    per = []
    for i in range(rows.N):
        R = [k for k in range(len(rows.b)) if rows.own[k] == i and not rows.deg[k]]
        Au = rows.A[np.ix_(R, rows.iw[i])]
        Th = rows.A[np.ix_(R, range(rows.nw, rows.n))]
        beta = -rows.b[R] - Th @ zstar
        best = math.inf
        for size in (1, 2, 3):
            for S in itertools.combinations(range(len(R)), size):
                Mx = Au[list(S)].T
                _, sv, Vt = np.linalg.svd(Mx)
                null = [Vt[k] for k in range(size) if k >= len(sv) or sv[k] <= 1e-12 * max(1.0, sv[0])]
                if len(null) != 1:
                    continue
                lam = null[0]
                if np.all(lam <= 1e-14):
                    lam = -lam
                if np.any(lam <= 1e-12):
                    continue
                nz = float(np.linalg.norm(Th[list(S)].T @ lam))
                if nz > 1e-12 * max(1.0, float(np.abs(lam).max())):
                    best = min(best, float(lam @ beta[list(S)]) / nz)
        per.append(best)
    return (min(per) if per else math.inf), per


# ----------------------------------------------------------------------------------------------- lemma
@dataclass
class LemmaConstants:
    a: float          # norm bound on A_A used in C0, E, switching
    smin: float       # lower bound on sigma_min(A_A) (m-th singular value); +inf if m = 0
    kappa_p: float
    c: float          # per-substep contraction (0 if kappa_p <= 0)
    c0_ok: bool
    C1: float
    C1_label: str
    C2: float
    C2_label: str


def lemma_constants(rows: AugmentedRows, kkt: KKTPoint, r: float, tol: CertTolerances) -> LemmaConstants:
    AA = rows.A[kkt.active]
    m = AA.shape[0]
    pmin, pmax = float(rows.P.min()), float(rows.P.max())
    if m == 0:
        a, smin = 0.0, math.inf
    elif tol.norm == "spectral":
        sv = np.linalg.svd(AA, compute_uv=False)
        a, smin = float(sv[0]) * (1 + 1e-12), float(sv[-1]) * (1 - 1e-12)
    else:
        a = float(np.sqrt(np.sum(AA ** 2)))
        lmin = float(np.linalg.eigvalsh(AA @ AA.T).min()) - tol.smin_safety * (1 + a * a)
        smin = math.sqrt(max(lmin, 0.0))
    kap = 2 * r * pmin - r ** 2 * (pmax ** 2 + 3 * a ** 2) - 2 * r ** 3 * pmax * a ** 2 - 2 * r ** 4 * a ** 4
    c0 = (r * a < 1.0) and kap >= 0.0
    cc = min(kap, r ** 2 * smin ** 2 / 2) / (1 + r * a) if c0 else 0.0
    gg = rows.A @ kkt.p + rows.b
    inact = np.flatnonzero(kkt.inactive)
    C1, C1l = math.inf, ""
    for k in inact:
        v = (-gg[k] / rows.row_norm[k]) if rows.row_norm[k] > 0 else math.inf
        if v < C1:
            C1, C1l = float(v), rows.labels[k]
    C2, C2l = math.inf, ""
    for y, k in zip(kkt.y_active, np.flatnonzero(kkt.active)):
        v = float(y) / (r * rows.row_norm[k])
        if v < C2:
            C2, C2l = v, rows.labels[k]
    return LemmaConstants(a=a, smin=smin, kappa_p=kap, c=cc, c0_ok=c0, C1=C1, C1_label=C1l, C2=C2, C2_label=C2l)


def _U(ep, ey, AA, r):
    return float(ep @ ep + ey @ ey + 2 * r * ep @ (AA.T @ ey)) if AA.size else float(ep @ ep + ey @ ey)


# ----------------------------------------------------------------------------------------------- certifier
VERSIONS = ("contraction", "non_increase")


@dataclass
class CertificateResult:
    version: str                     # "contraction" or "non_increase" (c_k := 0)
    status: str
    reason: str | None
    reason_kind: str | None
    k_fail: int | None               # first QP state at which a condition fails or cannot be checked
    certified_states: int            # QP states 0..certified_states-1 satisfy every condition
    num_states: int
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Certification:
    results: dict[str, CertificateResult]
    steps: dict[str, list] = field(default_factory=dict)
    initial: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {"results": {v: r.as_dict() for v, r in self.results.items()}, "initial": self.initial}
        s = self.steps
        for key in ("E", "E_ni", "E_adm", "rho", "C1", "C2", "delta", "a", "kappa_p", "c", "y_min", "slack_min"):
            arr = np.asarray(s.get(key, []), float)
            arr = arr[np.isfinite(arr)] if arr.size else arr
            if arr.size:
                d[f"{key}_min"], d[f"{key}_max"] = float(arr.min()), float(arr.max())
        if s.get("delta"):
            d["variation"] = float(np.sum(s["delta"]))
        for key in ("E", "E_ni"):
            if s.get(key) and s.get("E_adm"):
                E, Ea = np.asarray(s[key], float), np.asarray(s["E_adm"], float)[: len(s[key])]
                d[f"Gamma_{key}"] = float(np.max(E / Ea))
                d[f"min_gap_{key}"] = float(np.min(Ea - E))
        return d


def _status_for(kind: str, k: int) -> str:
    if kind == "ambiguity":
        return CERT_AMBIG
    if kind == "sufficient_condition":
        return CERT_FAIL if k == 0 else CERT_PARTIAL
    return CERT_NA if k == 0 else CERT_PARTIAL


def certify(model: CertModel, positions: list, chi0: dict[int, FastState], tol: CertTolerances | None = None,
            check_model_at: tuple[int, ...] = (0,), prop6_check: bool = True, progress=None) -> Certification:
    """R6.1 Thm 3 along the physical sequence ``positions`` (the QP states x_0..x_{K-1}) from the fast state chi0.

    Both recursions are evaluated in one pass: "contraction" (with c_k) and "non_increase" (c_k := 0). A state
    k is certified for a version iff the KKT prerequisites hold, the partition equals that at k = 0, C0 holds,
    E_k < rho_k, and (for k <= K-2, i.e. when substeps at x_k follow) E_k <= min(C1_k, C2_k)."""
    tol = tol or CertTolerances()
    K = len(positions)
    r, M = model.r, model.M
    keys = ("E", "E_ni", "E_adm", "rho", "rho_agents", "C1", "C1_label", "C2", "C2_label", "delta", "a", "smin",
            "kappa_p", "c", "y_min", "slack_min", "kkt_status", "active", "residual", "clarabel_diff", "kappa_sw")
    steps: dict[str, list] = {k: [] for k in keys}
    init: dict[str, Any] = {}
    results: dict[str, CertificateResult] = {}

    def fail_all(code, k, msg):
        kind = REASON_KIND.get(code, "prerequisite")
        for v in VERSIONS:
            if v not in results:
                results[v] = CertificateResult(v, _status_for(kind, k), code, kind, k, k, K, msg)

    def done():
        for v in VERSIONS:
            results.setdefault(v, CertificateResult(v, CERT_OK, None, None, None, K, K, f"all {K} QP states certified"))
        return Certification(results={v: results[v] for v in VERSIONS}, steps=steps, initial=init)

    if model.unsupported:
        fail_all("UNSUPPORTED", 0, "; ".join(model.unsupported))
        return done()
    rows0 = model.rows(positions[0])
    for k_chk in check_model_at:
        if k_chk < K:
            rk = rows0 if k_chk == 0 else model.rows(positions[k_chk])
            dp, dg = rk.check_fast_dynamics(seed=k_chk)
            dq = rk.check_local_qp(seed=k_chk)
            init.setdefault("model_check", []).append({"k": k_chk, "fast_p": dp, "fast_g": dg, "local_qp": dq})
            if max(dp, dg, dq) > tol.model_tol:
                fail_all("MODEL", 0, f"model/implementation mismatch at k={k_chk}: {dp:.1e} {dg:.1e} {dq:.1e}")
                return done()
    p0, y0 = rows0.to_vec(chi0)
    init["y0_min"] = float(y0.min()) if y0.size else 0.0
    if y0.size and y0.min() < 0:
        fail_all("Y0_NEG", 0, "initial duals must be nonnegative")
        return done()
    R = {v: None for v in VERSIONS}
    prev = None
    part0 = None
    for k in range(K):
        rows = rows0 if k == 0 else model.rows(positions[k])
        if k > 0 and tuple(np.flatnonzero(rows.deg)) != part0[1]:
            dgc = [rows.labels[i] for i in sorted(set(np.flatnonzero(rows.deg)) ^ set(part0[1]))]
            fail_all("PART", k, f"degenerate-row set changes at k={k}: {dgc}")
            break
        kkt = solve_kkt(rows, tol)
        steps["kkt_status"].append(kkt.status)
        if kkt.status != "ok":
            fail_all(kkt.status, k, f"k={k}: {kkt.note}")
            break
        part = (tuple(np.flatnonzero(kkt.active)), tuple(np.flatnonzero(rows.deg)))
        if k == 0:
            part0 = part
            if np.any(y0[kkt.inactive] != 0.0):
                bad = [rows.labels[i] for i in np.flatnonzero(kkt.inactive & (y0 != 0.0))]
                fail_all("DUAL0", 0, f"inactive duals of chi_0 not exactly zero: {bad[:6]}")
                break
            init["active"] = [rows.labels[i] for i in part[0]]
            init["degenerate"] = [rows.labels[i] for i in part[1]]
        elif part != part0:
            gained = [rows.labels[i] for i in sorted(set(part[0]) - set(part0[0]))]
            lost = [rows.labels[i] for i in sorted(set(part0[0]) - set(part[0]))]
            dgc = [rows.labels[i] for i in sorted(set(part[1]) ^ set(part0[1]))]
            fail_all("PART", k, f"partition change at k={k}: active +{gained} -{lost}; degenerate changes {dgc}")
            break
        lc = lemma_constants(rows, kkt, r, tol)
        rho, rho_ag = m2_margin(rows, kkt.p[rows.nw:], tol)
        AA = rows.A[kkt.active]
        chi = np.concatenate([kkt.p, kkt.y_active])
        if k == 0:
            ep, ey = p0 - kkt.p, y0[kkt.active] - kkt.y_active
            R0 = math.sqrt(max(_U(ep, ey, AA, r), 0.0))
            R = {v: R0 for v in VERSIONS}
            init.update(e0=float(np.linalg.norm(np.r_[ep, ey])), sqrtU0=R0)
            dlt, ksw = 0.0, 1.0
        else:
            pchi, pc, pAA, pa = prev
            dlt = float(np.linalg.norm(chi - pchi))
            dA = float(np.linalg.norm(AA - pAA, 2) if tol.norm == "spectral" else np.linalg.norm(AA - pAA))
            ksw = 1 + r * dA / (1 - r * pa)
            for v in VERSIONS:
                Rp = R[v] * (1 - pc) ** (M / 2) if v == "contraction" else R[v]
                R[v] = math.sqrt(ksw) * Rp + math.sqrt(1 + r * lc.a) * dlt
        E = {v: (R[v] / math.sqrt(1 - r * lc.a) if lc.c0_ok else math.inf) for v in VERSIONS}
        E_adm = min(lc.C1, lc.C2, rho)
        for key, val in (("E", E["contraction"]), ("E_ni", E["non_increase"]), ("E_adm", E_adm), ("rho", rho),
                         ("rho_agents", rho_ag), ("C1", lc.C1), ("C1_label", lc.C1_label), ("C2", lc.C2),
                         ("C2_label", lc.C2_label), ("delta", dlt), ("a", lc.a), ("smin", lc.smin),
                         ("kappa_p", lc.kappa_p), ("c", lc.c), ("y_min", kkt.y_min), ("slack_min", kkt.slack_min_norm),
                         ("active", [rows.labels[i] for i in part[0]]), ("residual", kkt.residual),
                         ("clarabel_diff", kkt.clarabel_diff), ("kappa_sw", ksw)):
            steps[key].append(val)
        prev = (chi, lc.c, AA, lc.a)
        if progress and k % 500 == 0:
            progress(k, K)
        for v in VERSIONS:
            if v in results:
                continue
            Ev = E[v]
            fail = None
            if not math.isfinite(rho):
                fail = ("M2_UNVERIFIED", f"k={k}: M2 margin not verified")
            elif not lc.c0_ok:
                fail = ("C0", f"k={k}: C0 fails (r a = {r * lc.a:.3e}, kappa_p = {lc.kappa_p:.3e})")
            else:
                cands = [("RHO", rho, Ev >= rho)]
                if k <= K - 2:
                    cands += [("C1", lc.C1, Ev > lc.C1), ("C2", lc.C2, Ev > lc.C2)]
                violated = [(n, val) for n, val, bad in cands if bad]
                if violated:
                    n, val = min(violated, key=lambda t: t[1])
                    fail = (n, f"k={k}: E = {Ev:.6e} vs {n} = {val:.6e}")
                else:
                    gap = min(val for _, val, _ in cands) - Ev
                    if gap < tol.gap_ambig:
                        fail = ("GAP_AMBIG", f"k={k}: E_adm - E = {gap:.2e} below tolerance {tol.gap_ambig:.0e}")
            if fail:
                code, msg = fail
                kind = REASON_KIND[code]
                results[v] = CertificateResult(v, _status_for(kind, k), code, kind, k, k, K, msg)
                if prop6_check and code == "RHO" and "prop6_at_fail" not in init:
                    s_star = rows.local_margins(p0[rows.nw:])
                    init["prop6_at_fail"] = {"k": k, "version": v, "z0_local_margins": s_star,
                                             "z0_feasible": all(lp_verdict(s, Tolerances()) == "feasible"
                                                                for s in s_star)}
        if all(v in results for v in VERSIONS):
            break
    return done()


# ----------------------------------------------------------------------------------------------- helpers
def zero_fast_state(model: CertModel) -> dict[int, FastState]:
    from models.fast_state import init_fast_state_complete_graph
    N, K = len(model.env.agents), len(model.env.obstacles)
    return {i: init_fast_state_complete_graph(i, N, K) for i in range(N)}


def observed_errors(model: CertModel, positions: list, snapshots: list[dict[int, FastState]], num_states: int,
                    tol: CertTolerances | None = None) -> dict[str, list]:
    """Observed full-state errors |chi_k - chi*_k| and the observed Lyapunov radius (reported separately from the
    analytic certificate; requires the snapshots of the fast state at QP times)."""
    tol = tol or CertTolerances()
    out = {"err": [], "radius": [], "inactive_zero": []}
    for k in range(min(num_states, len(snapshots), len(positions))):
        rows = model.rows(positions[k])
        kkt = solve_kkt(rows, tol)
        if kkt.status != "ok":
            break
        p, y = rows.to_vec(snapshots[k])
        ystar = np.zeros(len(rows.b))
        ystar[kkt.active] = kkt.y_active
        ystar[rows.deg] = y[rows.deg]
        out["err"].append(float(np.linalg.norm(np.r_[p - kkt.p, y - ystar])))
        iz = bool(np.all(y[kkt.inactive] == 0.0))
        out["inactive_zero"].append(iz)
        AA = rows.A[kkt.active]
        a = float(np.sqrt(np.sum(AA ** 2))) if tol.norm != "spectral" else float(np.linalg.svd(AA, compute_uv=False)[0])
        U = _U(p - kkt.p, y[kkt.active] - kkt.y_active, AA, model.r)
        out["radius"].append(math.sqrt(max(U, 0)) / math.sqrt(1 - model.r * a) if iz else math.nan)
    return out

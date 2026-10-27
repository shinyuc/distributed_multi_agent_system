"""Construction helpers and physical-validity checks for the 48-scenario study (geometry only; no QP)."""

from __future__ import annotations

import math

import numpy as np

from controller.fast_dynamics import alpha_ij_connectivity, compute_V_i, connectivity_sigma, h_ij

RULE_TOL = 1e-9          # tolerance on the construction-rule inequalities (exact-arithmetic boundaries)


def descent_dir(x, Sigma, center):
    d = -np.asarray(Sigma, float) @ (np.asarray(x, float) - np.asarray(center, float))
    return d / np.linalg.norm(d)


def head_on_obstacle(x, r, g, Sigma, center):
    """Obstacle of radius r centred on agent x's CLF descent ray at surface gap g (study plan v2 §4)."""
    c = np.asarray(x, float) + (r + g) * descent_dir(x, Sigma, center)
    return [float(c[0]), float(c[1])], float(r)


def lattice(cols: int, rows: int, s: float, x0: float):
    return [[x0 - c * s, (r - (rows - 1) / 2) * s] for c in range(cols) for r in range(rows)]


def pair_distances(X):
    X = [np.asarray(p, float) for p in X]
    return [(i, j, float(np.linalg.norm(X[i] - X[j]))) for i in range(len(X)) for j in range(i + 1, len(X))]


def cutoff_margin(X, d_c, eps_c):
    """min over pairs closer than d_c + 0.05 of the distance to the cutoffs d_c and d_c - eps_c."""
    vals = [min(abs(d - d_c), abs(d - (d_c - eps_c))) for _, _, d in pair_distances(X) if d < d_c + 0.05]
    return min(vals) if vals else math.inf


def lambda2(X, d_c, eps_c):
    sig = connectivity_sigma(d_c, eps_c)
    n = len(X)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = alpha_ij_connectivity(tuple(X[i]), tuple(X[j]), d_c, eps_c, sig)
    ev = np.linalg.eigvalsh(np.diag(A.sum(1)) - A)
    return float(ev[1]) if n > 1 else 0.0


def q_values(X, Sigma, center, eps):
    return [float(compute_V_i(tuple(x), tuple(center), Sigma, eps)) for x in X]


def lattice_with_cutoff_rule(cols, rows, s_min, x0, d_c, eps_c, step=0.005, margin=0.02):
    """Smallest spacing s >= s_min on a `step` grid whose pairwise distances all keep `margin` from both cutoffs."""
    s = s_min
    for _ in range(2000):
        X = lattice(cols, rows, round(s, 6), x0)
        if cutoff_margin(X, d_c, eps_c) >= margin - RULE_TOL:
            return X, round(s, 6)
        s += step
    raise ValueError("no admissible lattice spacing")


def physical_validity(X, obstacles, cfg_like: dict) -> dict:
    """Physical-validity metrics and construction-rule checks (study plan v2 §4)."""
    Sigma, center, eps = cfg_like["Sigma"], cfg_like["center"], cfg_like["epsilon"]
    d_c, eps_c, eta, d0 = cfg_like["d_c"], cfg_like["epsilon_c"], cfg_like["eta"], cfg_like["d0"]
    X = [np.asarray(p, float) for p in X]
    D = pair_distances(X)
    minpair = min((d for *_, d in D), default=math.inf)
    clear = min((float(np.linalg.norm(x - np.asarray(o[0]))) - o[1] for x in X for o in obstacles), default=math.inf)
    l2 = lambda2(X, d_c, eps_c)
    q = q_values(X, Sigma, center, eps)
    sig = [math.exp(-1.0 / v) if v > 0 else 0.0 for v in q]
    H = {(i, j): h_ij(tuple(X[i]), tuple(X[j]), d_c) for i, j, _ in D}
    cm = cutoff_margin(X, d_c, eps_c)
    rules = {
        "pair_ge_d0_plus_0.05": minpair >= d0 + 0.05 - RULE_TOL,
        "pair_safe_H_ge_0": minpair >= d0 - RULE_TOL,
        "obstacle_clearance_ge_0.05": clear >= 0.05 - RULE_TOL,
        "lambda2_ge_eta_plus_0.01": l2 >= eta + 0.01 - RULE_TOL,
        "lambda2_ge_eta": l2 >= eta - RULE_TOL,
        "cutoff_margin_ge_0.02": cm >= 0.02 - RULE_TOL,
    }
    return {"min_pair_distance": minpair, "min_obstacle_clearance": clear, "lambda2": l2, "lambda2_minus_eta": l2 - eta,
            "cutoff_margin": cm, "q": q, "sigma": sig, "h_edges": {f"{i}-{j}": v for (i, j), v in H.items() if v > 0},
            "rules": rules}

"""Auxiliary feasibility LPs and constraint-violation checks (diagnostics only).

Feasibility margin LP
---------------------
For constraint rows a_r^T x <= b_r (actuator bounds excluded), each row is
normalized by ||a_r||_2 and the LP

    minimize    s
    subject to  (a_r/||a_r||)^T x - s <= b_r/||a_r||   for every non-actuator row
                -u_max <= u <= u_max                    (hard, never relaxed)

is solved with HiGHS. ``s*`` is a signed margin in control units:

* s* < -feas_tol : strictly feasible; some admissible u satisfies every row with
                   Euclidean clearance |s*| from each half-space boundary.
* s* > +feas_tol : infeasible; no u in the actuator box satisfies all rows, and the
                   best u still misses some half-space by s*.
* |s*| <= feas_tol : borderline.

Rows with a_r = 0 cannot be changed by u; they are checked directly (0 <= b_r).
A violated zero row makes the QP infeasible regardless of u (s* = +inf).

The relaxation variable s exists only in this LP. It is never passed back to the
controller.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linprog

GROUPS: tuple[str, ...] = ("pairwise", "obstacle", "connectivity", "clf")
ACTUATOR_GROUPS = {"actuator", "bounds"}

# Class meanings (codes are stable; new classes are appended):
#   solved             OSQP optimal           and LP verified feasible (s* < -feas_tol)
#   solved_inaccurate  OSQP optimal_inaccurate and LP verified feasible
#   borderline         |s*| <= feas_tol (any OSQP status)
#   numerical_failure  OSQP failed (non-optimal status/exception) but LP says feasible
#   infeasible         OSQP failed and LP confirms infeasible (s* > feas_tol)
#   inconsistent       OSQP reported success but LP says infeasible (disagreement;
#                      the OSQP input is still applied - diagnostics never intervene)
#   lp_error           the auxiliary LP itself failed; feasibility is NOT verified
CLASSES: tuple[str, ...] = (
    "solved",
    "solved_inaccurate",
    "borderline",
    "numerical_failure",
    "infeasible",
    "inconsistent",
    "lp_error",
)
VERIFIED_SOLVED = ("solved", "solved_inaccurate")
CLASS_CODE = {name: idx for idx, name in enumerate(CLASSES)}
SOLVER_SUCCESS = {"optimal", "optimal_inaccurate"}

_S_LOWER = -1e9
_B_CLIP = 1e12


@dataclass(frozen=True)
class Tolerances:
    feas_tol: float = 1e-6
    """Band |s*| <= feas_tol (control units) is classified as borderline."""
    viol_tol: float = 1e-4
    """Normalized violation of the applied u above this is flagged. OSQP (eps 1e-5,
    no polishing) routinely returns active rows violated by ~5e-6, so the flag sits
    an order of magnitude above the solver tolerance."""
    zero_row_norm: float = 1e-150
    """Rows with ||a|| at or below this are treated as u-independent (a = 0)."""
    zero_row_b_tol: float = 1e-9
    """A zero row 0 <= b counts as violated when b < -zero_row_b_tol."""
    binding_tol: float = 1e-9
    """|LP dual| above this marks a row as binding at the LP optimum."""


def row_group(label: str) -> str:
    return label.split("(", 1)[0]


def _normalize(
    A: np.ndarray,
    b: np.ndarray,
    labels: list[str],
    tol: Tolerances,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Return normalized nonzero rows and a list of violated zero rows."""
    norms = np.linalg.norm(A, axis=1) if A.size else np.zeros((0,))
    keep = norms > tol.zero_row_norm
    degenerate: list[dict[str, Any]] = []
    for idx in np.flatnonzero(~keep):
        if b[idx] < -tol.zero_row_b_tol:
            degenerate.append({"label": labels[idx], "b": float(b[idx])})
    A_hat = A[keep] / norms[keep, None]
    b_hat = np.clip(b[keep] / norms[keep], -_B_CLIP, _B_CLIP)
    kept_labels = [labels[idx] for idx in np.flatnonzero(keep)]
    return A_hat, b_hat, kept_labels, degenerate


def margin_lp(
    A_hat: np.ndarray,
    b_hat: np.ndarray,
    labels: list[str],
    var_bounds: list[tuple[float | None, float | None]],
    tol: Tolerances,
) -> dict[str, Any]:
    """Solve min s s.t. A_hat x - s <= b_hat, x within var_bounds."""
    n = len(var_bounds)
    if A_hat.shape[0] == 0:
        return {"s_star": -math.inf, "lp_status": "no_rows", "x": None, "binding": []}
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.hstack([A_hat, -np.ones((A_hat.shape[0], 1))])
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_hat,
        bounds=list(var_bounds) + [(_S_LOWER, None)],
        method="highs",
    )
    if res.status != 0 or res.x is None:
        return {"s_star": math.nan, "lp_status": f"highs_status_{res.status}: {res.message}", "x": None, "binding": []}
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    binding: list[str] = []
    if marg is not None:
        binding = [labels[k] for k in np.flatnonzero(np.abs(marg) > tol.binding_tol)]
    return {
        "s_star": float(res.x[-1]),
        "lp_status": "optimal",
        "x": res.x[:-1].tolist(),
        "binding": binding,
    }


def _select_rows(
    A: np.ndarray,
    b: np.ndarray,
    labels: list[str],
    groups: Iterable[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    wanted = set(groups) if groups is not None else None
    idx = [
        k
        for k, label in enumerate(labels)
        if row_group(label) not in ACTUATOR_GROUPS
        and (wanted is None or row_group(label) in wanted)
    ]
    n_cols = A.shape[1] if A.ndim == 2 else 0
    if not idx:
        return np.zeros((0, n_cols)), np.zeros((0,)), []
    return A[idx], b[idx], [labels[k] for k in idx]


def local_feasibility(
    A_raw: np.ndarray,
    b_raw: np.ndarray,
    labels: list[str],
    u_max: float,
    tol: Tolerances,
    groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Feasibility margin of a local QP (u in R^2) with hard actuator bounds."""
    A_sel, b_sel, lab_sel = _select_rows(A_raw, b_raw, labels, groups)
    A_hat, b_hat, kept, degenerate = _normalize(A_sel, b_sel, lab_sel, tol)
    out = margin_lp(A_hat, b_hat, kept, [(-u_max, u_max)] * 2, tol)
    out["degenerate_violated"] = degenerate
    out["s_lp"] = out["s_star"]
    if degenerate:
        out["s_star"] = math.inf
        out["binding"] = [d["label"] for d in degenerate]
    return out


def lp_verdict(s_star: float, tol: Tolerances) -> str:
    if math.isnan(s_star):
        return "unknown"
    if s_star > tol.feas_tol:
        return "infeasible"
    if s_star < -tol.feas_tol:
        return "feasible"
    return "borderline"


def classify(solver_status: str | None, solver_success: bool, s_star: float, tol: Tolerances) -> str:
    verdict = lp_verdict(s_star, tol)
    if verdict == "unknown":
        return "lp_error"
    if verdict == "borderline":
        return "borderline"
    if solver_success:
        if verdict == "infeasible":
            return "inconsistent"
        return "solved_inaccurate" if solver_status == "optimal_inaccurate" else "solved"
    if verdict == "infeasible":
        return "infeasible"
    return "numerical_failure"


def constraint_violation(
    A_raw: np.ndarray,
    b_raw: np.ndarray,
    A_scaled: np.ndarray,
    b_scaled: np.ndarray,
    labels: list[str],
    u: np.ndarray,
    tol: Tolerances,
) -> dict[str, Any]:
    """Violation of the applied u on the original (unscaled) rows, all rows incl. bounds.

    ``raw``        : max_r (a_r^T u - b_r) in the rows' own units.
    ``normalized`` : max_r (a_r^T u - b_r)/||a_r|| (control units); zero rows use -b_r.
    ``scaled``     : max_r on the rows the solver actually saw.
    """
    r_raw = A_raw @ u - b_raw
    norms = np.linalg.norm(A_raw, axis=1)
    r_norm = np.where(norms > tol.zero_row_norm, r_raw / np.where(norms > 0, norms, 1.0), r_raw)
    r_scaled = A_scaled @ u - b_scaled
    k_raw = int(np.argmax(r_raw))
    k_norm = int(np.argmax(r_norm))
    return {
        "raw": float(r_raw[k_raw]),
        "raw_label": labels[k_raw],
        "normalized": float(r_norm[k_norm]),
        "normalized_label": labels[k_norm],
        "scaled": float(np.max(r_scaled)),
    }


def group_slacks(
    A_raw: np.ndarray,
    b_raw: np.ndarray,
    labels: list[str],
    u: np.ndarray,
    tol: Tolerances,
) -> dict[str, float]:
    """Minimum normalized slack (b_r - a_r^T u)/||a_r|| per group at u (nan if none)."""
    norms = np.linalg.norm(A_raw, axis=1)
    out: dict[str, float] = {}
    for group in GROUPS:
        idx = [
            k for k, label in enumerate(labels)
            if row_group(label) == group and norms[k] > tol.zero_row_norm
        ]
        if not idx:
            out[group] = math.nan
            continue
        sl = (b_raw[idx] - A_raw[idx] @ u) / norms[idx]
        out[group] = float(np.min(sl))
    return out


def group_subset_analysis(
    feasibility_fn,
    present_groups: list[str],
    tol: Tolerances,
) -> dict[str, Any]:
    """Margin for every nonempty subset of constraint groups (actuator bounds always hard).

    Returns all subset margins and the minimal infeasible subsets (infeasible, but
    every proper subset feasible or borderline).
    """
    subsets: dict[str, float] = {}
    infeasible: list[frozenset[str]] = []
    for r in range(1, len(present_groups) + 1):
        for combo in itertools.combinations(present_groups, r):
            s = feasibility_fn(combo)["s_star"]
            subsets["+".join(combo)] = s
            if lp_verdict(s, tol) == "infeasible":
                infeasible.append(frozenset(combo))
    minimal = [
        sorted(sub) for sub in infeasible
        if not any(other < sub for other in infeasible)
    ]
    return {"subset_margins": subsets, "minimal_infeasible_subsets": minimal}


def centralized_feasibility(
    A_scaled: np.ndarray,
    b_scaled: np.ndarray,
    row_labels: list[str],
    row_blocks: list[str],
    control_dim: int,
    n_dec: int,
    u_max: float,
    tol: Tolerances,
    groups: Iterable[str] | None = None,
    z_fixed: np.ndarray | None = None,
) -> dict[str, Any]:
    """Margin LP for the centralized augmented constraints.

    Decision vector = (u_1..u_N, z). Actuator bounds stay hard. With ``z_fixed=None``
    the mismatch variables are free (exists-z check); otherwise they are pinned.
    Positive row scaling in the centralized builder does not change the normalized rows.
    """
    wanted = set(groups) if groups is not None else None
    idx = [
        k for k, block in enumerate(row_blocks)
        if block not in ACTUATOR_GROUPS and (wanted is None or block in wanted)
    ]
    labels = [row_labels[k] for k in idx]
    A_sel = A_scaled[idx] if idx else np.zeros((0, n_dec))
    b_sel = b_scaled[idx] if idx else np.zeros((0,))
    bounds: list[tuple[float | None, float | None]] = [(-u_max, u_max)] * control_dim
    if z_fixed is None:
        bounds += [(None, None)] * (n_dec - control_dim)
    else:
        # Pinned z: move z to the right-hand side and normalize by the u-part only,
        # so the margin equals the worst local-QP margin (consistency check).
        b_sel = b_sel - A_sel[:, control_dim:] @ np.asarray(z_fixed, dtype=float)
        A_sel = A_sel[:, :control_dim]
    A_hat, b_hat, kept, degenerate = _normalize(A_sel, b_sel, labels, tol)
    out = margin_lp(A_hat, b_hat, kept, bounds, tol)
    out["degenerate_violated"] = degenerate
    out["s_lp"] = out["s_star"]
    if degenerate:
        out["s_star"] = math.inf
        out["binding"] = [d["label"] for d in degenerate]
    out["x"] = None  # keep reports small
    return out

"""Per-control-update feasibility recorder attached to DistributedController.qp_observer.

Outputs (in ``out_dir``):
  qp_log.csv.gz          one row per (step, agent): solver status, LP margin, violations,
                         and ``applied`` (1 = the input was applied to the state, 0 = the
                         step was not applied because the run terminated, -1 = unknown)
  step_log.csv.gz        one row per step: spectrum and graph statistics
  topology_events.csv    every edge appearance/disappearance in each graph
  arrays.npz             compact per-step/per-agent arrays for plotting
  failures/*.json        full snapshot (x, z, w, y, A, b, analyses) of each non-solved QP
  summary.json           counts and extremes (written by ``close``)

The recorder only reads controller data. It never returns anything to the controller.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from controller.centralized_qp import build_centralized_qp_data
from controller.fast_dynamics import (
    alpha_ij_connectivity,
    clf_coupling_gates,
    compute_V_i,
    compute_beta_i,
    connectivity_sigma,
    h_ij,
)
from controller.local_qp import build_local_qp_data
from models.fast_state import EdgeMismatchState, FastState
from diagnostics.feasibility import (
    CLASS_CODE,
    CLASSES,
    GROUPS,
    SOLVER_SUCCESS,
    VERIFIED_SOLVED,
    Tolerances,
    centralized_feasibility,
    classify,
    constraint_violation,
    group_slacks,
    group_subset_analysis,
    local_feasibility,
    lp_verdict,
    row_group,
)

GRAPHS = ("comm", "alpha", "clf")
QP_COLUMNS = [
    "step", "t", "agent", "solver_status", "solver_success", "class", "applied", "lp_verdict",
    "s_star", "lp_status", "binding_rows", "degenerate_rows",
    "viol_raw", "viol_raw_row", "viol_norm", "viol_norm_row", "viol_scaled", "viol_flag",
    "slack_pairwise", "slack_obstacle", "slack_connectivity", "slack_clf",
    "u_x", "u_y", "num_iters", "solve_time", "pri_res", "dua_res",
    "deg_comm", "deg_alpha", "beta_norm",
    "conn_nominal_rhs", "conn_mismatch", "clf_V", "clf_gamma3W", "clf_mismatch",
    "pair_mismatch_maxabs",
]
STEP_COLUMNS = [
    "step", "t", "lambda2", "lambda3", "spectral_gap",
    "edges_comm", "edges_alpha", "edges_clf", "comp_comm", "comp_alpha", "comp_clf",
    "events_comm", "events_alpha", "events_clf", "min_pair_distance",
]
STATUS_CODES = ["optimal", "optimal_inaccurate", "infeasible", "infeasible_inaccurate",
                "unbounded", "unbounded_inaccurate", "user_limit", "solver_error", "exception", "other"]


def _components(n: int, edges: set[tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n)})


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else repr(v)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


def _fast_state_snapshot(fs: FastState) -> dict[str, Any]:
    return {
        "w": list(fs.w),
        "z_pair_self": {j: e.self_component for j, e in fs.z_pair.items()},
        "z_c": fs.z_c,
        "z_clf": fs.z_clf,
        "y_pair_self": {j: e.self_component for j, e in fs.y_pair.items()},
        "y_c": fs.y_c,
        "y_clf": fs.y_clf,
        "y_obs": list(fs.y_obs),
        "y_u": list(fs.y_u),
    }


def _zero_mismatch(fs: FastState) -> FastState:
    return FastState(
        w=fs.w,
        z_pair={j: EdgeMismatchState(0.0, e.neighbor_component) for j, e in fs.z_pair.items()},
        z_c=0.0,
        z_clf=0.0,
        y_pair=fs.y_pair,
        y_c=fs.y_c,
        y_clf=fs.y_clf,
        y_obs=fs.y_obs,
        y_u=fs.y_u,
    )


def _z_vector(decision_labels: list[str], control_dim: int, all_fast_states: dict[int, FastState]) -> np.ndarray:
    """Current mismatch values in the centralized decision ordering."""
    values: list[float] = []
    for label in decision_labels[control_dim:]:
        if label.startswith("z_pair["):
            pair, comp = label[len("z_pair["):].split("].")
            i, j = (int(v) for v in pair.split(","))
            k = int(comp)
            other = j if k == i else i
            values.append(all_fast_states[k].z_pair[other].self_component)
        elif label.startswith("z_c["):
            values.append(all_fast_states[int(label[4:-1])].z_c)
        elif label.startswith("z_clf["):
            values.append(all_fast_states[int(label[6:-1])].z_clf)
        else:
            raise ValueError(f"Unknown decision label {label!r}")
    return np.asarray(values, dtype=float)


class FeasibilityRecorder:
    """Callable observer: ``controller.qp_observer = recorder``."""

    def __init__(
        self,
        out_dir: Path,
        *,
        agent_ids: list[int],
        n_steps: int,
        dt: float,
        u_max: float,
        d_c: float,
        epsilon_c: float,
        xi: float,
        tol: Tolerances = Tolerances(),
        max_detail_files: int = 200,
        progress_every: int = 5000,
    ) -> None:
        self.out_dir = Path(out_dir)
        (self.out_dir / "failures").mkdir(parents=True, exist_ok=True)
        self.agent_ids = list(agent_ids)
        self.col = {a: k for k, a in enumerate(self.agent_ids)}
        self.n = len(agent_ids)
        self.n_steps = n_steps
        self.dt = dt
        self.u_max = u_max
        self.d_c = d_c
        self.epsilon_c = epsilon_c
        self.xi = xi
        self.tol = tol
        self.max_detail_files = max_detail_files
        self.progress_every = progress_every

        shape = (n_steps, self.n)
        self.a_s_star = np.full(shape, np.nan)
        self.a_viol_norm = np.full(shape, np.nan)
        self.a_viol_raw = np.full(shape, np.nan)
        self.a_class = np.full(shape, -1, dtype=np.int8)
        self.a_status = np.full(shape, -1, dtype=np.int8)
        self.a_slack = np.full((n_steps, self.n, len(GROUPS)), np.nan, dtype=np.float32)
        self.a_conn_mismatch = np.full(shape, np.nan, dtype=np.float32)
        self.a_clf_mismatch = np.full(shape, np.nan, dtype=np.float32)
        self.a_positions = np.full((n_steps, self.n, 2), np.nan)
        self.a_lambda2 = np.full((n_steps,), np.nan)
        self.a_lambda3 = np.full((n_steps,), np.nan)
        self.a_edges = np.full((n_steps, len(GRAPHS)), -1, dtype=np.int16)
        self.a_events = np.zeros((n_steps, len(GRAPHS)), dtype=np.int16)

        self._qp_file = gzip.open(self.out_dir / "qp_log.csv.gz", "wt", newline="")
        self._qp = csv.writer(self._qp_file)
        self._qp.writerow(QP_COLUMNS)
        self._step_file = gzip.open(self.out_dir / "step_log.csv.gz", "wt", newline="")
        self._step = csv.writer(self._step_file)
        self._step.writerow(STEP_COLUMNS)
        self._ev_file = open(self.out_dir / "topology_events.csv", "w", newline="")
        self._ev = csv.writer(self._ev_file)
        self._ev.writerow(["step", "t", "graph", "i", "j", "event"])

        self._cur_step: int | None = None
        self._prev_edges: dict[str, set[tuple[int, int]]] | None = None
        self._centralized_cache: dict[int, dict[str, Any]] = {}
        self.class_counts: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.max_viol: dict[str, Any] = {"normalized": -math.inf, "raw": -math.inf}
        self.tightest_margin: dict[str, Any] = {"s_star": -math.inf}
        self.failures: list[dict[str, Any]] = []
        self._pending_rows: list[list[Any]] = []
        self._pending_step: int | None = None
        self.n_viol_flag = 0
        self.max_pri_res = -math.inf
        self.n_pri_res_above_eps = 0
        self.detail_files = 0
        self.steps_recorded = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------ step level
    def _flush_pending(self, applied: int) -> None:
        for row in self._pending_rows:
            # failed solves never produce an applied input
            row[QP_COLUMNS.index("applied")] = applied if row[QP_COLUMNS.index("solver_success")] else 0
            self._qp.writerow(row)
        self._pending_rows = []

    def _begin_step(self, step: int, all_agent_states: dict, runtime: dict) -> None:
        # Reaching a new step means the previous step's inputs were applied.
        self._flush_pending(applied=1)
        self._cur_step = step
        self.steps_recorded = step + 1
        ids = self.agent_ids
        pos = {i: all_agent_states[i].position for i in ids}
        for i in ids:
            self.a_positions[step, self.col[i]] = pos[i]
        thr = self.d_c - self.epsilon_c
        sigma_target = runtime.get("sigma")
        center = runtime.get("target_center")
        eps = float(runtime.get("epsilon", 1e-2))
        h_i = {
            i: (math.exp(-1.0 / v) if (v := compute_V_i(pos[i], center, sigma_target, eps)) > 0.0 else 0.0)
            for i in ids
        }
        edges = {g: set() for g in GRAPHS}
        min_d = math.inf
        adj = np.zeros((self.n, self.n))
        sig = connectivity_sigma(self.d_c, self.epsilon_c)
        for a, i in enumerate(ids):
            for j in ids[a + 1:]:
                d = math.dist(pos[i], pos[j])
                min_d = min(min_d, d)
                h = h_ij(pos[i], pos[j], self.d_c)
                if h > 0.0:
                    edges["comm"].add((i, j))
                    if h_i[i] > 0.0 and h_i[j] > 0.0:
                        edges["clf"].add((i, j))
                if d < thr:
                    edges["alpha"].add((i, j))
                w = alpha_ij_connectivity(pos[i], pos[j], self.d_c, self.epsilon_c, sig)
                adj[self.col[i], self.col[j]] = adj[self.col[j], self.col[i]] = w
        eig = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)
        lam2 = float(runtime.get("lambda_2", eig[1] if self.n > 1 else 0.0))
        lam3 = float(eig[2]) if self.n > 2 else math.nan
        t = step * self.dt
        n_events = []
        for g_idx, g in enumerate(GRAPHS):
            count = 0
            if self._prev_edges is not None:
                for e in sorted(edges[g] - self._prev_edges[g]):
                    self._ev.writerow([step, t, g, e[0], e[1], "appear"])
                    count += 1
                for e in sorted(self._prev_edges[g] - edges[g]):
                    self._ev.writerow([step, t, g, e[0], e[1], "disappear"])
                    count += 1
            n_events.append(count)
            self.a_edges[step, g_idx] = len(edges[g])
            self.a_events[step, g_idx] = count
        self._prev_edges = edges
        self.a_lambda2[step] = lam2
        self.a_lambda3[step] = lam3
        comps = [_components(self.n, {(self.col[i], self.col[j]) for i, j in edges[g]}) for g in GRAPHS]
        self._step.writerow(
            [step, t, lam2, lam3, lam3 - lam2,
             *[len(edges[g]) for g in GRAPHS], *comps, *n_events, min_d]
        )
        self._edges_now = edges
        if self.progress_every and step % self.progress_every == 0:
            bad = sum(v for k, v in self.class_counts.items() if k not in ("solved",))
            print(
                f"[feasibility] step={step} t={t:.3f} elapsed={time.time() - self._t0:.0f}s "
                f"non_solved={bad} counts={dict(self.class_counts)}",
                file=sys.stderr,
                flush=True,
            )

    # ------------------------------------------------------------------ per QP
    def __call__(
        self,
        *,
        step_idx: int | None,
        agent_id: int,
        qp_data,
        solve_result,
        qp_kwargs: dict[str, Any],
        env,
        runtime_kwargs: dict[str, Any],
    ) -> None:
        if not isinstance(step_idx, int):
            return
        if step_idx != self._cur_step:
            self._begin_step(step_idx, qp_kwargs["all_agent_states"], runtime_kwargs)
        tol = self.tol
        labels = list(qp_data.row_labels)
        A_raw = np.asarray(qp_data.A_raw, dtype=float)
        b_raw = np.asarray(qp_data.b_raw, dtype=float)
        A_sc = np.asarray(qp_data.A, dtype=float)
        b_sc = np.asarray(qp_data.b, dtype=float)

        feas = local_feasibility(A_raw, b_raw, labels, self.u_max, tol)
        s_star = feas["s_star"]
        status = solve_result.status or "none"
        cls = classify(status, solve_result.success, s_star, tol)
        verdict = lp_verdict(s_star, tol)

        if solve_result.success:
            u_eval = np.asarray(solve_result.u, dtype=float)
            viol = constraint_violation(A_raw, b_raw, A_sc, b_sc, labels, u_eval, tol)
        else:
            u_eval = np.asarray(feas["x"], dtype=float) if feas["x"] is not None else np.zeros(2)
            viol = {"raw": math.nan, "raw_label": "", "normalized": math.nan, "normalized_label": "", "scaled": math.nan}
        slacks = group_slacks(A_raw, b_raw, labels, u_eval, tol)
        viol_flag = bool(solve_result.success and viol["normalized"] > tol.viol_tol)
        dec = self._decomposition(agent_id, qp_kwargs)

        step, k = step_idx, self.col[agent_id]
        t = step * self.dt
        self.a_s_star[step, k] = s_star
        self.a_viol_norm[step, k] = viol["normalized"]
        self.a_viol_raw[step, k] = viol["raw"]
        self.a_class[step, k] = CLASS_CODE[cls]
        self.a_status[step, k] = STATUS_CODES.index(status) if status in STATUS_CODES else len(STATUS_CODES) - 1
        for g_idx, g in enumerate(GROUPS):
            self.a_slack[step, k, g_idx] = slacks[g]
        self.a_conn_mismatch[step, k] = dec["conn_mismatch"]
        self.a_clf_mismatch[step, k] = dec["clf_mismatch"]

        self.class_counts[cls] += 1
        self.status_counts[status] += 1
        if solve_result.success:
            if viol["normalized"] > self.max_viol["normalized"]:
                self.max_viol.update(normalized=viol["normalized"], normalized_row=viol["normalized_label"],
                                     normalized_step=step, normalized_agent=agent_id)
            if viol["raw"] > self.max_viol["raw"]:
                self.max_viol.update(raw=viol["raw"], raw_row=viol["raw_label"], raw_step=step, raw_agent=agent_id)
            if s_star > self.tightest_margin["s_star"]:
                self.tightest_margin = {"s_star": s_star, "step": step, "agent": agent_id, "binding": feas["binding"]}

        if viol_flag:
            self.n_viol_flag += 1
        if solve_result.pri_res is not None:
            self.max_pri_res = max(self.max_pri_res, solve_result.pri_res)
            self.n_pri_res_above_eps += solve_result.pri_res > 1e-5
        self._pending_rows.append([
            step, t, agent_id, status, int(solve_result.success), cls, None, verdict,
            s_star, feas["lp_status"], ";".join(feas["binding"]),
            ";".join(d["label"] for d in feas["degenerate_violated"]),
            viol["raw"], viol["raw_label"], viol["normalized"], viol["normalized_label"], viol["scaled"], int(viol_flag),
            *[slacks[g] for g in GROUPS],
            solve_result.u[0] if solve_result.success else math.nan,
            solve_result.u[1] if solve_result.success else math.nan,
            solve_result.num_iters, solve_result.solve_time, solve_result.pri_res, solve_result.dua_res,
            dec["deg_comm"], dec["deg_alpha"], dec["beta_norm"],
            dec["conn_nominal_rhs"], dec["conn_mismatch"], dec["clf_V"], dec["clf_gamma3W"], dec["clf_mismatch"],
            dec["pair_mismatch_maxabs"],
        ])

        if cls not in VERIFIED_SOLVED or viol_flag:
            self._record_detail(step, agent_id, cls, status, solve_result, feas, viol, slacks, dec,
                                qp_data, qp_kwargs, runtime_kwargs)

    def _decomposition(self, i: int, kw: dict[str, Any]) -> dict[str, Any]:
        """Split connectivity/CLF right-hand sides into nominal and mismatch parts."""
        fs_all: dict[int, FastState] = kw["all_fast_states"]
        xs = {j: s.position for j, s in kw["all_agent_states"].items()}
        x_i = xs[i]
        n_agents = kw["n_agents"]
        center, sigma, eps = kw["target_center"], kw["sigma"], kw["epsilon"]
        v_i = compute_V_i(x_i, center, sigma, eps)
        h_i = math.exp(-1.0 / v_i) if v_i > 0.0 else 0.0
        clf_coupling = kw.get("clf_coupling", "gated")
        conn_mm = 0.0
        clf_mm = 0.0
        deg_comm = deg_alpha = 0
        thr = kw["d_c"] - kw["epsilon_c"]
        for j, fs_j in fs_all.items():
            if j == i:
                continue
            h = h_ij(x_i, xs[j], kw["d_c"])
            deg_comm += h > 0.0
            deg_alpha += math.dist(x_i, xs[j]) < thr
            conn_mm += h * (fs_all[i].z_c - fs_j.z_c)
            if h_i > 0.0 or clf_coupling == "communication":
                v_j = compute_V_i(xs[j], center, sigma, eps)
                h_j = math.exp(-1.0 / v_j) if v_j > 0.0 else 0.0
                g_i, g_j = clf_coupling_gates(h_i, h_j, clf_coupling)
                clf_mm += h * g_i * g_j * (fs_all[i].z_clf - fs_j.z_clf)
        pair_mm = 0.0
        for j, e in fs_all[i].z_pair.items():
            h = h_ij(x_i, xs[j], kw["d_c"])
            pair_mm = max(pair_mm, abs(h * (e.self_component - fs_all[j].z_pair[i].self_component)))
        if kw["enable_connectivity_constraint"]:
            beta = compute_beta_i(i, fs_all[i], fs_all, kw["all_agent_states"], kw["d_c"], kw["epsilon_c"], kw["v2"])
            beta_norm = math.hypot(*beta)
            conn_nominal = float(kw["lambda_2"]) / n_agents - kw["eta"] / n_agents
        else:
            beta_norm, conn_nominal = math.nan, math.nan
        return {
            "deg_comm": deg_comm,
            "deg_alpha": deg_alpha,
            "beta_norm": beta_norm,
            "conn_nominal_rhs": conn_nominal,
            "conn_mismatch": kw["alpha_connectivity"] * conn_mm,
            "clf_V": v_i,
            "clf_gamma3W": kw["gamma_3"] * math.exp(-1.0 / v_i) * v_i if v_i > 0.0 else 0.0,
            "clf_mismatch": kw["alpha_clf"] * clf_mm,
            "pair_mismatch_maxabs": pair_mm,
        }

    # ------------------------------------------------------------------ failures
    def _centralized_check(self, step: int, qp_kwargs: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
        if step in self._centralized_cache:
            return self._centralized_cache[step]
        kw = qp_kwargs
        cqp = build_centralized_qp_data(
            all_agent_states=kw["all_agent_states"],
            all_fast_states=kw["all_fast_states"],
            obstacles=kw["obstacles"],
            u_max=kw["u_max"],
            xi=self.xi,
            gamma_1=kw["gamma_1"],
            d_0=kw["d_0"],
            gamma_2=kw["gamma_2"],
            eta=kw["eta"],
            d_c=kw["d_c"],
            epsilon_c=kw["epsilon_c"],
            gamma_3=kw["gamma_3"],
            target_center=kw["target_center"],
            sigma=kw["sigma"],
            epsilon=kw["epsilon"],
            enable_connectivity_constraint=kw["enable_connectivity_constraint"],
            alpha_clf=kw["alpha_clf"],
            alpha_connectivity=kw["alpha_connectivity"],
            lambda_2=kw["lambda_2"],
            v2=kw["v2"],
            clf_coupling=kw.get("clf_coupling", "gated"),
        )
        A = np.asarray(cqp.A, dtype=float)
        b = np.asarray(cqp.b, dtype=float)
        n_dec = len(cqp.q)
        args = (A, b, cqp.row_labels, cqp.row_blocks, cqp.control_dim, n_dec, self.u_max, self.tol)
        free = centralized_feasibility(*args)
        z_now = _z_vector(cqp.decision_labels, cqp.control_dim, kw["all_fast_states"])
        pinned = centralized_feasibility(*args, z_fixed=z_now)
        present = [g for g in GROUPS if g in set(cqp.row_blocks)]
        out: dict[str, Any] = {
            "z_free": {"s_star": free["s_star"], "verdict": lp_verdict(free["s_star"], self.tol),
                       "binding": free["binding"], "degenerate_violated": free["degenerate_violated"]},
            "z_pinned_to_current": {"s_star": pinned["s_star"], "verdict": lp_verdict(pinned["s_star"], self.tol),
                                    "binding": pinned["binding"]},
            "block_counts": cqp.block_counts,
        }
        if lp_verdict(free["s_star"], self.tol) != "feasible":
            out["z_free_group_subsets"] = group_subset_analysis(
                lambda groups: centralized_feasibility(*args, groups=groups), present, self.tol
            )
        self._centralized_cache = {step: out}
        return out

    def _record_detail(self, step, agent_id, cls, status, solve_result, feas, viol, slacks, dec,
                       qp_data, qp_kwargs, runtime) -> None:
        labels = list(qp_data.row_labels)
        A_raw = np.asarray(qp_data.A_raw, dtype=float)
        b_raw = np.asarray(qp_data.b_raw, dtype=float)
        present = sorted({row_group(l) for l in labels} & set(GROUPS), key=GROUPS.index)
        subsets = group_subset_analysis(
            lambda groups: local_feasibility(A_raw, b_raw, labels, self.u_max, self.tol, groups=groups),
            present,
            self.tol,
        )
        fs_all: dict[int, FastState] = qp_kwargs["all_fast_states"]
        zero_states = {j: _zero_mismatch(fs) for j, fs in fs_all.items()}
        cf_kwargs = {**qp_kwargs, "state": zero_states[agent_id], "all_fast_states": zero_states}
        cf_qp = build_local_qp_data(**cf_kwargs)
        cf = local_feasibility(np.asarray(cf_qp.A_raw), np.asarray(cf_qp.b_raw), list(cf_qp.row_labels), self.u_max, self.tol)
        central = self._centralized_check(step, qp_kwargs, runtime)

        summary = {
            "step": step,
            "t": step * self.dt,
            "agent": agent_id,
            "class": cls,
            "solver_status": status,
            "solver_success": bool(solve_result.success),
            "lp_status": feas["lp_status"],
            "s_star": feas["s_star"],
            "binding_rows": feas["binding"],
            "minimal_infeasible_subsets": subsets["minimal_infeasible_subsets"],
            "zero_mismatch_s_star": cf["s_star"],
            "centralized_z_free_verdict": central["z_free"]["verdict"],
            "centralized_z_free_s_star": central["z_free"]["s_star"],
            "viol_normalized": viol["normalized"],
        }
        self.failures.append(summary)
        if self.detail_files >= self.max_detail_files:
            return
        self.detail_files += 1
        xs = qp_kwargs["all_agent_states"]
        detail = {
            **summary,
            "solver": {k: getattr(solve_result, k) for k in ("reason", "status", "num_iters", "solve_time", "pri_res", "dua_res")},
            "u_returned": list(solve_result.u) if solve_result.success else None,
            "lp_u": feas["x"],
            "degenerate_violated": feas["degenerate_violated"],
            "violation": viol,
            "group_slacks_at_u": slacks,
            "group_subset_margins": subsets["subset_margins"],
            "decomposition": dec,
            "zero_mismatch_counterfactual": {"s_star": cf["s_star"], "binding": cf["binding"],
                                             "verdict": lp_verdict(cf["s_star"], self.tol)},
            "centralized": central,
            "lambda_2": qp_kwargs["lambda_2"],
            "v2": qp_kwargs["v2"],
            "positions": {j: list(s.position) for j, s in xs.items()},
            "fast_states": {j: _fast_state_snapshot(fs) for j, fs in fs_all.items()},
            "row_labels": labels,
            "A_raw": qp_data.A_raw,
            "b_raw": qp_data.b_raw,
            "A_scaled": qp_data.A,
            "b_scaled": qp_data.b,
            "tolerances": vars(self.tol) if hasattr(self.tol, "__dict__") else str(self.tol),
        }
        path = self.out_dir / "failures" / f"step{step:07d}_agent{agent_id}_{cls}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(detail), f, indent=1)

    # ------------------------------------------------------------------ output
    def close(self, extra: dict[str, Any] | None = None, steps_completed: int | None = None) -> dict[str, Any]:
        """Flush logs and write summary.json.

        ``steps_completed`` = number of state updates actually applied by the simulator;
        it decides whether the last recorded step's inputs were applied.
        """
        if self._cur_step is not None:
            if steps_completed is None:
                self._flush_pending(applied=-1)
            else:
                self._flush_pending(applied=1 if self._cur_step < steps_completed else 0)
        for f in (self._qp_file, self._step_file, self._ev_file):
            f.close()
        n = self.steps_recorded
        np.savez_compressed(
            self.out_dir / "arrays.npz",
            agent_ids=np.asarray(self.agent_ids),
            dt=self.dt,
            classes=np.asarray(CLASSES),
            groups=np.asarray(GROUPS),
            graphs=np.asarray(GRAPHS),
            status_codes=np.asarray(STATUS_CODES),
            s_star=self.a_s_star[:n],
            viol_norm=self.a_viol_norm[:n],
            viol_raw=self.a_viol_raw[:n],
            cls=self.a_class[:n],
            status=self.a_status[:n],
            slack=self.a_slack[:n],
            conn_mismatch=self.a_conn_mismatch[:n],
            clf_mismatch=self.a_clf_mismatch[:n],
            positions=self.a_positions[:n],
            lambda2=self.a_lambda2[:n],
            lambda3=self.a_lambda3[:n],
            edges=self.a_edges[:n],
            events=self.a_events[:n],
        )
        not_verified = [f for f in self.failures if f["class"] not in VERIFIED_SOLVED]
        first_not_verified = not_verified[0] if not_verified else None
        first_terminating = next((f for f in self.failures if not f["solver_success"]), None)
        cls_arr = self.a_class[:n]
        solved_mask = np.isin(cls_arr, [CLASS_CODE[c] for c in VERIFIED_SOLVED])
        osqp_ok_mask = np.isin(self.a_status[:n], [0, 1])  # optimal / optimal_inaccurate
        v = self.a_viol_norm[:n][osqp_ok_mask]
        v = v[np.isfinite(v)]
        buckets = {"<=1e-9": 0, "(1e-9,1e-6]": 0, "(1e-6,1e-5]": 0, "(1e-5,1e-4]": 0, ">1e-4": 0}
        if v.size:
            edges = [1e-9, 1e-6, 1e-5, 1e-4]
            idx = np.searchsorted(edges, v, side="left")
            for key, count in zip(buckets, np.bincount(idx, minlength=5)):
                buckets[key] = int(count)
        summary = {
            "steps_recorded": n,
            "qp_solves": int(sum(self.class_counts.values())),
            "class_counts": {c: int(self.class_counts.get(c, 0)) for c in CLASSES},
            "solver_status_counts": dict(self.status_counts),
            "verification": {
                "verified_feasible_and_solved": int(solved_mask.sum()),
                "not_verified": len(not_verified),
                "osqp_success_but_lp_infeasible": {
                    "count": int(self.class_counts.get("inconsistent", 0)),
                    "note": "OSQP reported success but the independent LP found no admissible u; "
                            "the OSQP input was applied (diagnostics never intervene).",
                    "cases": [f for f in not_verified if f["class"] == "inconsistent"][:200],
                },
                "lp_errors": {
                    "count": int(self.class_counts.get("lp_error", 0)),
                    "note": "The auxiliary LP failed; feasibility of these QPs is NOT verified.",
                    "cases": [f for f in not_verified if f["class"] == "lp_error"][:200],
                },
                "borderline": int(self.class_counts.get("borderline", 0)),
                "solver_failures_lp_feasible (numerical_failure)": int(self.class_counts.get("numerical_failure", 0)),
                "solver_failures_lp_infeasible (infeasible)": int(self.class_counts.get("infeasible", 0)),
            },
            "tolerance": {
                "description": "Normalized violation max_r (a_r^T u - b_r)/||a_r|| of OSQP-successful solves, "
                               "control units; OSQP eps_abs = eps_rel = 1e-5, no polishing.",
                "violation_buckets": buckets,
                "violation_quantiles": (
                    {q: float(np.quantile(v, q)) for q in (0.5, 0.9, 0.99, 1.0)} if v.size else None
                ),
                "count_above_viol_tol": int(self.n_viol_flag),
                "viol_tol": self.tol.viol_tol,
                "optimal_inaccurate_count": int(self.status_counts.get("optimal_inaccurate", 0)),
                "max_osqp_primal_residual": None if not math.isfinite(self.max_pri_res) else self.max_pri_res,
                "count_osqp_primal_residual_above_1e-5": int(self.n_pri_res_above_eps),
            },
            "max_violation_of_solved_qps": self.max_viol,
            "tightest_lp_margin_of_solved_qps": self.tightest_margin,
            "first_terminating_failure": first_terminating,
            "first_not_verified": first_not_verified,
            "first_non_solved": first_not_verified,
            "non_solved_qps": not_verified[:500],
            "solved_with_violation_flag": sum(1 for f in self.failures if f["class"] in VERIFIED_SOLVED),
            "topology_events": {g: int(self.a_events[:n, k].sum()) for k, g in enumerate(GRAPHS)},
            "min_lambda2": float(np.nanmin(self.a_lambda2[:n])) if n else None,
            "tolerances": vars(self.tol),
            "wall_time_s": time.time() - self._t0,
        }
        if extra:
            summary.update(extra)
        with (self.out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2)
        return summary

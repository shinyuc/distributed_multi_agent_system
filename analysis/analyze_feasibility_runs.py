"""Post-hoc analysis of a completed feasibility run, compared against a previous run.

Read-only with respect to the run directories: every result (JSON, figures) goes to a
new unique directory under ``--out-root``. The YAML config is parsed directly (not via
config.load_config, which writes an initialization record as a side effect).

Usage:
    python -m analysis.analyze_feasibility_runs \
        --new outputs/feasibility/<new run> --previous outputs/feasibility/default_5_agents \
        --config configs/default_5_agents.yaml
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import maximum_filter1d
from scipy.stats import spearmanr

from diagnostics.plot_feasibility import CRITICAL, GRID, INK, INK_2, MUTED, SERIES, SURFACE, _style, plot_run
from utils.run_dirs import make_unique_run_dir

AGENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GROUPS = ["pairwise", "obstacle", "connectivity", "clf"]
ACTIVE_TOL = 1e-4  # normalized slack below this at the applied u counts as active


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load_run(d: Path) -> dict:
    run = {"dir": d}
    run["meta"] = json.loads((d / "run_meta.json").read_text())
    run["summary"] = json.loads((d / "summary.json").read_text())
    run["arrays"] = dict(np.load(d / "arrays.npz"))
    tj = json.loads((d / "trajectories.json").read_text())
    run["traj"] = np.asarray(tj["trajectories"], dtype=float)
    run["traj_json"] = tj
    with (d / "topology_events.csv").open() as f:
        run["events"] = list(csv.DictReader(f))
    rr = d / "run_report.json"
    run["report"] = json.loads(rr.read_text()) if rr.exists() else None
    return run


# ----------------------------------------------------------------------------- QP logs
def stream_qp_logs(new: dict, prev: dict, top_k: int = 40) -> dict:
    """One pass over both qp_log files (same row order: step-major, agent-minor)."""
    applied_u = new["traj_json"].get("control_history_applied")
    status, cls, verdict, applied, success = Counter(), Counter(), Counter(), Counter(), Counter()
    binding_patterns, active_groups = Counter(), Counter()
    max_raw = {"value": -math.inf}
    max_norm = {"value": -math.inf}
    pri_above = 0
    top: list[tuple[float, int, dict]] = []
    cmp_mismatch = Counter()
    max_du = 0.0
    applied_mismatch = 0
    ratio_clf_max = {"value": 0.0}
    ratio_conn_max = {"value": 0.0}
    n = 0
    fn, fp = gzip.open(new["dir"] / "qp_log.csv.gz", "rt"), gzip.open(prev["dir"] / "qp_log.csv.gz", "rt")
    for rn, rp in zip(csv.DictReader(fn), csv.DictReader(fp)):
        n += 1
        step, agent = int(rn["step"]), int(rn["agent"])
        status[rn["solver_status"]] += 1
        cls[rn["class"]] += 1
        verdict[rn["lp_verdict"]] += 1
        applied[rn.get("applied", "n/a")] += 1
        success[rn["solver_success"]] += 1
        vr, vn, s = _f(rn["viol_raw"]), _f(rn["viol_norm"]), _f(rn["s_star"])
        if vr > max_raw["value"]:
            max_raw = {"value": vr, "row": rn["viol_raw_row"], "step": step, "agent": agent}
        if vn > max_norm["value"]:
            max_norm = {"value": vn, "row": rn["viol_norm_row"], "step": step, "agent": agent}
        pri_above += _f(rn["pri_res"]) > 1e-5
        groups = sorted({b.split("(", 1)[0] for b in rn["binding_rows"].split(";") if b})
        binding_patterns["+".join(groups) or "none"] += 1
        for g in GROUPS:
            if _f(rn[f"slack_{g}"]) < ACTIVE_TOL:
                active_groups[g] += 1
        g3w, clf_mm = _f(rn["clf_gamma3W"]), _f(rn["clf_mismatch"])
        if g3w > 0 and abs(clf_mm) / g3w > ratio_clf_max["value"]:
            ratio_clf_max = {"value": abs(clf_mm) / g3w, "step": step, "agent": agent}
        cn, cm = _f(rn["conn_nominal_rhs"]), _f(rn["conn_mismatch"])
        if abs(cn) > 0 and abs(cm) / abs(cn) > ratio_conn_max["value"]:
            ratio_conn_max = {"value": abs(cm) / abs(cn), "step": step, "agent": agent}
        item = (s, n, {k: rn[k] for k in (
            "step", "t", "agent", "s_star", "binding_rows", "slack_pairwise", "slack_obstacle",
            "slack_connectivity", "slack_clf", "u_x", "u_y", "deg_comm", "deg_alpha", "beta_norm",
            "conn_nominal_rhs", "conn_mismatch", "clf_V", "clf_gamma3W", "clf_mismatch",
            "pair_mismatch_maxabs", "viol_norm")})
        if len(top) < top_k:
            heapq.heappush(top, item)
        elif s > top[0][0]:
            heapq.heapreplace(top, item)
        # previous run, same (step, agent)
        assert int(rp["step"]) == step and int(rp["agent"]) == agent
        for key in ("u_x", "u_y", "s_star", "solver_status", "class", "viol_norm", "binding_rows", "lp_verdict"):
            if rn[key] != rp[key]:
                cmp_mismatch[key] += 1
        max_du = max(max_du, abs(_f(rn["u_x"]) - _f(rp["u_x"])), abs(_f(rn["u_y"]) - _f(rp["u_y"])))
        if applied_u is not None:
            ua = applied_u[str(agent)][step]
            if ua[0] != _f(rn["u_x"]) or ua[1] != _f(rn["u_y"]):
                applied_mismatch += 1
    leftover_prev = sum(1 for _ in csv.DictReader(fp))
    fn.close()
    fp.close()
    return {
        "rows": n,
        "rows_only_in_previous": leftover_prev,
        "solver_status": dict(status),
        "class": dict(cls),
        "lp_verdict": dict(verdict),
        "applied_flag": dict(applied),
        "solver_success": dict(success),
        "max_violation_raw": max_raw,
        "max_violation_normalized": max_norm,
        "osqp_primal_residual_above_1e-5": int(pri_above),
        "lp_binding_group_patterns": dict(binding_patterns.most_common()),
        "active_groups_at_applied_u": {g: int(active_groups[g]) for g in GROUPS},
        "max_ratio_clf_mismatch_to_gamma3W": ratio_clf_max,
        "max_ratio_conn_mismatch_to_nominal_rhs": ratio_conn_max,
        "top_tightest_qps": [r for _, _, r in sorted(top, key=lambda x: -x[0])],
        "compare_with_previous": {
            "fields_differing_row_count": dict(cmp_mismatch),
            "max_abs_diff_u": max_du,
        },
        "applied_history_vs_qp_solution_mismatches": None if applied_u is None else applied_mismatch,
    }


# ----------------------------------------------------------------------------- safety etc.
def safety_connectivity_task(run: dict, cfg: dict) -> dict:
    traj = run["traj"]  # (K+1, N, 2)
    dt = cfg["simulation"]["dt"]
    n = traj.shape[1]
    d0 = cfg["controller"]["safety_distance"]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dists = np.stack([np.linalg.norm(traj[:, i] - traj[:, j], axis=1) for i, j in pairs], axis=1)
    k, p = np.unravel_index(np.argmin(dists), dists.shape)
    out = {"inter_agent": {
        "min_distance": float(dists[k, p]), "step": int(k), "t": float(k * dt), "pair": list(pairs[p]),
        "safety_distance": d0, "samples_below_safety_distance": int((dists < d0).any(axis=1).sum()),
    }}
    obstacles = cfg["obstacles"]["obstacles"]
    clear = np.stack([
        np.linalg.norm(traj - np.asarray(o["center"]), axis=2) - o["radius"] for o in obstacles
    ], axis=2)  # (K+1, N, M)
    k, i, m = np.unravel_index(np.argmin(clear), clear.shape)
    viol = clear < 0
    vk, vi, vm = np.nonzero(viol)
    episodes = []
    if vk.size:
        for agent, obs in sorted(set(zip(vi.tolist(), vm.tolist()))):
            steps = np.flatnonzero(viol[:, agent, obs])
            episodes.append({
                "agent": agent, "obstacle": obs, "n_samples": int(steps.size),
                "first_step": int(steps[0]), "last_step": int(steps[-1]),
                "t_first": float(steps[0] * dt), "t_last": float(steps[-1] * dt),
                "max_penetration": float(-clear[steps, agent, obs].min()),
                "barrier_h_min": float(((clear[steps, agent, obs] + obstacles[obs]["radius"]) ** 2
                                        - obstacles[obs]["radius"] ** 2).min()),
            })
    out["obstacles"] = {
        "min_clearance": float(clear[k, i, m]), "step": int(k), "t": float(k * dt), "agent": int(i),
        "obstacle": int(m), "obstacle_center": obstacles[m]["center"], "obstacle_radius": obstacles[m]["radius"],
        "samples_with_penetration": int(viol.any(axis=(1, 2)).sum()), "episodes": episodes,
    }
    a = run["arrays"]
    eta = cfg["fast_dynamics"]["eta"]
    lam = a["lambda2"]
    out["connectivity"] = {
        "eta": eta,
        "lambda2_weighted_min": float(np.nanmin(lam)), "lambda2_weighted_min_step": int(np.nanargmin(lam)),
        "lambda2_weighted_min_t": float(np.nanargmin(lam) * dt),
        "lambda2_weighted_always_above_eta": bool(np.all(lam > eta)),
        "lambda2_weighted_min_margin_over_eta": float(np.nanmin(lam) - eta),
        "lambda2_binary_comm_min": float(min(run["traj_json"]["lambda2_comm"])),
        "lambda2_binary_cons_min": float(min(run["traj_json"]["lambda2_cons"])),
        "note": "lambda2_weighted is the alpha-graph value used in the connectivity constraint, "
                "evaluated at each control update (states x_0..x_{K-1}); binary values over all K+1 states.",
    }
    ev = run["events"]
    by_step: dict[int, list[str]] = {}
    for e in ev:
        by_step.setdefault(int(e["step"]), []).append(f"{e['graph']} ({e['i']},{e['j']}) {e['event']}")
    out["topology_events"] = {
        "total_edge_events": len(ev),
        "per_graph": dict(Counter(e["graph"] for e in ev)),
        "distinct_event_times": [{"step": s, "t": s * dt, "events": v} for s, v in sorted(by_step.items())],
    }
    c = np.asarray(cfg["target"]["center"])
    sig = np.asarray(cfg["target"]["Sigma"])
    dx = traj - c
    phi = np.einsum("kni,ij,knj->kn", dx, sig, dx)
    inside = phi <= 1.0
    task = {}
    for agent in range(n):
        idx = np.flatnonzero(inside[:, agent])
        first = int(idx[0]) if idx.size else None
        exits_after = None if first is None else int((~inside[first:, agent]).sum())
        task[agent] = {"first_entry_step": first, "first_entry_t": None if first is None else first * dt,
                       "samples_outside_after_first_entry": exits_after,
                       "inside_at_final_state": bool(inside[-1, agent]),
                       "final_phi": float(phi[-1, agent])}
    out["task"] = {
        "per_agent": task,
        "all_entered": all(v["first_entry_step"] is not None for v in task.values()),
        "all_inside_at_final": all(v["inside_at_final_state"] for v in task.values()),
        "all_stayed_inside_after_entry": all(v["samples_outside_after_first_entry"] == 0 for v in task.values()),
        "all_inside_from_step": int(max(v["first_entry_step"] for v in task.values()))
        if all(v["first_entry_step"] is not None for v in task.values()) else None,
    }
    return out


def margin_analysis(run: dict, cfg: dict) -> dict:
    a = run["arrays"]
    dt = cfg["simulation"]["dt"]
    s = a["s_star"]
    k, i = np.unravel_index(np.nanargmax(s), s.shape)
    worst_step = np.nanmax(s, axis=1)
    # episodes: worst margin per 1-s window
    w = int(round(1.0 / dt))
    n_win = len(worst_step) // w
    win_max = worst_step[: n_win * w].reshape(n_win, w).max(axis=1)
    order = np.argsort(-win_max)[:10]
    episodes = []
    for wi in order:
        seg = s[wi * w:(wi + 1) * w]
        kk, ii = np.unravel_index(np.nanargmax(seg), seg.shape)
        episodes.append({"window_t": [wi * w * dt, (wi + 1) * w * dt], "s_star": float(seg[kk, ii]),
                         "step": int(wi * w + kk), "agent": int(ii)})
    per_agent = {int(j): {"max_s_star": float(np.nanmax(s[:, j])), "step": int(np.nanargmax(s[:, j])),
                          "t": float(np.nanargmax(s[:, j]) * dt)} for j in range(s.shape[1])}
    # topology events: rolling worst margin in +-0.5 s compared with the whole run
    roll = maximum_filter1d(worst_step, size=w + 1, mode="nearest")
    ev_steps = sorted({int(e["step"]) for e in run["events"]})
    near = []
    for es in ev_steps:
        es_c = min(es, len(worst_step) - 1)
        lo, hi = max(0, es_c - 2 * w), min(len(worst_step), es_c + 2 * w)
        near.append({
            "step": es, "t": es * dt,
            "worst_s_star_within_0.5s": float(roll[es_c]),
            "percentile_among_all_steps": float((roll < roll[es_c]).mean() * 100),
            "worst_s_star_within_2s": float(worst_step[lo:hi].max()),
            "s_star_at_event_minus_1s": float(worst_step[max(0, es_c - w)]),
            "s_star_at_event": float(worst_step[es_c]),
            "s_star_at_event_plus_1s": float(worst_step[min(len(worst_step) - 1, es_c + w)]),
        })
    # mismatch vs margin (per agent, Spearman)
    corr = {}
    for j in range(s.shape[1]):
        corr[int(j)] = {
            "spearman_s_star_vs_abs_clf_mismatch": float(spearmanr(s[:, j], np.abs(a["clf_mismatch"][:, j])).statistic),
            "spearman_s_star_vs_abs_conn_mismatch": float(spearmanr(s[:, j], np.abs(a["conn_mismatch"][:, j])).statistic),
        }
    slack = a["slack"]
    active_frac = {g: float(np.mean(slack[:, :, gi] < ACTIVE_TOL)) for gi, g in enumerate(GROUPS)}
    return {
        "tightest": {"s_star": float(s[k, i]), "step": int(k), "t": float(k * dt), "agent": int(i)},
        "s_star_quantiles": {q: float(np.nanquantile(s, q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
        "per_agent_tightest": per_agent,
        "tightest_1s_windows": episodes,
        "near_topology_events": near,
        "mismatch_margin_correlation": corr,
        "fraction_active_at_applied_u": active_frac,
        "max_abs_clf_mismatch": float(np.nanmax(np.abs(a["clf_mismatch"]))),
        "max_abs_conn_mismatch": float(np.nanmax(np.abs(a["conn_mismatch"]))),
    }


def compare_runs(new: dict, prev: dict) -> dict:
    an, ap = new["arrays"], prev["arrays"]
    out = {
        "trajectories_identical": bool(np.array_equal(new["traj"], prev["traj"])),
        "trajectory_shape": list(new["traj"].shape),
        "max_abs_position_diff": float(np.max(np.abs(new["traj"] - prev["traj"]))) if new["traj"].shape == prev["traj"].shape else None,
        "final_positions_new": new["traj"][-1].tolist(),
        "final_positions_previous": prev["traj"][-1].tolist(),
        "arrays_identical": {k: bool(np.array_equal(an[k], ap[k], equal_nan=True))
                             for k in ("s_star", "viol_norm", "viol_raw", "cls", "status", "slack", "lambda2",
                                       "lambda3", "edges", "events", "conn_mismatch", "clf_mismatch", "positions")
                             if k in an and k in ap},
        "events_identical": new["events"] == prev["events"],
        "class_counts_new": new["summary"]["class_counts"],
        "class_counts_previous": prev["summary"]["class_counts"],
        "lambda2_comm_identical": new["traj_json"]["lambda2_comm"] == prev["traj_json"]["lambda2_comm"],
    }
    return out


# ----------------------------------------------------------------------------- figures
def fig_zoom(run: dict, cfg: dict, center_step: int, agent: int, path: Path, half_window_s: float = 3.0) -> None:
    a = run["arrays"]
    dt = cfg["simulation"]["dt"]
    w = int(half_window_s / dt)
    lo, hi = max(0, center_step - w), min(a["s_star"].shape[0], center_step + w)
    t = np.arange(lo, hi) * dt
    ev = sorted({int(e["step"]) for e in run["events"] if lo <= int(e["step"]) < hi})
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
        for es in ev:
            ax.axvline(es * dt, color=INK_2, lw=0.8, ls=":")
    ax = axes[0]
    for j in range(a["s_star"].shape[1]):
        ax.plot(t, a["s_star"][lo:hi, j], color=AGENT_COLORS[j], lw=1.4, label=f"agent {j}",
                ls="--" if j == 1 else "-")
    ax.axvline(center_step * dt, color=CRITICAL, lw=1, ls="--")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="lower right", ncol=5, frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_ylabel("s* (control units)", color=INK_2, fontsize=9)
    ax.set_title("Margin s* per agent (agents 0 and 1 are mirror images and overlap)",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    for gi, g in enumerate(GROUPS):
        y = a["slack"][lo:hi, agent, gi].astype(float)
        if np.isfinite(y).any():
            ax.plot(t, np.clip(y, 1e-7, None), color=SERIES[gi], lw=1.4, label=g)
    ax.set_yscale("log")
    ax.legend(loc="lower left", ncol=4, frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_ylabel("min slack at applied u", color=INK_2, fontsize=9)
    ax.set_title(f"Agent {agent}: tightest slack per group (clipped at 1e-7; CLF always active)",
                 loc="left", fontsize=10, color=INK)
    ax = axes[2]
    ax.plot(t, np.abs(a["clf_mismatch"][lo:hi, agent]), color=SERIES[3], lw=1.4, label="|CLF mismatch term|")
    conn = np.abs(a["conn_mismatch"][lo:hi, agent])
    if np.any(conn > 0):
        ax.plot(t, conn, color=SERIES[2], lw=1.4, label="|connectivity mismatch term|")
    else:
        ax.text(0.01, 0.04, "connectivity mismatch term ≡ 0 in this window", transform=ax.transAxes,
                fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_ylabel("magnitude", color=INK_2, fontsize=9)
    ax.set_title(f"Agent {agent}: mismatch term in the CLF row", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("time [s]  (dotted: topology events; dashed: tightest margin)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_event_aligned(run: dict, cfg: dict, path: Path, half_window_s: float = 2.0) -> None:
    a = run["arrays"]
    dt = cfg["simulation"]["dt"]
    worst = np.nanmax(a["s_star"], axis=1)
    w = int(half_window_s / dt)
    ev = sorted({int(e["step"]) for e in run["events"]})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    stack = []
    for es in ev:
        lo, hi = es - w, es + w
        if lo < 0 or hi > len(worst):
            continue
        seg = worst[lo:hi]
        stack.append(seg)
        ax.plot((np.arange(lo, hi) - es) * dt, seg, color=MUTED, lw=0.9, alpha=0.8)
    if stack:
        ax.plot((np.arange(-w, w)) * dt, np.median(np.vstack(stack), axis=0), color=SERIES[0], lw=2,
                label=f"median over {len(stack)} event times")
    ax.axvline(0, color=INK_2, lw=1, ls="--")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_xlabel("time relative to topology event [s]", color=INK_2, fontsize=9)
    ax.set_ylabel("max over agents of s*", color=INK_2, fontsize=9)
    ax.set_title("Worst local feasibility margin around each topology change (gray: individual events)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_obstacle(run: dict, cfg: dict, safety: dict, path: Path, pad_s: float = 2.0) -> None:
    """Whole penetration episode of the deepest (agent, obstacle) pair, with pad_s on each side."""
    ep = safety["obstacles"]
    traj = run["traj"]
    dt = cfg["simulation"]["dt"]
    obs = cfg["obstacles"]["obstacles"][ep["obstacle"]]
    episode = next(e for e in ep["episodes"] if e["agent"] == ep["agent"] and e["obstacle"] == ep["obstacle"])
    w = int(pad_s / dt)
    lo, hi = max(0, episode["first_step"] - w), min(traj.shape[0], episode["last_step"] + w)
    t = np.arange(lo, hi) * dt
    clear = np.linalg.norm(traj[lo:hi, ep["agent"]] - np.asarray(obs["center"]), axis=1) - obs["radius"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
    ax = axes[0]
    ax.plot(t, clear, color=SERIES[0], lw=1.6)
    ax.axhline(0, color=CRITICAL, lw=1)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.axvspan(episode["t_first"], episode["t_last"], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_ylabel("‖x − s‖ − r", color=INK_2, fontsize=9)
    ax.set_title(f"Agent {ep['agent']} clearance from obstacle {ep['obstacle']} (negative = inside)",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    qk_lo, qk_hi = lo, min(hi, run["arrays"]["slack"].shape[0])
    ax.plot(np.arange(qk_lo, qk_hi) * dt,
            np.clip(run["arrays"]["slack"][qk_lo:qk_hi, ep["agent"], GROUPS.index("obstacle")].astype(float), -1e-4, None),
            color=SERIES[1], lw=1.4)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_ylabel("min obstacle-row slack at applied u", color=INK_2, fontsize=9)
    ax.set_xlabel("time [s]", color=INK_2, fontsize=9)
    ax.set_title("Obstacle CBF row slack at the applied input (normalized; < 0 = violated by OSQP tolerance)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="outputs/analysis")
    args = ap.parse_args()

    out = make_unique_run_dir(args.out_root, Path(args.new).name + "_analysis")
    cfg = yaml.safe_load(Path(args.config).read_text())
    new, prev = load_run(Path(args.new)), load_run(Path(args.previous))

    results = {
        "inputs": {"new": args.new, "previous": args.previous, "config": args.config,
                   "new_meta": new["meta"], "previous_meta": prev["meta"],
                   "new_completion": {k: new["summary"].get(k) for k in (
                       "verdict", "horizon_completed", "termination_reason", "steps_configured", "steps_completed")}},
        "qp_logs": stream_qp_logs(new, prev),
        "margins": margin_analysis(new, cfg),
        "safety_connectivity_task": safety_connectivity_task(new, cfg),
        "comparison": compare_runs(new, prev),
        "new_summary_tolerance": new["summary"].get("tolerance"),
        "new_summary_verification": {k: v for k, v in new["summary"].get("verification", {}).items()},
    }
    (out / "analysis.json").write_text(json.dumps(results, indent=2, default=str))

    # figures (raw run directories untouched: the overview plot is made from a copy)
    plot_in = out / "overview_input"
    plot_in.mkdir()
    for name in ("arrays.npz", "summary.json"):
        shutil.copy2(new["dir"] / name, plot_in / name)
    png = plot_run(plot_in, cfg["fast_dynamics"]["eta"])
    shutil.move(str(png), out / "fig1_feasibility_overview.png")
    shutil.rmtree(plot_in)
    tight = results["margins"]["tightest"]
    fig_zoom(new, cfg, tight["step"], tight["agent"], out / "fig2_tightest_margin_zoom.png")
    fig_event_aligned(new, cfg, out / "fig3_margin_around_topology_events.png")
    fig_obstacle(new, cfg, results["safety_connectivity_task"], out / "fig4_obstacle_clearance.png")
    print(out)


if __name__ == "__main__":
    main()

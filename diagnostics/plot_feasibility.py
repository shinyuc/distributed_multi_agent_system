"""Plot feasibility diagnostics written by diagnostics.run_feasibility.

Usage: python -m diagnostics.plot_feasibility outputs/feasibility/<run> [--eta 0.1]

Panels (shared time axis):
  1. lambda_2 (runtime, weighted alpha-graph) with the eta threshold
  2. worst local feasibility margin min_i s*_i(t); non-solved QPs marked
  3. s*_i(t) per agent (diverging: blue = feasible margin, red = infeasible)
  4. tightest normalized slack per constraint group at the applied input
  5. magnitude of the mismatch terms in the connectivity / CLF rows
  6. topology events (edge appears/disappears) in the comm, alpha and CLF graphs
Long runs are binned to ~3000 columns keeping the worst value per bin.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRITICAL = "#d03b3b"
WARNING = "#fab219"
DIVERGING = LinearSegmentedColormap.from_list(
    "margin", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c9c8", "#e34948", "#8f1f1f"]
)
GRAPH_NAMES = {"comm": "comm (d < d_c)", "alpha": "α (d < d_c−ε_c)", "clf": "CLF coupling"}


def _bin(t: np.ndarray, y: np.ndarray, n_bins: int, reduce) -> tuple[np.ndarray, np.ndarray]:
    if len(t) <= n_bins:
        return t, y
    edges = np.linspace(0, len(t), n_bins + 1).astype(int)
    tb = np.array([t[a] for a, b in zip(edges[:-1], edges[1:]) if b > a])
    with np.errstate(all="ignore"):
        yb = np.array([reduce(y[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:]) if b > a])
    return tb, yb


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_run(run_dir: Path, eta: float | None, n_bins: int = 3000) -> Path:
    d = np.load(run_dir / "arrays.npz")
    summary = json.loads((run_dir / "summary.json").read_text())
    if eta is None:
        eta = summary.get("run_meta", {}).get("eta")
    dt = float(d["dt"])
    n = d["s_star"].shape[0]
    t = np.arange(n) * dt
    agents = d["agent_ids"]
    classes = list(d["classes"])
    s = d["s_star"].astype(float)
    s_plot = np.where(np.isinf(s), 1e3, s)

    fig, axes = plt.subplots(
        6, 1, figsize=(10, 12.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3, 1.1, 1.3, 1.1, 0.6]},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    # 1. lambda_2
    ax = axes[0]
    tb, lb = _bin(t, d["lambda2"], n_bins, np.nanmin)
    ax.plot(tb, lb, color=SERIES[0], lw=1.6, marker="o" if len(tb) < 30 else None, ms=4)
    if eta is not None:
        ax.axhline(eta, color=MUTED, lw=1, ls="--")
        ax.annotate(f"η = {eta:g}", (t[-1] if n > 1 else 0, eta), xytext=(-4, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylabel("λ₂ (weighted)", color=INK_2, fontsize=9)
    ax.set_title("Algebraic connectivity used in the connectivity rows", loc="left", fontsize=10, color=INK)

    # 2. worst margin
    ax = axes[1]
    worst = np.nanmax(s_plot, axis=1)
    tb, wb = _bin(t, worst, n_bins, np.nanmax)
    ax.plot(tb, wb, color=SERIES[0], lw=1.6, marker="o" if len(tb) < 30 else None, ms=4)
    ax.axhline(0.0, color=INK_2, lw=1)
    ax.set_yscale("symlog", linthresh=1e-3)
    bad = np.argwhere(d["cls"] >= classes.index("borderline"))
    if len(bad):
        for code, color, name in ((classes.index("infeasible"), CRITICAL, "infeasible"),
                                  (classes.index("numerical_failure"), WARNING, "numerical failure"),
                                  (classes.index("borderline"), WARNING, "borderline")):
            sel = bad[d["cls"][bad[:, 0], bad[:, 1]] == code]
            if len(sel):
                ax.scatter(t[sel[:, 0]], s_plot[sel[:, 0], sel[:, 1]], s=36, color=color,
                           edgecolor=SURFACE, linewidth=1.5, zorder=5, label=f"{name} ({len(sel)})")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=3, labelcolor=INK_2, borderaxespad=0.2)
    ax.set_ylabel("max over agents of s*", color=INK_2, fontsize=9)
    ax.set_title("Worst local feasibility margin s* (control units; > 0 = no admissible u)",
                 loc="left", fontsize=10, color=INK)

    # 3. per-agent heatmap
    ax = axes[2]
    tb, sb = _bin(t, s_plot, n_bins, np.nanmax)
    span = max(float(np.nanmax(np.abs(sb))) if np.isfinite(sb).any() else 1.0, 1e-2)
    norm = SymLogNorm(linthresh=1e-3, vmin=-span, vmax=span)
    t_edges = np.append(tb, tb[-1] + (tb[1] - tb[0] if len(tb) > 1 else dt))
    mesh = ax.pcolormesh(t_edges, np.arange(len(agents) + 1) - 0.5, sb.T, cmap=DIVERGING, norm=norm,
                         shading="flat")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([f"agent {a}" for a in agents])
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.set_label("s*", fontsize=8, color=INK_2)
    ax.set_title("Per-agent margin s*_i(t)  (blue: feasible margin, red: infeasible)",
                 loc="left", fontsize=10, color=INK)

    # 4. group slacks
    ax = axes[3]
    groups = list(d["groups"])
    slack = d["slack"].astype(float)
    for g_idx, g in enumerate(groups):
        y = np.nanmin(slack[:, :, g_idx], axis=1)
        if not np.isfinite(y).any():
            continue
        tb, yb = _bin(t, np.clip(y, 1e-7, None), n_bins, np.nanmin)
        ax.plot(tb, yb, color=SERIES[g_idx], lw=1.4, label=g, marker="o" if len(tb) < 30 else None, ms=4)
    ax.set_yscale("log")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=4, labelcolor=INK_2, borderaxespad=0.2)
    ax.set_ylabel("min slack (clipped at 1e-7)", color=INK_2, fontsize=9)
    ax.set_title("Tightest slack per group at applied u",
                 loc="left", fontsize=10, color=INK)

    # 5. mismatch magnitude
    ax = axes[4]
    for k, (key, name) in enumerate((("conn_mismatch", "connectivity"), ("clf_mismatch", "CLF"))):
        y = np.nanmax(np.abs(d[key].astype(float)), axis=1)
        tb, yb = _bin(t, np.clip(y, 1e-12, None), n_bins, np.nanmax)
        ax.plot(tb, yb, color=SERIES[[2, 3][k]], lw=1.4, label=name, marker="o" if len(tb) < 30 else None, ms=4)
    ax.set_yscale("log")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=2, labelcolor=INK_2, borderaxespad=0.2)
    ax.set_ylabel("max_i |mismatch term|", color=INK_2, fontsize=9)
    ax.set_title("Mismatch term Σ_j h_ij(z_i − z_j) in connectivity / CLF rows",
                 loc="left", fontsize=10, color=INK)

    # 6. topology events
    ax = axes[5]
    graphs = list(d["graphs"])
    events = d["events"]
    for g_idx, g in enumerate(graphs):
        steps = np.flatnonzero(events[:, g_idx] > 0)
        ax.vlines(t[steps], g_idx - 0.35, g_idx + 0.35, color=INK_2, lw=0.8)
        ax.annotate(f"{int(events[:, g_idx].sum())}", (1.0, g_idx), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_yticks(range(len(graphs)))
    ax.set_yticklabels([GRAPH_NAMES.get(g, g) for g in graphs])
    ax.set_ylim(-0.6, len(graphs) - 0.4)
    ax.grid(False, axis="y")
    ax.set_title("Topology events (edge appears / disappears); totals at right", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("time [s]", color=INK_2, fontsize=9)

    meta = summary.get("run_meta", {})
    cc = summary["class_counts"]
    fig.suptitle(
        f"{meta.get('config_path', run_dir.name)} — {summary['qp_solves']} local QPs over "
        f"{summary['steps_recorded']} updates: solved {cc['solved']}, inaccurate {cc['solved_inaccurate']}, "
        f"infeasible {cc['infeasible']}, numerical {cc['numerical_failure']}, borderline {cc['borderline']}",
        x=0.01, ha="left", fontsize=10, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = run_dir / "feasibility_diagnostics.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--eta", type=float, default=None)
    args = parser.parse_args()
    for rd in args.run_dirs:
        print(plot_run(Path(rd), args.eta))


if __name__ == "__main__":
    main()

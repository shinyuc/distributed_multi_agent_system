"""Tabulate summary.json files from several feasibility runs (Markdown to stdout).

Usage: python -m diagnostics.summarize_runs outputs/feasibility/*/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _fmt(v, spec=".3g") -> str:
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    return format(v, spec)


def main() -> None:
    header = ("run", "updates", "QP solves", "solved", "inaccurate", "infeasible", "numerical", "borderline",
              "max viol (norm.)", "max viol (raw)", "tightest margin (solved)", "first non-solved", "topology events")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for rd in sys.argv[1:]:
        p = Path(rd) / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        cc = s["class_counts"]
        mv = s["max_violation_of_solved_qps"]
        a = np.load(Path(rd) / "arrays.npz")
        solved = (a["cls"] >= 0) & (a["cls"] <= 1)  # solved / solved_inaccurate
        tightest = float(np.max(a["s_star"][solved])) if solved.any() else None
        fb = s.get("first_non_solved")
        first = "–" if not fb else f"step {fb['step']} (t={fb['t']:.4g}s), agent {fb['agent']}, {fb['class']}"
        ev = s.get("topology_events", {})
        print("| " + " | ".join([
            Path(rd).name, str(s["steps_recorded"]), str(s["qp_solves"]),
            str(cc["solved"]), str(cc["solved_inaccurate"]), str(cc["infeasible"]),
            str(cc["numerical_failure"]), str(cc["borderline"]),
            _fmt(mv.get("normalized")), _fmt(mv.get("raw")), _fmt(tightest), first,
            "/".join(str(ev.get(g, 0)) for g in ("comm", "alpha", "clf")),
        ]) + " |")


if __name__ == "__main__":
    main()

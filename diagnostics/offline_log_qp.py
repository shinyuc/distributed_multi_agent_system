"""Offline feasibility analysis of a local QP printed in an old simulator log.

Parses the first block of ``[QP INFEASIBLE ROW] label=..., A=[...], b=...`` lines
(scaled rows, as printed by Simulator._print_infeasibility_diagnostics) and runs the
same margin LP / group-subset analysis as the online recorder. Positive row scaling
does not change the normalized rows, so the scaled rows are sufficient. The actuator
bound is read from the actuator rows (a = s^-1 e_x, b = u_max s^-1).

Usage: python -m diagnostics.offline_log_qp logs/run_5_agents.log
"""

from __future__ import annotations

import ast
import json
import re
import sys

import numpy as np

from diagnostics.feasibility import GROUPS, Tolerances, group_subset_analysis, local_feasibility, row_group

ROW_RE = re.compile(r"\[QP INFEASIBLE ROW\] label=(.*?), A=(\[.*?\]), b=(\S+)")


def parse(path: str) -> dict:
    header: dict = {}
    rows: list[tuple[str, list[float], float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[QP INFEASIBLE] first infeasible") and "first" not in header:
                header["first"] = line.strip()
            elif line.startswith("[QP INFEASIBLE] solver_status") and "status" not in header:
                header["status"] = line.strip()
            elif line.startswith("[QP INFEASIBLE] lambda2") and "lambda2" not in header:
                header["lambda2"] = line.strip()
            m = ROW_RE.search(line)
            if m:
                label = m.group(1)
                if rows and label == rows[0][0]:
                    break  # second printout of the same QP
                rows.append((label, ast.literal_eval(m.group(2)), float(m.group(3))))
            elif rows and not line.startswith("[QP INFEASIBLE"):
                break
    return {"header": header, "rows": rows}


def main() -> None:
    data = parse(sys.argv[1])
    labels = [r[0] for r in data["rows"]]
    A = np.asarray([r[1] for r in data["rows"]], dtype=float)
    b = np.asarray([r[2] for r in data["rows"]], dtype=float)
    box = [k for k, l in enumerate(labels) if l.startswith("actuator(ux<=umax")][0]
    u_max = float(b[box] / A[box, 0])
    tol = Tolerances()
    full = local_feasibility(A, b, labels, u_max, tol)
    present = sorted({row_group(l) for l in labels} & set(GROUPS), key=GROUPS.index)
    subsets = group_subset_analysis(
        lambda g: local_feasibility(A, b, labels, u_max, tol, groups=g), present, tol
    )
    print(json.dumps({**data["header"], "u_max": u_max, "n_rows": len(labels),
                      "s_star": full["s_star"], "binding_rows": full["binding"],
                      **subsets}, indent=2))


if __name__ == "__main__":
    main()

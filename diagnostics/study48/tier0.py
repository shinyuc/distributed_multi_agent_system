"""Tier-0 static screening at x_0 for the generated study scenarios (no simulation).

    python -m diagnostics.study48.tier0 --study STUDY_DIR [--ids A01,B01,...] [--pilots] [--tag TAG]

Per scenario: physical validity (with the legacy / initial-connectivity flags); Levels 1-3 with confirmed F/I/U
verdicts (diagnostics.static_feasibility) at the actual initial mismatch (zero, or the generated warm start);
Level 4 where Level 2 is F: the regularized KKT point, the M2 assessment of rho (certified lower bound, dual upper
bound, status; the true rho is >= 0; an upper bound <= 1e-9 means numerically zero, not a proof of rho = 0), the initial full-state error, C0-C2 and the frozen-state sufficient
conditions (continuous Thm 1: |chi_0 - chi*| < rho, with rho the certified lower bound; discrete Remark 6.2 via the
certifier on [x_0, x_0]) under the spectral norm bound (primary) and the Frobenius norm (comparison).
Where the componentwise Level 2 disagrees with a solver-based check, diagnostics.study48.disagreement is run.
A step-0 preflight solves every agent's step-0 local QP exactly as the simulator will (same builder, same OSQP
call) and predicts whether a closed-loop run stops at step 0 without applying a control.
Construction failures (geometry rules) are reported separately from QP infeasibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from config import load_config
from controller.local_qp import solve_local_qp_detailed
from diagnostics.certificate import CertModel, CertTolerances, certify, solve_kkt, zero_fast_state
from diagnostics.feasibility import SOLVER_SUCCESS, Tolerances, local_feasibility, lp_verdict
from diagnostics.static_feasibility import graphs, static_check

NORMS = ("spectral", "frobenius")      # primary first


def _initial_state(model, entry, study: Path):
    if entry.get("warm_start"):
        return pickle.load(open(study / entry["warm_start"]["file"], "rb"))["warm"], "warm"
    return zero_fast_state(model), "zero"


def step0_preflight(model, rows, chi0) -> dict:
    """Every agent's step-0 local QP as the simulator builds and solves it (static; nothing is applied)."""
    tol = Tolerances()
    u_max = model.cfg.controller.u_max
    agents = []
    for i in range(rows.N):
        qp = rows.local_qp(i, chi0)
        lp = local_feasibility(np.asarray(qp.A_raw), np.asarray(qp.b_raw), qp.row_labels, u_max, tol)
        res = solve_local_qp_detailed(qp)
        agents.append({"agent": i, "osqp_status": res.status, "osqp_success": bool(res.success),
                       "lp_s_star": float(lp["s_star"]), "lp_verdict": lp_verdict(lp["s_star"], tol)})
    stop = [a["agent"] for a in agents if not a["osqp_success"]]
    unsafe = [a["agent"] for a in agents if a["osqp_success"] and a["lp_verdict"] == "infeasible"]
    border = [a["agent"] for a in agents if a["lp_verdict"] == "borderline"]
    if stop:
        pred = "terminates_at_step0"           # simulator: every QP solved, then stop; no control applied
    elif unsafe:
        pred = "UNSAFE_solver_success_on_LP_infeasible_QP"
    else:
        pred = "proceeds"
    return {"agents": agents, "prediction": pred, "osqp_failed_agents": stop, "inconsistent_agents": unsafe,
            "borderline_agents": border,
            "solver_failure_on_lp_feasible_qp": [a["agent"] for a in agents
                                                 if not a["osqp_success"] and a["lp_verdict"] == "feasible"],
            "note": "matches the Level-3 verdicts" if all(
                (a["lp_verdict"] == "infeasible") == (not a["osqp_success"]) for a in agents)
            else "OSQP status and LP verdict differ for some agent"}


def _frozen(model, x0, chi0, norm):
    cert = certify(model, [x0, x0], chi0, CertTolerances(norm=norm), check_model_at=(0,))
    s = cert.steps
    out = {"results": {v: r.as_dict() for v, r in cert.results.items()}}
    if s.get("E"):
        rho, e0 = s["rho"][0], cert.initial.get("e0")
        out.update(a=s["a"][0], a_lo=s["a_lo"][0], r_a=model.r * s["a"][0], kappa_p=s["kappa_p"][0],
                   c0_status=s["c0_status"][0], c=s["c"][0], E0=s["E"][0], E_adm=s["E_adm"][0],
                   thm1_continuous=bool(e0 is not None and e0 < rho))
    return cert, out


def screen_scenario(study: Path, entry: dict) -> dict:
    t0 = time.perf_counter()
    cfg = load_config(str(study / entry["config"]))
    model = CertModel(cfg)
    x0 = [list(map(float, p)) for p in cfg.agents.initial_positions]
    chi0, init = _initial_state(model, entry, study)
    rows = model.rows(x0)
    p0, y0 = rows.to_vec(chi0)
    z0 = p0[rows.nw:]
    g = graphs(rows)
    rep = static_check(model, x0, z0)
    pv = entry["physical_validity"]
    out = {"id": entry["id"], "group": entry["group"], "pilot": entry["pilot"], "num_agents": rows.N, "init": init,
           "legacy_case": bool(entry.get("legacy", False)),
           "initial_connectivity_ok": bool(pv["rules"]["lambda2_ge_eta"]),
           "construction_ok": entry["construction_ok"], "construction_rule_failures": entry["construction_rule_failures"],
           "physical": {k: pv[k] for k in ("min_pair_distance", "min_obstacle_clearance", "lambda2",
                                           "lambda2_minus_eta", "cutoff_margin", "q", "sigma")},
           "lambda2_model": rows.lam2, "graphs": g, "levels": rep.as_dict(), "category": rep.category,
           "consistency": rep.consistency, "step0_preflight": step0_preflight(model, rows, chi0)}
    if any(c.startswith("DISAGREEMENT") for c in rep.consistency):
        from diagnostics.study48.disagreement import diagnose
        out["disagreement"] = diagnose(model, x0)
    lv = {}
    if rep.level2.verdict == "F":
        kkt = solve_kkt(rows, CertTolerances())
        lv["kkt_status"] = kkt.status
        lv["kkt_note"] = kkt.note
        if kkt.status in ("L2_INFEASIBLE", "SOLVER") and (rep.level2_lifted.get("verified")
                                                          or out.get("disagreement", {}).get("lifted", {}).get("exact_verified")):
            lv["kkt_interpretation"] = ("KKT solver failure: Level 2 is F with a verified explicit allocation, and the "
                                        "regularized QP is strictly convex, so the KKT point exists")
        if kkt.status == "ok":
            lv.update(active=[rows.labels[i] for i in np.flatnonzero(kkt.active)],
                      degenerate=[rows.labels[i] for i in np.flatnonzero(rows.deg)], licq_rank=kkt.rank, m=kkt.m,
                      y_min=kkt.y_min, slack_min=kkt.slack_min_norm, z_star_norm=float(np.linalg.norm(kkt.p[rows.nw:])),
                      y_star_norm=float(np.linalg.norm(kkt.y_active)) if kkt.m else 0.0)
            frozen = {}
            for norm in NORMS:
                cert, fz = _frozen(model, x0, chi0, norm)
                frozen[norm] = fz
                if norm == NORMS[0]:
                    s = cert.steps
                    lv["model_check"] = cert.initial.get("model_check")
                    if s.get("rho"):
                        e0 = cert.initial.get("e0")
                        lv.update(rho_lower=s["rho"][0], rho_candidate=s["rho_candidate"][0],
                                  rho_upper=s["rho_upper"][0], rho_status=s["rho_status"][0],
                                  rho_structural_zero_rows=s["rho_structural_zero_rows"][0],
                                  rho_agents_candidate=s["rho_agents"][0], e0=e0, sqrtU0=cert.initial.get("sqrtU0"),
                                  C1=s["C1"][0], C1_row=s["C1_label"][0], C2=s["C2"][0], C2_row=s["C2_label"][0],
                                  z0_in_Zfeas=rep.level3_verdict == "F")
            lv["frozen"] = frozen
    else:
        lv["skipped"] = f"Level 2 verdict {rep.level2.verdict}: no KKT point computed"
    out["level4"] = lv
    out["wall_time_s"] = time.perf_counter() - t0
    return out


def _fz(l4, norm, key, default=""):
    return l4.get("frozen", {}).get(norm, {}).get(key, default)


def _cert(l4, norm):
    r = l4.get("frozen", {}).get(norm, {}).get("results", {}).get("non_increase", {})
    return r.get("status", ""), r.get("reason") or ""


def _row(r):
    L = r["levels"]
    l4 = r["level4"]
    ph = r["physical"]
    row = {"id": r["id"], "group": r["group"], "pilot": r["pilot"], "N": r["num_agents"], "init": r["init"],
           "legacy": r["legacy_case"], "initial_connectivity_ok": r["initial_connectivity_ok"],
           "construction_ok": r["construction_ok"], "rule_failures": ";".join(r["construction_rule_failures"]),
           "min_pair": round(ph["min_pair_distance"], 4),
           "min_clearance": round(ph["min_obstacle_clearance"], 4) if math.isfinite(ph["min_obstacle_clearance"]) else "inf",
           "lambda2_minus_eta": round(ph["lambda2_minus_eta"], 4),
           "L1": L["level1"]["verdict"], "L1_margin": L["level1"]["lp_margin"],
           "L2": L["level2"]["verdict"], "L2_margin": L["level2"]["lp_margin"],
           "L2_aug": L["level2_augmented"]["verdict"], "L2_repo": L["level2_repository"].get("verdict"),
           "lifted_ok": L["level2_lifted"].get("verified"),
           "L3": L["level3_verdict"], "L3_agents": "".join(v["verdict"] for v in L["level3"]),
           "category": r["category"], "consistency": ";".join(r["consistency"]),
           "step0": r["step0_preflight"]["prediction"],
           "kkt": l4.get("kkt_status", ""), "rho_lower": l4.get("rho_lower", ""), "rho_upper": l4.get("rho_upper", ""),
           "rho_status": l4.get("rho_status", ""), "rho_candidate": l4.get("rho_candidate", ""),
           "e0": l4.get("e0", ""), "C1": l4.get("C1", ""), "C2": l4.get("C2", "")}
    for norm in NORMS:
        st, rs = _cert(l4, norm)
        row.update({f"{norm}_a": _fz(l4, norm, "a"), f"{norm}_r_a": _fz(l4, norm, "r_a"),
                    f"{norm}_kappa_p": _fz(l4, norm, "kappa_p"), f"{norm}_C0": _fz(l4, norm, "c0_status"),
                    f"{norm}_E0": _fz(l4, norm, "E0"), f"{norm}_thm1": _fz(l4, norm, "thm1_continuous"),
                    f"{norm}_cert": st, f"{norm}_reason": rs})
    row["time_s"] = round(r["wall_time_s"], 3)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier-0 static screening of the study scenarios")
    ap.add_argument("--study", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--pilots", action="store_true")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--outdir", default="tier0")
    a = ap.parse_args(argv)
    study = Path(a.study)
    man = json.load(open(study / "manifest.json"))["scenarios"]
    sel = [e for e in man if (not a.ids or e["id"] in a.ids.split(",")) and (not a.pilots or e["pilot"])]
    outdir = study / a.outdir
    outdir.mkdir(exist_ok=True)
    rows = []
    for e in sel:
        r = screen_scenario(study, e)
        (outdir / f"{e['id']}.json").write_text(json.dumps(r, indent=2, default=str))
        rows.append(_row(r))
        x = rows[-1]
        print(f"[tier0] {e['id']}: {x['category']} L1={x['L1']} L2={x['L2']} L3={x['L3']} step0={x['step0']} "
              f"rho={x['rho_status']} spectral={x['spectral_cert']}/{x['spectral_reason']} "
              f"frobenius={x['frobenius_cert']}/{x['frobenius_reason']} ({x['time_s']} s)", file=sys.stderr)
    with open(outdir / f"summary_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (outdir / f"summary_{a.tag}.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

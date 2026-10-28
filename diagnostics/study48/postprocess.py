"""Automated post-processing of the 48-scenario dataset (experiments/feasibility_study/study48/POSTPROCESSING_PLAN.md).

    python -m diagnostics.study48.postprocess --dataset DATASET_ROOT --out OUT_DIR [--workers 8] [--ids A01,...]
                                              [--sample-every 10] [--tail 50] [--cert-max-states K] [--no-certificates]

DATASET_ROOT holds ``scenarios/<ID>`` (symlinks to the batch scenario directories: ``scenario.yaml`` and ``run/``);
``superseded/`` and other entries are ignored. Nothing in the dataset is written: every file is opened read-only, and
OUT_DIR must be new or empty and outside the dataset. No closed-loop simulation is run; every quantity is computed
statically from the recorded outputs with the validated checkers:

* P0 integrity: ran configuration vs repository configuration vs manifest hash, warm-start hash, strict guard,
  snapshot setting, commit, modified tracked files, initial fast state;
* P1 outcome: batch-style summary row, termination kind, failing agents, QP class counts, safety / connectivity
  minima, recorded mismatch maxima;
* P2 key states (x_0, the termination step, the last QP state): Levels 1-3 with confirmed F/I/U verdicts and their
  witnesses / certificates (diagnostics.static_feasibility), recorder vs static per-agent agreement, the B04/B12-type
  disagreement diagnosis, and an interpretation of the termination;
* P4 sampled trajectory: Levels 1-3 at the actual mismatch and Level 3 at zero mismatch every ``sample_every`` steps
  and on the last ``tail`` QP states, where a fast-state snapshot exists (no interpolation, no replay);
* P5 certificates: R6.1 Thm 3 along the recorded positions under the spectral (primary) and Frobenius norms, rho
  bounds and status, and observed errors |chi_k - chi*_k| at sampled states vs the certified radius;
* P6 Tier-0 comparison: predicted step-0 behaviour, category at x_0 and frozen certificate vs observed.

The fast state at step k (the one the step-k QPs use) comes from a recorded failure file of step k or from the
snapshot ``snaps[k / every]`` when k is a multiple of the snapshot interval; otherwise the state is reported as
unavailable and every result depending on it is unresolved.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = REPO / "experiments/feasibility_study"
NORMS = ("spectral", "frobenius")


@dataclass(frozen=True)
class Options:
    sample_every: int = 10
    tail: int = 50
    cert_max_states: int | None = None
    certificates: bool = True
    observed_errors: bool = True


# ----------------------------------------------------------------------------------------------- inputs
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_inputs(inputs: Path = DEFAULT_INPUTS) -> dict:
    """Manifest entries, Tier-0 rows, repository configuration and warm-start paths per scenario id."""
    inputs = Path(inputs)
    man = json.loads((inputs / "study48/manifest.json").read_text())
    t0 = {r["id"]: r for r in json.loads((inputs / "study48/tier0_summary_all.json").read_text())}
    out = {}
    for e in man["scenarios"]:
        sub = "pilot16" if e["pilot"] else "remaining32"
        warm = inputs / sub / e["warm_start"]["file"] if e.get("warm_start") else None
        out[e["id"]] = {"entry": e, "tier0": t0.get(e["id"]), "config": inputs / sub / e["config"], "warm": warm}
    return {"scenarios": out, "manifest_sha256": _sha(inputs / "study48/manifest.json"), "inputs": str(inputs),
            "pairs": [(e["id"], o, k) for e in man["scenarios"] for o, k in e["pairs"]]}


def resolve_dataset(path: Path) -> Path:
    path = Path(path)
    return path / "scenarios" if (path / "scenarios").is_dir() else path


# ----------------------------------------------------------------------------------------------- run data
class RunData:
    """Read-only access to one scenario directory (``scenario.yaml`` + ``run/``)."""

    def __init__(self, sdir: Path):
        self.sdir = Path(sdir)
        self.run = self.sdir / "run"
        self.summary = self._json("summary.json")
        self.run_meta = self._json("run_meta.json")
        self.pilot_runner = self._json("pilot_runner.json")
        self.warm_start = self._json("warm_start.json")
        tj = self._json("trajectories.json") or {}
        self.positions = tj.get("trajectories") or []
        self.every, self.snaps = 0, None
        if (self.run / "fast_snapshots.pkl").exists():
            with open(self.run / "fast_snapshots.pkl", "rb") as f:
                d = pickle.load(f)
            self.every, self.snaps = int(d["every"]), d["snaps"]
        self.failures = []
        if (self.run / "failures").is_dir():
            for p in sorted((self.run / "failures").glob("step*.json")):
                with open(p) as f:
                    d = json.load(f)
                d["_file"] = p.name
                self.failures.append(d)

    def _json(self, name):
        p = self.run / name
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def qp_rows(self, steps: set[int]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        p = self.run / "qp_log.csv.gz"
        if not p.exists():
            return out
        with gzip.open(p, "rt", newline="") as f:
            for r in csv.DictReader(f):
                k = int(r["step"])
                if k in steps:
                    out.setdefault(k, []).append(r)
        return out

    def snapshot(self, k: int):
        if self.snaps is None or self.every <= 0 or k % self.every:
            return None
        j = k // self.every
        return self.snaps[j] if j < len(self.snaps) else None

    def num_qp_states(self) -> int:
        """QP states x_0..x_{K-1} (diagnostics.certify_run.default_num_states)."""
        if not self.summary:
            return 0
        k = int(self.summary.get("steps_completed") or 0)
        return k if self.summary.get("termination_reason") == "completed" else min(k + 1, len(self.positions))


def _z_from_failure(rows, fs_json) -> np.ndarray:
    z = np.zeros(rows.n - rows.nw)
    for i in range(rows.N):
        fs = fs_json[str(i)]
        for j, v in fs["z_pair_self"].items():
            z[rows.izp[(i, int(j))] - rows.nw] = v
        z[rows.izc[i] - rows.nw] = fs["z_c"]
        z[rows.izl[i] - rows.nw] = fs["z_clf"]
    return z


def state_at(run: RunData, model, k: int) -> dict:
    """Positions and mismatch z used by the step-k QPs, with their source; None if unavailable."""
    out = {"k": k, "source": None, "positions": None, "z": None, "fast_state": None, "notes": []}
    snap = run.snapshot(k)
    fails = [f for f in run.failures if f.get("step") == k]
    pos_traj = run.positions[k] if k < len(run.positions) else None
    if fails:
        f = fails[0]
        pos = [f["positions"][str(i)] for i in range(len(f["positions"]))]
        rows = model.rows(pos)
        z = _z_from_failure(rows, f["fast_states"])
        out.update(source=f"failure:{f['_file']}", positions=pos, z=z)
        for g in fails[1:]:
            if _z_from_failure(rows, g["fast_states"]).tolist() != z.tolist():
                out["notes"].append(f"failure files of step {k} disagree on the fast state ({g['_file']})")
        if pos_traj is not None and not np.allclose(pos_traj, pos, rtol=0, atol=0):
            out["notes"].append("failure-file positions differ from trajectories[k]")
        if snap is not None:
            zs = rows.to_vec(snap)[0][rows.nw:]
            out["notes"].append("snapshot agrees with failure file" if zs.tolist() == z.tolist()
                                else "SNAPSHOT DIFFERS FROM FAILURE FILE")
            out["fast_state"] = snap
        return out
    if snap is not None and pos_traj is not None:
        rows = model.rows(pos_traj)
        out.update(source=f"snapshot[{k // run.every}] (every {run.every})", positions=pos_traj,
                   z=rows.to_vec(snap)[0][rows.nw:], fast_state=snap)
        return out
    out["notes"].append("no failure file for this step and no snapshot at this step "
                        f"(snapshot interval {run.every or 'none'}): state unavailable")
    return out


# ----------------------------------------------------------------------------------------------- classification
def _strip(v: dict) -> dict:
    return {k: x for k, x in v.items() if k != "tolerances"}


def classify_state(model, st: dict, repository_check: bool = True, keep_allocation: bool = False) -> dict:
    from diagnostics.static_feasibility import graphs, static_check
    rep = static_check(model, st["positions"], st["z"], repository_check=repository_check)
    d = rep.as_dict()
    lifted = dict(d["level2_lifted"])
    if not keep_allocation:
        lifted.pop("u", None)
        lifted.pop("z", None)
    out = {"category": rep.category, "consistency": rep.consistency,
           "L1": _strip(d["level1"]), "L2": _strip(d["level2"]), "L2_augmented": _strip(d["level2_augmented"]),
           "L2_repository": d["level2_repository"], "L2_lifted": lifted,
           "L3": [_strip(v) for v in d["level3"]], "L3_verdict": rep.level3_verdict,
           "L3_agents": "".join(v["verdict"] for v in d["level3"]),
           "graphs": graphs(model.rows(st["positions"])), "z_inf_norm": float(np.max(np.abs(st["z"]))) if st["z"].size else 0.0}
    if any(c.startswith("DISAGREEMENT") for c in rep.consistency):
        from diagnostics.study48.disagreement import diagnose
        try:
            out["disagreement"] = diagnose(model, st["positions"])
        except Exception as exc:                          # noqa: BLE001 - recorded, not fatal
            out["disagreement"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _agreement(cls: dict, qrows: list[dict]) -> dict:
    """Recorder (OSQP status, LP verdict) vs static Level-3 verdict per agent at one step."""
    per, flags = [], []
    for r in sorted(qrows, key=lambda r: int(r["agent"])):
        i = int(r["agent"])
        sv = cls["L3"][i]["verdict"] if i < len(cls["L3"]) else None
        rv, ok = r["lp_verdict"], r["solver_success"] in ("1", "True", "true")
        per.append({"agent": i, "osqp_status": r["solver_status"], "osqp_success": ok, "recorder_class": r["class"],
                    "recorder_lp_verdict": rv, "recorder_s_star": float(r["s_star"]) if r["s_star"] not in ("", "nan") else None,
                    "static_L3": sv, "static_margin": cls["L3"][i]["lp_margin"] if i < len(cls["L3"]) else None})
        if rv == "infeasible" and sv == "F":
            flags.append(f"agent {i}: recorder LP infeasible but static Level 3 F (witness)")
        if rv == "feasible" and sv == "I":
            flags.append(f"agent {i}: recorder LP feasible but static Level 3 I (certificate)")
        if ok and sv == "I":
            flags.append(f"agent {i}: OSQP reported success on a QP the static check certifies infeasible")
        if not ok and sv == "F":
            flags.append(f"agent {i}: OSQP failure ({r['solver_status']}) on a QP with a verified feasible witness")
    return {"agents": per, "flags": flags}


def _termination_kind(summary: dict | None) -> str:
    if not summary:
        return "missing_summary"
    tr = summary.get("termination_reason")
    if tr == "completed":
        return "completed"
    if tr == "exception":
        exc = summary.get("termination_exception") or {}
        return "strict_guard" if exc.get("type") == "StrictFeasibilityGuard" else "exception"
    return tr or "unknown"


def _interpret_termination(kind: str, cls: dict | None, failing: list[int], agreement: dict | None) -> dict:
    """What the termination state shows, keeping certified, solver-reported and unresolved apart."""
    if kind == "completed":
        return {"result": "finite_horizon_completed",
                "statement": "the configured horizon was simulated; this says nothing about later times"}
    if kind not in ("local_qp_failure", "strict_guard"):
        return {"result": "not_a_feasibility_outcome", "statement": f"run ended by {kind}; data problem, not an outcome"}
    if cls is None:
        return {"result": "unresolved_state_unavailable",
                "statement": "the solver-reported failure cannot be checked: the state at the termination step is not recorded"}
    v = {a: cls["L3"][a]["verdict"] for a in failing if a < len(cls["L3"])}
    if failing and all(x == "I" for x in v.values()):
        res = "certified_infeasible"
        what = {"A": "no feasible control exists even for the centralized problem (Level 1 I)",
                "B": "the centralized problem is feasible but no mismatch allocation exists (Level 2 I)",
                "C": "a mismatch allocation exists (Level 2 F) but the actual mismatch makes a local QP infeasible"}
        st = (f"every failing agent's local QP is infeasible, confirmed by a re-verified certificate; category "
              f"{cls['category']}: {what.get(cls['category'], cls['category'])}")
    elif any(x == "F" for x in v.values()):
        res = "solver_failure_on_verified_feasible_qp"
        st = "at least one failing agent's QP has a re-verified feasible witness: a numerical solver failure"
    else:
        res = "unresolved"
        st = "at least one failing agent's static verdict is U (numerically ambiguous)"
    if not failing:
        res, st = "unresolved", "no failing agent recorded"
    certs = {}
    for a in failing:
        if a < len(cls["L3"]):
            c = cls["L3"][a].get("certificate") or {}
            certs[a] = {"verdict": cls["L3"][a]["verdict"], "lp_margin": cls["L3"][a]["lp_margin"],
                        "witness_slack": cls["L3"][a].get("witness_slack"), "certificate_type": c.get("type"),
                        "support": c.get("support") or ([c["label"]] if c.get("label") else None),
                        "phi": c.get("phi"), "err": c.get("err"), "floor": c.get("floor"), "note": cls["L3"][a].get("note")}
    return {"result": res, "statement": st, "failing_agents": failing, "failing_static_L3": v, "certificates": certs,
            "L1_at_termination": cls["L1"]["verdict"], "L2_at_termination": cls["L2"]["verdict"],
            "category_at_termination": cls["category"],
            "agreement_flags": agreement["flags"] if agreement else []}


# ----------------------------------------------------------------------------------------------- certificates
def _certificates(model, run: RunData, chi0, K: int, opts: Options) -> dict:
    from diagnostics.certificate import CertTolerances, certify
    if K < 1 or chi0 is None:
        return {"skipped": "no QP state or no initial fast state"}
    Kc = K if opts.cert_max_states is None else min(K, opts.cert_max_states)
    out = {"num_qp_states": K, "evaluated_states": Kc, "truncated": Kc < K}
    for norm in NORMS:
        t0 = time.perf_counter()
        c = certify(model, run.positions[:Kc], chi0, CertTolerances(norm=norm), check_model_at=(0, Kc - 1))
        s = c.summary()
        st = c.steps
        out[norm] = {"results": {v: {k: r.as_dict()[k] for k in ("status", "reason", "reason_kind", "k_fail",
                                                                  "certified_states", "detail")}
                                 for v, r in c.results.items()},
                     "initial": {k: c.initial.get(k) for k in ("e0", "sqrtU0", "model_check", "prop6_at_fail")},
                     "stats": {k: s.get(k) for k in ("E_max", "E_ni_max", "Gamma_E", "Gamma_E_ni", "min_gap_E",
                                                     "min_gap_E_ni", "rho_min", "a_max", "kappa_p_min", "variation")},
                     "k0": {k: (st[k][0] if st.get(k) else None) for k in ("rho", "rho_candidate", "rho_upper",
                                                                            "rho_status", "rho_structural_zero_rows",
                                                                            "c0_status", "a", "kappa_p", "E", "E_ni",
                                                                            "C1", "C2", "kkt_status")},
                     "rho_status_counts": {x: st["rho_status"].count(x) for x in sorted(set(st.get("rho_status", [])))},
                     "c0_status_counts": {x: st["c0_status"].count(x) for x in sorted(set(st.get("c0_status", [])))},
                     "wall_s": time.perf_counter() - t0}
        if norm == NORMS[0]:
            out["_E_ni"] = st.get("E_ni", [])
    return out


def _observed_errors(model, run: RunData, ks: list[int], E_ni: list[float], certified: int) -> list[dict]:
    """|chi_k - chi*_k| at sampled states with a snapshot, vs the certified non-increase radius E_k."""
    from diagnostics.certificate import CertTolerances, solve_kkt
    out = []
    for k in ks:
        snap = run.snapshot(k)
        if snap is None or k >= len(run.positions):
            continue
        rows = model.rows(run.positions[k])
        kkt = solve_kkt(rows, CertTolerances())
        if kkt.status != "ok":
            out.append({"k": k, "kkt_status": kkt.status})
            continue
        p, y = rows.to_vec(snap)
        ystar = np.zeros(len(rows.b))
        ystar[kkt.active] = kkt.y_active
        ystar[rows.deg] = y[rows.deg]
        e = float(np.linalg.norm(np.r_[p - kkt.p, y - ystar]))
        row = {"k": k, "e": e, "inactive_duals_zero": bool(np.all(y[kkt.inactive] == 0.0))}
        if k < certified and k < len(E_ni):
            row.update(E_ni=E_ni[k], within_certified_radius=bool(e <= E_ni[k] * (1 + 1e-9) + 1e-15))
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------------- one scenario
def _check(name, ok, detail=""):
    return {"check": name, "status": "PASS" if ok is True else ("NOT_RECORDED" if ok is None else "FAIL"),
            "detail": str(detail)}


def _cmp_initial(model, run: RunData, info: dict) -> tuple[object, bool | None]:
    from diagnostics.certificate import zero_fast_state
    if info["warm"] is not None:
        with open(info["warm"], "rb") as f:
            expected = pickle.load(f)["warm"]
    else:
        expected = zero_fast_state(model)
    snap0 = run.snapshot(0)
    if snap0 is None:
        return expected, None
    rows = model.rows(run.positions[0])
    (p1, y1), (p2, y2) = rows.to_vec(snap0), rows.to_vec(expected)
    return snap0, bool(np.array_equal(p1, p2) and np.array_equal(y1, y2))


def analyze_scenario(sdir: Path, sid: str, info: dict | None, opts: Options = Options()) -> dict:
    from config import load_config
    from diagnostics.certificate import CertModel
    from diagnostics.run_batch import summarize_scenario
    t_start = time.perf_counter()
    sdir = Path(sdir)
    e = (info or {}).get("entry") or {}
    out: dict = {"id": sid, "group": e.get("group"), "pilot": e.get("pilot"), "legacy": e.get("legacy"),
                 "initial_connectivity_ok": e.get("initial_connectivity_ok"), "dataset_dir": str(sdir),
                 "resolved_dir": os.path.realpath(sdir), "notes": []}
    run = RunData(sdir)
    cfg_path = sdir / "scenario.yaml"
    with open(cfg_path) as f:
        ran = yaml.safe_load(f)
    # ---- P0 integrity
    chk = []
    if info:
        with open(info["config"]) as f:
            repo_cfg = yaml.safe_load(f)
        chk.append(_check("repository config SHA-256 = manifest", _sha(info["config"]) == e["config_sha256"]))
        chk.append(_check("ran config = repository config (parsed)", ran == repo_cfg))
        if info["warm"] is not None:
            chk.append(_check("warm start SHA-256 = repository file",
                              None if not run.warm_start else run.warm_start["sha256"] == _sha(info["warm"])))
    pr = run.pilot_runner or {}
    chk.append(_check("strict feasibility guard on", None if not pr else pr.get("strict_guard") is True))
    git = pr.get("git") or {}
    chk.append(_check("no modified tracked files", None if "tracked_files_modified" not in git
                      else git["tracked_files_modified"] is False, git.get("commit")))
    chk.append(_check("summary present", run.summary is not None))
    model = CertModel(load_config(str(cfg_path)))
    chi0, init_ok = _cmp_initial(model, run, info or {"warm": None}) if run.positions else (None, None)
    chk.append(_check("initial fast state = expected (zero or warm)", init_ok))
    out["integrity"] = {"checks": chk, "commit": git.get("commit") or (run.run_meta or {}).get("git_commit"),
                        "snapshot_every": run.every, "num_snapshots": len(run.snaps) if run.snaps is not None else 0,
                        "num_failure_files": len(run.failures), "all_pass": all(c["status"] == "PASS" for c in chk)}
    # ---- P1 outcome
    row = summarize_scenario(sdir, {"status": "finished", "returncode": None}, ran)
    kind = _termination_kind(run.summary)
    s = run.summary or {}
    failing = sorted({int(q["agent_id"]) for q in s.get("qp_failures") or []})
    if kind == "strict_guard":
        failing = sorted({int(f["agent"]) for f in run.failures if f.get("step") == s.get("termination_step")
                          and f.get("class") == "inconsistent"})
    K = run.num_qp_states()
    out["outcome"] = {"termination_kind": kind, "termination_step": s.get("termination_step"),
                      "steps_completed": s.get("steps_completed"), "steps_configured": s.get("steps_configured"),
                      "num_qp_states": K, "failing_agents": failing,
                      "failure_kinds": sorted({q.get("failure_kind") for q in s.get("qp_failures") or []} - {None}),
                      "termination_exception": (s.get("termination_exception") or {}).get("message"),
                      "class_counts": s.get("class_counts"), "verdict": s.get("verdict"),
                      "batch_row": {k: row.get(k) for k in ("status", "min_pairwise_distance", "min_obstacle_clearance",
                                                            "min_weighted_lambda2", "min_lambda2_comm_binary",
                                                            "max_abs_conn_coupling", "max_abs_clf_coupling",
                                                            "max_abs_pair_coupling", "first_target_entry_step",
                                                            "tightest_lp_margin", "qp_inconsistent", "qp_borderline",
                                                            "qp_numerical_failure", "qp_infeasible")}}
    if (run.run / "arrays.npz").exists():
        a = np.load(run.run / "arrays.npz")
        n = len(run.positions)
        for key in ("conn_mismatch", "clf_mismatch"):
            v = np.abs(a[key][:n]) if a[key].size else np.zeros(0)
            out["outcome"][f"max_abs_{key}"] = float(np.nanmax(v)) if v.size and np.isfinite(v).any() else None
    # ---- P2 key states
    kt = s.get("termination_step") if kind != "completed" else None
    keys = {"initial": 0}
    if kt is not None:
        keys["termination"] = int(kt)
    if kind == "completed" and K >= 1:
        keys["last_qp_state"] = K - 1
    ks = sorted(set(range(0, K, max(1, opts.sample_every))) | set(range(max(0, K - opts.tail), K)))
    qrows = run.qp_rows(set(keys.values()) | set(ks))
    states = {}
    for name, k in keys.items():
        st = state_at(run, model, k)
        rec = {"k": k, "source": st["source"], "notes": st["notes"]}
        if st["z"] is not None:
            cls = classify_state(model, st, repository_check=True, keep_allocation=(name == "termination"))
            rec["classification"] = cls
            rec["agreement"] = _agreement(cls, qrows.get(k, []))
        rec["recorder_failures"] = [{x: f.get(x) for x in ("agent", "class", "solver_status", "s_star", "binding_rows",
                                                           "minimal_infeasible_subsets", "zero_mismatch_s_star",
                                                           "centralized_z_free_verdict", "centralized_z_free_s_star",
                                                           "decomposition")}
                                    for f in run.failures if f.get("step") == k]
        states[name] = rec
    out["key_states"] = states
    tcls = states.get("termination", {}).get("classification")
    out["termination_interpretation"] = _interpret_termination(kind, tcls, failing,
                                                               states.get("termination", {}).get("agreement"))
    # ---- P4 sampled trajectory
    from diagnostics.static_feasibility import StaticTolerances, _combine, _level3
    samples, unavailable = [], []
    for k in ks:
        snap = run.snapshot(k)
        if snap is None or k >= len(run.positions):
            unavailable.append(k)
            continue
        rows = model.rows(run.positions[k])
        st = {"positions": run.positions[k], "z": rows.to_vec(snap)[0][rows.nw:]}
        cls = classify_state(model, st, repository_check=False)
        l3z = _level3(rows, np.zeros(rows.n - rows.nw), StaticTolerances())
        ag = _agreement(cls, qrows.get(k, []))
        samples.append({"k": k, "category": cls["category"], "L1": cls["L1"]["verdict"], "L1_margin": cls["L1"]["lp_margin"],
                        "L2": cls["L2"]["verdict"], "L2_margin": cls["L2"]["lp_margin"], "L3": cls["L3_verdict"],
                        "L3_agents": cls["L3_agents"],
                        "L3_worst_margin": max((v["lp_margin"] for v in cls["L3"] if v["lp_margin"] is not None), default=None),
                        "L3_zero_mismatch": _combine([v.verdict for v in l3z]),
                        "L3_zero_mismatch_agents": "".join(v.verdict for v in l3z), "z_inf_norm": cls["z_inf_norm"],
                        "consistency": ";".join(cls["consistency"]), "agreement_flags": ";".join(ag["flags"])})
    first = lambda key, val: next((r["k"] for r in samples if r[key] == val), None)
    out["trajectory"] = {"sampled_states": len(samples), "requested_states": len(ks), "unavailable_states": unavailable,
                         "sample_every": opts.sample_every, "tail": opts.tail,
                         "first_sampled_L1_I": first("L1", "I"), "first_sampled_L2_I": first("L2", "I"),
                         "first_sampled_L3_I": first("L3", "I"), "first_sampled_U": next((r["k"] for r in samples if "U" in (r["L1"], r["L2"], r["L3"])), None),
                         "categories": {c: sum(r["category"] == c for r in samples) for c in sorted({r["category"] for r in samples})},
                         "agreement_flag_states": [r["k"] for r in samples if r["agreement_flags"]],
                         "max_z_inf_norm": max((r["z_inf_norm"] for r in samples), default=None)}
    out["_samples"] = samples
    # ---- P5 certificates
    if opts.certificates and run.positions:
        cert = _certificates(model, run, chi0, K, opts)
        E_ni = cert.pop("_E_ni", [])
        out["certificates"] = cert
        if opts.observed_errors and "spectral" in cert:
            ncert = cert["spectral"]["results"]["non_increase"]["certified_states"]
            out["observed_errors"] = _observed_errors(model, run, ks, E_ni, ncert)
    # ---- P6 Tier-0 comparison
    t0 = (info or {}).get("tier0")
    if t0:
        ic = states["initial"].get("classification", {})
        obs_step0 = "terminates_at_step0" if kind in ("local_qp_failure", "strict_guard") and kt == 0 else "proceeds"
        cmp = {"tier0_category": t0["category"], "observed_initial_category": ic.get("category"),
               "category_match": t0["category"] == ic.get("category"),
               "tier0_step0": t0["step0"], "observed_step0": obs_step0, "step0_match": t0["step0"] == obs_step0,
               "tier0_frozen_cert_spectral": (t0["spectral_cert"] + " " + t0["spectral_reason"]).strip(),
               "tier0_frozen_cert_frobenius": (t0["frobenius_cert"] + " " + t0["frobenius_reason"]).strip()}
        if "certificates" in out and "spectral" in out["certificates"]:
            for norm in NORMS:
                r = out["certificates"][norm]["results"]["non_increase"]
                cmp[f"observed_cert_{norm}"] = f"{r['status']} {r['reason'] or ''}".strip()
        out["tier0_comparison"] = cmp
    out["wall_s"] = time.perf_counter() - t_start
    return out


# ----------------------------------------------------------------------------------------------- outputs
def summary_row(r: dict) -> dict:
    if "analysis_error" in r:
        return {"id": r["id"], "analysis_error": r["analysis_error"].splitlines()[-1]}
    o, ti, tr = r["outcome"], r["termination_interpretation"], r["trajectory"]
    ini = r["key_states"].get("initial", {}).get("classification", {})
    term = r["key_states"].get("termination", {})
    tc = term.get("classification", {})
    c = r.get("certificates", {})
    row = {"id": r["id"], "group": r["group"], "pilot": r["pilot"], "legacy": r["legacy"],
           "integrity_all_pass": r["integrity"]["all_pass"], "commit": (r["integrity"]["commit"] or "")[:7],
           "snapshot_every": r["integrity"]["snapshot_every"], "termination_kind": o["termination_kind"],
           "termination_step": o["termination_step"], "steps": f"{o['steps_completed']}/{o['steps_configured']}",
           "failing_agents": " ".join(map(str, o["failing_agents"])), "initial_category": ini.get("category"),
           "initial_L3_agents": ini.get("L3_agents"), "termination_state_source": term.get("source"),
           "termination_category": tc.get("category"), "termination_L1": tc.get("L1", {}).get("verdict"),
           "termination_L2": tc.get("L2", {}).get("verdict"), "termination_L3_agents": tc.get("L3_agents"),
           "termination_result": ti["result"], "termination_consistency": ";".join(tc.get("consistency", [])),
           "first_sampled_L3_I": tr["first_sampled_L3_I"], "first_sampled_L2_I": tr["first_sampled_L2_I"],
           "first_sampled_U": tr["first_sampled_U"], "sampled_states": tr["sampled_states"],
           "unavailable_sampled_states": len(tr["unavailable_states"]),
           "qp_inconsistent": (o["class_counts"] or {}).get("inconsistent"),
           "qp_borderline": (o["class_counts"] or {}).get("borderline"),
           "qp_numerical_failure": (o["class_counts"] or {}).get("numerical_failure"),
           "max_abs_conn_mismatch": o.get("max_abs_conn_mismatch"), "max_abs_clf_mismatch": o.get("max_abs_clf_mismatch"),
           "min_pairwise_distance": o["batch_row"].get("min_pairwise_distance"),
           "min_obstacle_clearance": o["batch_row"].get("min_obstacle_clearance")}
    for norm in NORMS:
        if norm in c:
            ni = c[norm]["results"]["non_increase"]
            row[f"cert_{norm}"] = ni["status"]
            row[f"cert_{norm}_reason"] = ni["reason"]
            row[f"cert_{norm}_certified"] = f"{ni['certified_states']}/{c['evaluated_states']}"
    if "spectral" in c:
        row["rho_status_k0"] = c["spectral"]["k0"]["rho_status"]
    obs = r.get("observed_errors") or []
    chk = [x for x in obs if "within_certified_radius" in x]
    row["observed_error_checks"] = f"{sum(x['within_certified_radius'] for x in chk)}/{len(chk)}" if chk else ""
    tc0 = r.get("tier0_comparison") or {}
    row.update(tier0_category=tc0.get("tier0_category"), category_match=tc0.get("category_match"),
               tier0_step0=tc0.get("tier0_step0"), step0_match=tc0.get("step0_match"),
               tier0_cert_spectral=tc0.get("tier0_frozen_cert_spectral"))
    return row


def _f(x, nd=3):
    if x is None or x == "":
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(results: list[dict], meta: dict, pairs: list) -> str:
    ok = [r for r in results if "analysis_error" not in r]
    err = [r for r in results if "analysis_error" in r]
    by = {r["id"]: r for r in ok}
    L = [f"# 48-scenario feasibility study: automated post-processing report", "",
         f"Generated {meta['finished_at']} by `diagnostics.study48.postprocess` at commit {meta['analysis_git']['commit']}"
         f"{' (modified tracked files)' if meta['analysis_git']['tracked_files_modified'] else ''}.",
         f"Dataset: `{meta['dataset']}` ({len(results)} scenarios analysed; manifest SHA-256 {meta['manifest_sha256'][:12]}…).", "",
         "All results are computed statically from the recorded outputs (no simulation). Terminology:",
         "* **certified infeasible**: the static checker returned I with a re-verified Farkas (or zero-row) certificate"
         " at the recorded state - a numerically verified statement in double precision, not an exact-arithmetic proof;",
         "* **solver-reported failure**: OSQP did not return a solution (or, for the strict guard, returned one for an"
         " LP-infeasible QP); it is listed separately from the static verdict;",
         "* **U / unresolved**: the verdict is numerically ambiguous, or the state needed to decide it was not recorded;",
         "* **completed** means the finite configured horizon was simulated; it does not show feasibility for later times;",
         "* certificates (R6.1 Thm 3) are sufficient conditions; a failed certificate does not imply infeasibility.", ""]
    kinds = {}
    for r in ok:
        kinds.setdefault(r["outcome"]["termination_kind"], []).append(r["id"])
    L += ["## 1. Outcomes", "", "| termination | count | scenarios |", "|---|---|---|"]
    L += [f"| {k} | {len(v)} | {' '.join(v)} |" for k, v in sorted(kinds.items())]
    res = {}
    for r in ok:
        res.setdefault(r["termination_interpretation"]["result"], []).append(r["id"])
    L += ["", "| termination result (static check at the termination state) | count | scenarios |", "|---|---|---|"]
    L += [f"| {k} | {len(v)} | {' '.join(v)} |" for k, v in sorted(res.items())]
    cats = {}
    for r in ok:
        c = r["key_states"].get("termination", {}).get("classification", {}).get("category")
        if c:
            cats.setdefault(c, []).append(r["id"])
    if cats:
        L += ["", "Category at the termination state (from the confirmed verdicts, not from the group labels):", "",
              "| category | count | scenarios |", "|---|---|---|"]
        L += [f"| {k} | {len(v)} | {' '.join(v)} |" for k, v in sorted(cats.items())]
    L += ["", "## 2. Per-scenario summary", "",
          "Certificates are the non-increase version along the recorded positions; for a terminated run they are"
          " evaluated on the QP states 0..k_term (the last one is the failing state). `first sampled L3 I` is the first"
          " sampled state whose actual mismatch makes some local QP infeasible (sampling resolution applies).", "",
          "| ID | group | termination | step | x₀ category | termination category (L3 agents) | result | first sampled L3 I | cert spectral | cert Frobenius | Tier-0 match (cat / step 0) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: r["id"]):
        sr = summary_row(r)
        tcat = f"{sr['termination_category']} ({sr['termination_L3_agents']})" if sr["termination_category"] else "–"
        cs = lambda n: f"{sr.get('cert_' + n, '–')}{(' ' + sr['cert_' + n + '_reason']) if sr.get('cert_' + n + '_reason') else ''}"
        L.append(f"| {r['id']}{' (legacy)' if r['legacy'] else ''} | {r['group']} | {sr['termination_kind']} | {_f(sr['termination_step'])} | "
                 f"{_f(sr['initial_category'])} | {tcat} | {sr['termination_result']} | {_f(sr['first_sampled_L3_I'])} | "
                 f"{cs('spectral')} | {cs('frobenius')} | {_f(sr['category_match'])} / {_f(sr['step0_match'])} |")
    comp = [r for r in ok if r["outcome"]["termination_kind"] == "completed"]
    L += ["", "## 3. Completed runs and certificates", "",
          "Finite-horizon results only. `certified` counts the QP states at which the non-increase version of the"
          " certificate holds.", "",
          "| ID | QP states | spectral (non-increase) | Frobenius (non-increase) | ρ status k=0 | observed e_k ≤ E_k |", "|---|---|---|---|---|---|"]
    for r in sorted(comp, key=lambda r: r["id"]):
        sr = summary_row(r)
        L.append(f"| {r['id']} | {r['outcome']['num_qp_states']} | {sr.get('cert_spectral', '–')} {sr.get('cert_spectral_reason') or ''} "
                 f"({sr.get('cert_spectral_certified', '–')}) | {sr.get('cert_frobenius', '–')} {sr.get('cert_frobenius_reason') or ''} "
                 f"({sr.get('cert_frobenius_certified', '–')}) | {_f(sr.get('rho_status_k0'))} | {sr['observed_error_checks'] or '–'} |")
    L += ["", "## 4. Tier-0 predictions vs observations", ""]
    mm = [r for r in ok if r.get("tier0_comparison") and not (r["tier0_comparison"]["category_match"] and r["tier0_comparison"]["step0_match"])]
    if mm:
        L += ["| ID | Tier-0 category | observed x₀ category | Tier-0 step 0 | observed step 0 |", "|---|---|---|---|---|"]
        L += [f"| {r['id']} | {r['tier0_comparison']['tier0_category']} | {r['tier0_comparison']['observed_initial_category']} | "
              f"{r['tier0_comparison']['tier0_step0']} | {r['tier0_comparison']['observed_step0']} |" for r in mm]
    else:
        L += ["All step-0 predictions and x₀ categories match the observations."]
    L += ["", "## 5. Numerical issues and unresolved results", ""]
    issues = []
    for r in sorted(ok, key=lambda r: r["id"]):
        for name, st in r["key_states"].items():
            cls = st.get("classification")
            if cls is None:
                issues.append(f"{r['id']} {name} (k={st['k']}): state unavailable - {'; '.join(st['notes'])}")
                continue
            for c in cls["consistency"]:
                issues.append(f"{r['id']} {name} (k={st['k']}): {c}")
            for fl in st.get("agreement", {}).get("flags", []):
                issues.append(f"{r['id']} {name} (k={st['k']}): {fl}")
            if "U" in (cls["L1"]["verdict"], cls["L2"]["verdict"], cls["L3_verdict"]):
                issues.append(f"{r['id']} {name} (k={st['k']}): U verdict (L1 {cls['L1']['verdict']}, L2 {cls['L2']['verdict']}, L3 {cls['L3_agents']})")
            for n in st["notes"]:
                if "DIFFERS" in n or "disagree" in n:
                    issues.append(f"{r['id']} {name} (k={st['k']}): {n}")
        cc = r["outcome"]["class_counts"] or {}
        for key in ("inconsistent", "numerical_failure", "borderline", "lp_error"):
            if cc.get(key):
                issues.append(f"{r['id']}: {cc[key]} QP(s) of recorder class `{key}` during the run")
        if r["trajectory"]["unavailable_states"]:
            issues.append(f"{r['id']}: {len(r['trajectory']['unavailable_states'])} of {r['trajectory']['requested_states']} "
                          f"requested trajectory samples have no snapshot (interval {r['integrity']['snapshot_every']}); not interpolated")
        if r["trajectory"]["agreement_flag_states"]:
            issues.append(f"{r['id']}: recorder/static disagreement at sampled steps {r['trajectory']['agreement_flag_states'][:10]}")
        for c in r["integrity"]["checks"]:
            if c["status"] != "PASS":
                issues.append(f"{r['id']}: integrity {c['status']}: {c['check']} {c['detail']}")
        for ob in r.get("observed_errors") or []:
            if ob.get("within_certified_radius") is False:
                issues.append(f"{r['id']}: observed error {ob['e']:.3e} exceeds the certified radius {ob['E_ni']:.3e} at k={ob['k']}")
    L += [f"* {x}" for x in issues] or ["None found."]
    if err:
        L += ["", "Analysis errors (scenario not analysed):", ""] + [f"* {r['id']}: {r['analysis_error'].splitlines()[-1]}" for r in err]
    L += ["", "## 6. Comparison pairs (manifest; descriptive)", "",
          "| pair | kind | outcome A | outcome B |", "|---|---|---|---|"]
    seen = set()
    for a, b, kind in pairs:
        key = tuple(sorted((a, b))) + (kind,)
        if key in seen or a not in by or b not in by:
            continue
        seen.add(key)
        oa, ob = (f"{by[x]['outcome']['termination_kind']}@{by[x]['outcome']['termination_step']}, "
                  f"{by[x]['termination_interpretation']['result']}" for x in (a, b))
        L.append(f"| {a} – {b} | {kind} | {oa} | {ob} |")
    L += ["", "Only single-parameter pairs support attribution to that parameter; broader contrasts are descriptive.", "",
          "## 7. Limitations", "",
          "* Verdicts are numerically verified in double precision (witness slack ≤ −1e-9, certificate margin > 1e-10"
          " plus a rounding bound); they become exact statements only where an exact-arithmetic or structural argument is given.",
          "* The trajectory analysis is sampled (every "
          f"{meta['options']['sample_every']} steps plus the last {meta['options']['tail']} QP states) and uses only"
          " recorded snapshots; between samples nothing is asserted.",
          "* A state without a failure file or snapshot is reported as unavailable; no replay or interpolation is used.",
          "* The certificates are evaluated along the recorded positions; they are sufficient conditions on the finite"
          " horizon of each run and say nothing about longer horizons.",
          "* ρ with status `upper_bound_below_tol` is numerically zero, which is not a proof that ρ = 0.",
          "* Group labels are design intents; all categories above are computed from the confirmed verdicts."]
    return "\n".join(L) + "\n"


def _git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                 # noqa: BLE001
            return None
    st = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "tracked_files_modified": None if st is None else bool(st)}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _worker(args):
    sdir, sid, info, opts = args
    try:
        return analyze_scenario(Path(sdir), sid, info, opts)
    except Exception:                                     # noqa: BLE001 - one scenario never stops the batch
        return {"id": sid, "analysis_error": traceback.format_exc()}


def run_postprocessing(dataset: Path, out: Path, inputs: Path = DEFAULT_INPUTS, ids: list[str] | None = None,
                       workers: int = 1, opts: Options = Options()) -> dict:
    dataset = Path(dataset).resolve()
    out = Path(out).resolve()
    sc_dir = resolve_dataset(dataset)
    root = sc_dir.parent if sc_dir.name == "scenarios" else sc_dir
    for base in (root, sc_dir):
        if out == base or base in out.parents:
            raise ValueError(f"output directory {out} must be outside the dataset {root}")
    for p in sc_dir.iterdir():
        rp = Path(os.path.realpath(p))
        if out == rp or rp in out.parents:
            raise ValueError(f"output directory {out} lies inside dataset run directory {rp}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be new or empty: {out}")
    (out / "scenarios").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    inp = load_inputs(inputs)
    present = sorted(p.name for p in sc_dir.iterdir() if (p / "run").is_dir())
    sel = [i for i in present if not ids or i in ids]
    missing = sorted(set(inp["scenarios"]) - set(present)) if not ids else sorted(set(ids) - set(present))
    unknown = sorted(set(present) - set(inp["scenarios"]))
    meta = {"dataset": str(dataset), "scenarios_dir": str(sc_dir), "out": str(out), "inputs": str(inputs),
            "manifest_sha256": inp["manifest_sha256"], "options": asdict(opts), "workers": workers,
            "analysis_git": _git_state(), "python": platform.python_version(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected": sel, "missing_from_dataset": missing, "not_in_manifest": unknown}
    if (root / "SOURCES.txt").exists():
        meta["dataset_sources"] = (root / "SOURCES.txt").read_text()
    jobs = [(str(sc_dir / i), i, inp["scenarios"].get(i), opts) for i in sel]
    results = []
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_worker, jobs):
                results.append(r)
                print(f"[postprocess] {r['id']}: {'ERROR' if 'analysis_error' in r else r['termination_interpretation']['result']}",
                      file=sys.stderr, flush=True)
    else:
        for j in jobs:
            r = _worker(j)
            results.append(r)
            print(f"[postprocess] {r['id']}: {'ERROR' if 'analysis_error' in r else r['termination_interpretation']['result']}",
                  file=sys.stderr, flush=True)
    for r in results:
        samples = r.pop("_samples", None)
        (out / "scenarios" / f"{r['id']}.json").write_text(json.dumps(r, indent=2, default=_json_default))
        if samples:
            with open(out / "samples" / f"{r['id']}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
                w.writeheader()
                w.writerows(samples)
    rows = [summary_row(r) for r in results]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=_json_default))
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["analysis_errors"] = [r["id"] for r in results if "analysis_error" in r]
    (out / "analysis_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    (out / "report.md").write_text(write_report(results, meta, inp["pairs"]))
    return {"meta": meta, "rows": rows, "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--cert-max-states", type=int, default=None)
    ap.add_argument("--no-certificates", action="store_true")
    ap.add_argument("--no-observed-errors", action="store_true")
    a = ap.parse_args(argv)
    opts = Options(sample_every=a.sample_every, tail=a.tail, cert_max_states=a.cert_max_states,
                   certificates=not a.no_certificates, observed_errors=not a.no_observed_errors)
    res = run_postprocessing(Path(a.dataset), Path(a.out), Path(a.inputs), a.ids.split(",") if a.ids else None,
                             a.workers, opts)
    m = res["meta"]
    print(f"[postprocess] {len(res['rows'])} scenarios -> {m['out']}; missing from dataset: {m['missing_from_dataset'] or 'none'}; "
          f"analysis errors: {m['analysis_errors'] or 'none'}", file=sys.stderr)
    return 1 if m["analysis_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

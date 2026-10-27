"""The 48 scenarios of the feasibility study (plan v2) and the configuration generator.

    python -m diagnostics.study48.scenarios --out STUDY_DIR [--no-warm]

Every scenario is the repository base configuration ``configs/distributed_10_agents.yaml`` with the study baseline
P0 (Method B M = 10, gated coupling, exact connectivity, alpha = 1) plus the scenario's own positions, obstacles,
horizon and parameter changes; A12 is the unchanged repro configuration. Construction rules are applied exactly as
declared in the plan; two parameters are calibrated by a declared rule on the Level-1 LP (A02: actuator threshold;
B03: Level-1 margin), never on Level-2/3 outcomes. Random families use explicit seeds and record rejections.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

from diagnostics.study48.geometry import (cutoff_margin, head_on_obstacle, lambda2, lattice, lattice_with_cutoff_rule,
                                          pair_distances, physical_validity, q_values)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "configs/distributed_10_agents.yaml"
REPRO = REPO / "configs/repro/step0_failure_2026-03-25.yaml"
SIG = [[2.0, 0.0], [0.0, 0.25]]
CEN = [0.0, 0.0]
FAR_OBS = [[-5.0, 1.0], 0.6]
PILOTS = {"A01", "A03", "A04", "A09", "B01", "B02", "B04", "B07", "C02", "C03", "C09", "C11", "D02", "D03", "D04", "D08"}

# ------------------------------------------------------------------------------------- fixed geometries
FAR2 = [[-1.25, 0.25], [-1.25, -0.25]]
A_GEOM = [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.05], [0.8, 0.1]]                    # Experiment A
B_GEOM = [[-0.56, 0.0], [-1.0, 0.0], [-0.75, 0.5]]                                 # Experiment B convoy
C_GEOM = [[-0.135, 2.927], [-0.498, 3.064], [-0.425, 2.687]]                       # Configuration C
D3_GEOM, D3_OBS = [[0.054, -3.868], [-0.14, -3.514]], [[0.78, -3.503], 0.6]       # Configuration D3
D2_GEOM, D2_OBS = [[-0.33, 3.381], [0.118, 3.226]], [[-0.788, 2.772], 0.6]        # Configuration D2
B01_GEOM = [[-0.55, 1.35], [0.0, 1.25], [0.55, 1.35]]
B04_GEOM = [[-0.55, 1.35], [0.0, 1.483], [0.55, 1.35]]
B05_GEOM = [[-0.55, 1.35], [-0.95, 1.05], [-0.1, 1.2], [0.28, 0.93], [0.72, 1.12]]
B06_GEOM = [[-0.5, 1.3], [-0.85, 1.05], [-0.85, 1.55], [-0.1, 1.12], [0.2, 0.84], [0.55, 1.0], [0.95, 0.85],
            [0.85, 1.35]]                       # O1a, O1b, O1c (blocked) | I1, I2 | O2, O3, O4 (free)
B07_GEOM = [[-0.5, 1.45], [0.0, 1.673], [0.5, 1.45]]
B12_GEOM = [[-0.44, 1.35], [0.0, 1.25], [0.44, 1.35]]


def _ho(x, r, g):
    return list(head_on_obstacle(x, r, g, SIG, CEN))


# ------------------------------------------------------------------------------------- random families
def _family_rules(X, obs, d0, eta, d_c=0.9, eps_c=0.1):
    D = pair_distances(X)
    if min(d for *_, d in D) < d0 + 0.05:
        return False
    if obs and min(np.linalg.norm(np.asarray(x) - np.asarray(o[0])) - o[1] for x in X for o in obs) < 0.05:
        return False
    if cutoff_margin(X, d_c, eps_c) < 0.02:
        return False
    return lambda2(X, d_c, eps_c) >= eta + 0.01


def family(name: str, seed: int, max_tries: int = 500000):
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(max_tries):
        if name == "FA":       # A10/A11: uniform box in front of a 4-obstacle wall
            obs = [[[-0.9, y], 0.2] for y in (-0.6, -0.2, 0.2, 0.6)]
            X = [[float(rng.uniform(-2.4, -1.2)), float(rng.uniform(-0.6, 0.6))] for _ in range(5)]
        elif name == "FD":     # D09/D10: uniform far box, far obstacle
            obs = [FAR_OBS]
            X = [[float(rng.uniform(-2.3, -1.3)), float(rng.uniform(-0.5, 0.5))] for _ in range(5)]
        elif name == "FB":     # B10/B11: 2-3 inside (q <= -0.1), the rest outside with q in [0.2, 1.0]
            n_in = int(rng.choice([2, 3]))
            X = []
            while len(X) < 5:
                p = [float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.8, 1.8))]
                qq = q_values([p], SIG, CEN, 0.5)[0]
                if (len(X) < n_in and qq <= -0.1) or (len(X) >= n_in and 0.2 <= qq <= 1.0):
                    X.append(p)
            qs = q_values(X, SIG, CEN, 0.5)
            k = int(np.argmax(qs))
            obs = [_ho(X[k], 0.15, 0.05)]
        else:
            raise ValueError(name)
        if _family_rules(X, obs, 0.2, 0.1):
            return X, obs, {"family": name, "seed": seed, "rejections": rej}
        rej += 1
    raise RuntimeError(f"family {name} seed {seed}: no admissible sample in {max_tries} tries")


# ------------------------------------------------------------------------------------- the 48 specs
def S(id_, group, objective, positions, obstacles, horizon_s, overrides=None, init="zero", pairs=(), calibration=None,
      exempt=(), notes="", source="P0", legacy=False):
    return dict(id=id_, group=group, pilot=id_ in PILOTS, objective=objective, positions=positions, obstacles=obstacles,
                horizon_s=horizon_s, overrides=dict(overrides or {}), init=init, pairs=list(pairs),
                calibration=calibration, exempt=list(exempt), notes=notes, source=source, legacy=legacy)


def scenario_specs() -> list[dict]:
    W2 = lambda: [_ho(x, 0.6, 0.1) for x in FAR2]
    sp = []
    # ---- Group A
    sp += [
        S("A01", "A", "actuator-limited CLF infeasibility", FAR2, [], 0.05, {"controller.clf_rate": 2.0,
          "controller.u_max": 0.05}, pairs=[("A02", "single-parameter")]),
        S("A02", "A", "near-threshold Level-1 behaviour (u_max = 1.05 x LP threshold)", FAR2, [], 0.5,
          {"controller.clf_rate": 2.0}, calibration={"rule": "u_max = 1.05 * Level-1 LP actuator threshold"},
          pairs=[("A01", "single-parameter")]),
        S("A03", "A", "CLF vs obstacles (both agents head-on blocked)", FAR2, W2(), 0.05, {"controller.clf_rate": 1.0},
          pairs=[("A05", "single-parameter"), ("B01", "contrast")]),
        S("A04", "A", "collision-chain blocking (convoy behind an obstacle)", [[-1.0, 0.0], [-1.35, 0.0], [-1.7, 0.0]],
          [[[-0.6, 0.0], 0.3]], 0.05, {"controller.clf_rate": 1.2}, pairs=[("C02", "contrast")]),
        S("A05", "A", "joint CBF rate (A03 with cbf_rate 1)", FAR2, W2(), 0.05,
          {"controller.clf_rate": 1.0, "controller.cbf_rate": 1.0}, pairs=[("A03", "single-parameter")]),
        S("A06", "A", "emergent centralized infeasibility at a narrow passage",
          [[-1.8, -0.3], [-1.8, 0.0], [-1.8, 0.3]], [[[-0.9, 0.42], 0.3], [[-0.9, -0.42], 0.3]], 2.0,
          {"controller.clf_rate": 0.25}, pairs=[("C10", "contrast")]),
        S("A07", "A", "CLF vs connectivity near the connectivity boundary (possible informative null)",
          [[-1.3, -0.63], [-1.3, 0.0], [-1.3, 0.63]], [[[-0.85, 0.0], 0.25]], 1.0, {"controller.clf_rate": 0.25},
          exempt=["lambda2_ge_eta_plus_0.01"], notes="designated connectivity-boundary case (lambda2 - eta in [1e-3, 1e-2])",
          pairs=[("C11", "contrast")]),
        S("A08", "A", "N-scaling of actuator infeasibility (N = 10)", lattice(2, 5, 0.42, -1.3), [], 0.05,
          {"controller.clf_rate": 2.0, "controller.u_max": 0.05}, pairs=[("A01", "contrast")]),
    ]
    a09, s09 = lattice_with_cutoff_rule(2, 4, 0.25, -1.5, 0.9, 0.1)
    sp += [
        S("A09", "A", "many simultaneous CBFs (dense N = 8 cluster, large obstacle head-on for the centroid)", a09,
          [[[-1.5 + 0.1 + 0.8, 0.0], 0.8]], 0.2, {"controller.clf_rate": 0.5},
          notes=f"2x4 lattice, spacing {s09} (smallest >= d0+0.05 on a 0.005 grid satisfying the cutoff rule)"),
    ]
    for id_, seed in (("A10", 101), ("A11", 102)):
        X, obs, fam = family("FA", seed)
        sp.append(S(id_, "A", "random clutter family FA (replicate)", X, obs, 0.2, {"controller.clf_rate": 1.0},
                    calibration=fam, pairs=[("A11" if id_ == "A10" else "A10", "replicate")]))
    sp.append(S("A12", "A", "blind classification of the step-0 failure repro (label provisional)", None, None, None,
                source="repro", exempt=["pair_ge_d0_plus_0.05", "obstacle_clearance_ge_0.05", "lambda2_ge_eta_plus_0.01",
                                        "cutoff_margin_ge_0.02"], legacy=True,
                notes="legacy diagnostic case: existing configuration used unchanged; its initial state violates the "
                      "connectivity requirement (lambda2 < eta), which is reported, not repaired; lambda2 < eta does "
                      "not by itself make the centralized QP infeasible"))
    # ---- Group B
    ob1 = _ho(B01_GEOM[0], 0.15, 0.05)
    sp += [
        S("B01", "B", "gated v-graph split by an inside agent; O1 blocked", B01_GEOM, [ob1], 0.1,
          {"controller.clf_rate": 2.0}, pairs=[("B02", "single-parameter"), ("B03", "single-parameter")]),
        S("B02", "B", "B01 in communication mode", B01_GEOM, [ob1], 0.1,
          {"controller.clf_rate": 2.0, "controller.clf_coupling": "communication"}, pairs=[("B01", "single-parameter")]),
        S("B03", "B", "B01 with a barely feasible Level 1 (u_max calibrated)", B01_GEOM, [ob1], 0.1,
          {"controller.clf_rate": 2.0}, calibration={"rule": "u_max such that the Level-1 LP margin is -5e-4"},
          pairs=[("B01", "single-parameter")]),
        S("B04", "B", "near-structural: connector just outside the target (sigma ~ 1e-9)", B04_GEOM,
          [_ho(B04_GEOM[0], 0.15, 0.05)], 0.1, {"controller.clf_rate": 2.0}, pairs=[("B01", "design-factor")]),
        S("B05", "B", "blocked component of size 2 with two inside separators", B05_GEOM,
          [_ho(B05_GEOM[0], 0.15, 0.05), _ho(B05_GEOM[1], 0.15, 0.05)], 0.1, {"controller.clf_rate": 2.0},
          pairs=[("B06", "contrast")]),
        S("B06", "B", "N = 8: blocked cluster of 3 and free cluster of 3 joined by 2 inside agents", B06_GEOM,
          [_ho(B06_GEOM[k], 0.15, 0.05) for k in (0, 1, 2)], 0.1,
          {"controller.clf_rate": 2.0, "fast_dynamics.eta": 0.05},
          notes="finalized geometry v1a; eta = 0.05 fallback (lambda2 = 0.097 < 0.11 required at eta = 0.1)",
          pairs=[("B05", "contrast")]),
        S("B07", "B", "approach to structural infeasibility (transition attempt)", B07_GEOM,
          [_ho(B07_GEOM[0], 0.15, 0.30)], 3.0, {"controller.clf_rate": 2.0}, pairs=[("B08", "single-parameter")]),
        S("B08", "B", "B07 in communication mode", B07_GEOM, [_ho(B07_GEOM[0], 0.15, 0.30)], 3.0,
          {"controller.clf_rate": 2.0, "controller.clf_coupling": "communication"}, pairs=[("B07", "single-parameter")]),
        S("B09", "B", "connectivity part of Prop 2 (control case near the connectivity boundary)",
          [[-1.3, -0.864], [-1.3, -0.288], [-1.3, 0.288], [-1.3, 0.864]], [[[-0.8, 0.55], 0.2]], 0.5,
          {"controller.clf_rate": 0.1}, exempt=["lambda2_ge_eta_plus_0.01"],
          notes="designated connectivity-boundary case", pairs=[("C11", "contrast")]),
    ]
    for id_, seed in (("B10", 201), ("B11", 202)):
        X, obs, fam = family("FB", seed)
        sp.append(S(id_, "B", "random inside/outside family FB (replicate)", X, obs, 0.1, {"controller.clf_rate": 2.0},
                    calibration=fam, pairs=[("B11" if id_ == "B10" else "B10", "replicate")]))
    sp.append(S("B12", "B", "exact v-edge at floating-point resolution (h ~ 1e-22)", B12_GEOM,
                [_ho(B12_GEOM[0], 0.15, 0.05)], 0.1, {"controller.clf_rate": 3.0},
                notes="designated cutoff case: O1-O2 at d_c - 0.02", pairs=[("B01", "contrast")]))
    # ---- Group C
    sp += [
        S("C01", "C", "calibration: Experiment A in communication mode; zero-row verdict test", A_GEOM, [FAR_OBS], 2.0,
          {"controller.clf_coupling": "communication"}, pairs=[("D01", "single-parameter")]),
        S("C02", "C", "calibration: Experiment B convoy (gated)", B_GEOM, [FAR_OBS], 5.0, {"controller.clf_rate": 0.5},
          pairs=[("C05", "design-factor"), ("C06", "single-parameter"), ("C07", "single-parameter"),
                 ("C08", "single-parameter"), ("A04", "contrast")]),
        S("C03", "C", "static necessity: D3 with zero initialization", D3_GEOM, [D3_OBS], 0.001,
          {"controller.clf_rate": 0.25}, pairs=[("C04", "design-factor"), ("D03", "design-factor")]),
        S("C04", "C", "insufficient warm start (stop at E ~ E_adm)", D3_GEOM, [D3_OBS], 0.3, {"controller.clf_rate": 0.25},
          init={"type": "E_rel", "factor": 1.0}, pairs=[("C03", "design-factor"), ("D03", "design-factor")]),
        S("C05", "C", "time-scale separation at fixed r (tau 0.005, M 40)", B_GEOM, [FAR_OBS], 5.0,
          {"controller.clf_rate": 0.5, "fast_dynamics.tau": 0.005, "fast_dynamics.substeps": 40},
          pairs=[("C02", "design-factor")]),
        S("C06", "C", "regularization xi = 0.2", B_GEOM, [FAR_OBS], 5.0, {"controller.clf_rate": 0.5,
          "fast_dynamics.xi": 0.2}, pairs=[("C02", "single-parameter")]),
        S("C07", "C", "Method B with M = 1 (r = 0.05)", B_GEOM, [FAR_OBS], 5.0, {"controller.clf_rate": 0.5,
          "fast_dynamics.substeps": 1}, pairs=[("C02", "single-parameter")]),
        S("C08", "C", "calibration: Experiment B convoy in communication mode", B_GEOM, [FAR_OBS], 5.0,
          {"controller.clf_rate": 0.5, "controller.clf_coupling": "communication"}, pairs=[("C02", "single-parameter")]),
        S("C09", "C", "near-boundary approach with active gated CLF coupling (no crossing assumed)",
          [[-0.19, 1.62], [0.19, 1.62], [0.0, 1.95]], [], 5.0, {"controller.clf_rate": 1.0},
          pairs=[("D02", "contrast")]),
        S("C10", "C", "dynamically active pairwise mismatch at a gap (d0 = 0.25)",
          [[-1.8, -0.62], [-1.8, -0.31], [-1.8, 0.0], [-1.8, 0.31], [-1.8, 0.62]],
          [[[-0.9, 0.5], 0.3], [[-0.9, -0.5], 0.3]], 3.0, {"controller.clf_rate": 0.25, "controller.safety_distance": 0.25},
          pairs=[("A06", "contrast")]),
        S("C11", "C", "candidate for dynamically active connectivity mismatch",
          [[-1.5, -1.028], [-1.5, -0.514], [-1.5, 0.0], [-1.5, 0.514], [-1.5, 1.028]],
          [[[-0.95, 0.3], 0.25], [[-0.95, -0.3], 0.25]], 5.0, {"controller.clf_rate": 0.1},
          exempt=["lambda2_ge_eta_plus_0.01"], notes="designated connectivity-boundary case",
          pairs=[("D06", "contrast"), ("A07", "contrast")]),
        S("C12", "C", "scale (N = 10) with a faster CLF", lattice(2, 5, 0.31, -1.3), [FAR_OBS], 2.0,
          {"controller.clf_rate": 0.25}, pairs=[("D08", "contrast")]),
    ]
    # ---- Group D
    sp += [
        S("D01", "D", "calibration: Experiment A gated (certified, mismatch inactive)", A_GEOM, [FAR_OBS], 2.0,
          pairs=[("C01", "single-parameter")]),
        S("D02", "D", "calibration: Configuration C (certified, active CLF mismatch, not necessary)", C_GEOM, [FAR_OBS],
          2.0, pairs=[("D12", "single-parameter")]),
        S("D03", "D", "warm start, necessary pairwise redistribution; certificate vs feasibility horizon", D3_GEOM,
          [D3_OBS], 0.3, {"controller.clf_rate": 0.25}, init={"type": "E_abs", "value": 0.0056},
          notes="stopping rule E <= 0.0056 (the D3 experiment's rule; ~0.1 E_adm)",
          pairs=[("C03", "design-factor"), ("C04", "design-factor")]),
        S("D04", "D", "5-agent, 7-obstacle baseline (distributed_10_agents.yaml has 5 agents)", None, None, 10.0,
          source="base", pairs=[("D05", "single-parameter"), ("D06", "single-parameter")]),
        S("D05", "D", "baseline with d0 = 0.30 (replica of the earlier Method B run)", None, None, 10.0,
          {"controller.safety_distance": 0.3}, source="base", pairs=[("D04", "single-parameter")],
          exempt=["pair_ge_d0_plus_0.05"], notes="calibration replica; initial pair margin 0.02 above d0 (safe)"),
        S("D06", "D", "baseline with eta = 0.125 (replica of the earlier connectivity run)", None, None, 10.0,
          {"fast_dynamics.eta": 0.125}, source="base", pairs=[("D04", "single-parameter")],
          exempt=["lambda2_ge_eta_plus_0.01"], notes="calibration replica; designated near-boundary connectivity"),
        S("D07", "D", "N = 8 nominal lattice", lattice(2, 4, 0.42, -1.35), [], 2.0, pairs=[("D08", "contrast")]),
        S("D08", "D", "N = 10 nominal lattice", lattice(2, 5, 0.42, -1.35), [], 2.0,
          pairs=[("D07", "contrast"), ("C12", "contrast")]),
    ]
    for id_, seed in (("D09", 401), ("D10", 402)):
        X, obs, fam = family("FD", seed)
        sp.append(S(id_, "D", "random open-space family FD (replicate)", X, obs, 2.0, calibration=fam,
                    pairs=[("D10" if id_ == "D09" else "D09", "replicate")]))
    sp += [
        S("D11", "D", "faster motion (backup corner D2) with a warm start", D2_GEOM, [D2_OBS], 0.05,
          {"controller.clf_rate": 0.5}, init={"type": "E_rel", "factor": 0.1}, pairs=[("D03", "contrast")]),
        S("D12", "D", "Configuration C in communication mode", C_GEOM, [FAR_OBS], 2.0,
          {"controller.clf_coupling": "communication"}, pairs=[("D02", "single-parameter")]),
    ]
    return sp


# ------------------------------------------------------------------------------------- YAML construction
def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def build_raw(spec: dict, u_max: float | None = None) -> dict:
    if spec["source"] == "repro":
        return yaml.safe_load(REPRO.read_text())
    raw = yaml.safe_load(BASE.read_text())
    dt = raw["simulation"]["dt"]
    _set(raw, "fast_dynamics.substeps", 10)
    _set(raw, "controller.clf_coupling", "gated")
    _set(raw, "simulation.steps", max(1, int(round(spec["horizon_s"] / dt))))
    if spec["source"] == "P0":
        raw["agents"].update(num_agents=len(spec["positions"]), initial_positions=[list(map(float, p))
                             for p in spec["positions"]], init_mode="manual")
        raw["obstacles"]["obstacles"] = [{"center": [float(o[0][0]), float(o[0][1])], "radius": float(o[1])}
                                         for o in spec["obstacles"]]
    for k, v in spec["overrides"].items():
        _set(raw, k, v)
    if u_max is not None:
        _set(raw, "controller.u_max", float(u_max))
    return raw


def _load(raw, tmpdir: Path, name: str):
    from config import load_config
    p = tmpdir / f"{name}.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(str(p))


def _level1(raw, tmpdir, name):
    from diagnostics.certificate import CertModel
    from diagnostics.static_feasibility import StaticTolerances, classify_polytope, level1_problem
    cfg = _load(raw, tmpdir, name)
    m = CertModel(cfg)
    rows = m.rows(cfg.agents.initial_positions)
    M, b, lab = level1_problem(rows)
    u = cfg.controller.u_max
    return classify_polytope(M, b, -u * np.ones(rows.nw), u * np.ones(rows.nw), lab, StaticTolerances())


def calibrate_actuator_threshold(spec, tmpdir, lo=1e-3, hi=20.0, rel=1e-6):
    """Bracketed bisection on u_max for the Level-1 verdict (feasibility is monotone in u_max)."""
    vlo, vhi = _level1(build_raw(spec, lo), tmpdir, "cal_lo"), _level1(build_raw(spec, hi), tmpdir, "cal_hi")
    if vhi.verdict != "F":
        return {"finite_threshold": False, "hi": hi, "verdict_hi": vhi.verdict, "note": "Level 1 not F at u_max = 20"}
    if vlo.verdict != "I":
        return {"finite_threshold": False, "lo": lo, "verdict_lo": vlo.verdict, "note": "Level 1 not I at the lower end"}
    it = 0
    while (hi - lo) / hi > rel and it < 200:
        mid = 0.5 * (lo + hi)
        v = _level1(build_raw(spec, mid), tmpdir, "cal_mid")
        if v.verdict == "F":
            hi, vhi = mid, v
        elif v.verdict == "I":
            lo, vlo = mid, v
        else:
            break
        it += 1
    return {"finite_threshold": True, "lo": lo, "hi": hi, "iterations": it, "threshold": hi,
            "evidence_lo": {"verdict": vlo.verdict, "phi": vlo.certificate["phi"] if vlo.certificate else None,
                            "support": vlo.certificate.get("support") if vlo.certificate else None},
            "evidence_hi": {"verdict": vhi.verdict, "witness_slack": vhi.witness_slack, "lp_margin": vhi.lp_margin}}


def calibrate_level1_margin(spec, tmpdir, target=-5e-4, lo=1e-3, hi=20.0, rel=1e-9):
    """u_max with Level-1 LP margin s*_1 = target (s* decreases monotonically as u_max grows)."""
    s = lambda u: _level1(build_raw(spec, u), tmpdir, "calm").lp_margin
    if not (s(lo) > target > s(hi)):
        return {"ok": False, "s_lo": s(lo), "s_hi": s(hi)}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if s(mid) > target else (lo, mid)
        if (hi - lo) / hi < rel:
            break
    v = _level1(build_raw(spec, hi), tmpdir, "calm")
    return {"ok": True, "u_max": hi, "lp_margin": v.lp_margin, "verdict": v.verdict, "witness_slack": v.witness_slack}


def generate(out: Path, warm: bool = True) -> dict:
    from diagnostics.certificate import CertModel
    out = Path(out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "warm").mkdir(exist_ok=True)
    tmp = out / ".tmp"
    tmp.mkdir(exist_ok=True)
    specs = scenario_specs()
    assert len(specs) == 48 and len({s["id"] for s in specs}) == 48
    manifest = []
    for spec in specs:
        u_override = None
        cal_out = None
        if spec["id"] == "A02":
            cal_out = calibrate_actuator_threshold(spec, tmp)
            u_override = 1.05 * cal_out["threshold"] if cal_out["finite_threshold"] else None
        elif spec["id"] == "B03":
            cal_out = calibrate_level1_margin(spec, tmp)
            u_override = cal_out["u_max"] if cal_out["ok"] else None
        raw = build_raw(spec, u_override)
        text = yaml.safe_dump(raw, sort_keys=False)
        path = out / "configs" / f"{spec['id']}.yaml"
        path.write_text(text)
        from config import load_config
        cfg = load_config(str(path))
        X = [list(map(float, p)) for p in cfg.agents.initial_positions]
        obs = [[list(map(float, o["center"])), float(o["radius"])] for o in cfg.obstacles.obstacles]
        f, c = cfg.fast_dynamics, cfg.controller
        pv = physical_validity(X, obs, dict(Sigma=cfg.target.Sigma, center=cfg.target.center, epsilon=cfg.target.epsilon,
                                            d_c=f.d_c, epsilon_c=f.epsilon_c, eta=f.eta, d0=c.safety_distance))
        failed = [k for k, ok in pv["rules"].items() if not ok and k not in spec["exempt"]]
        base_flat = _flatten(yaml.safe_load(BASE.read_text())) if spec["source"] != "repro" else {}
        changed = {k: v for k, v in _flatten(raw).items() if base_flat.get(k) != v} if base_flat else {}
        entry = {k: spec[k] for k in ("id", "group", "pilot", "objective", "horizon_s", "init", "pairs", "exempt",
                                      "notes", "source", "legacy")}
        entry.update(num_agents=len(X), num_obstacles=len(obs), steps=cfg.simulation.steps, dt=cfg.simulation.dt,
                     config=str(path.relative_to(out)), config_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     changed_from_base=changed, calibration=spec["calibration"], calibration_result=cal_out,
                     physical_validity=pv, construction_rule_failures=failed,
                     construction_ok=not failed, initial_connectivity_ok=bool(pv["rules"]["lambda2_ge_eta"]))
        if isinstance(spec["init"], dict) and warm:
            from diagnostics.study48.warmstart import frozen_warmup
            fs, info = frozen_warmup(CertModel(cfg), X, spec["init"], max_steps=6000)
            wpath = out / "warm" / f"{spec['id']}_warm.pkl"
            pickle.dump({"warm": fs, **{k: v for k, v in info.items() if k != "log_every50"}}, open(wpath, "wb"))
            entry["warm_start"] = {"file": str(wpath.relative_to(out)), **{k: v for k, v in info.items()
                                                                           if k != "log_every50"}}
        manifest.append(entry)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    doc = {"study": "48-scenario feasibility study (plan v2)", "base_config": str(BASE.relative_to(REPO)),
           "scenarios": manifest}
    (out / "manifest.json").write_text(json.dumps(doc, indent=2, default=_json_default))
    return doc


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the 48 study configurations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args(argv)
    doc = generate(Path(a.out), warm=not a.no_warm)
    bad = [s["id"] for s in doc["scenarios"] if not s["construction_ok"]]
    print(f"[study48] generated {len(doc['scenarios'])} scenarios in {a.out}; construction-rule failures: {bad or 'none'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

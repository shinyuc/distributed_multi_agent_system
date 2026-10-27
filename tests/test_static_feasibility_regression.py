"""Regression of the static Level 1-3 checker on saved states of earlier experiments (read-only).

Skipped unless TCNS_RESEARCH_DATA points to the research data folder (tgt/, clfc/). Optionally writes a JSON
report to STATIC_REGRESSION_REPORT.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from diagnostics.certificate import CertModel
from diagnostics.static_feasibility import static_check

DATA = Path(os.environ["TCNS_RESEARCH_DATA"]) if os.environ.get("TCNS_RESEARCH_DATA") else None
pytestmark = pytest.mark.skipif(DATA is None or not DATA.exists(), reason="TCNS_RESEARCH_DATA not set")
REPORT: dict = {}


def _model(rel):
    return CertModel(load_config(str(DATA / rel)))


def _z_from_snapshot(rows, obs, k):
    z = np.zeros(rows.n - rows.nw)
    for i in range(rows.N):
        for j in range(rows.N):
            if j != i:
                z[rows.izp[(i, j)] - rows.nw] = obs["snap_z_pair"][k, i, j]
        z[rows.izc[i] - rows.nw] = obs["snap_z_c"][k, i]
        z[rows.izl[i] - rows.nw] = obs["snap_z_clf"][k, i]
    return z


def _failure_state(rel):
    d = json.load(open(DATA / rel))
    pos = [d["positions"][str(i)] for i in range(len(d["positions"]))]
    return d, pos


def _z_from_failure(rows, d):
    z = np.zeros(rows.n - rows.nw)
    for i in range(rows.N):
        fs = d["fast_states"][str(i)]
        for j, v in fs["z_pair_self"].items():
            z[rows.izp[(i, int(j))] - rows.nw] = v
        z[rows.izc[i] - rows.nw] = fs["z_c"]
        z[rows.izl[i] - rows.nw] = fs["z_clf"]
    return z


def _summ(r):
    return {"L1": (r.level1.verdict, r.level1.lp_margin), "L2": (r.level2.verdict, r.level2.lp_margin),
            "L2_aug": (r.level2_augmented.verdict, r.level2_augmented.lp_margin), "L2_repo": r.level2_repository,
            "lifted": r.level2_lifted, "L3": [(v.verdict, v.lp_margin) for v in (r.level3 or [])],
            "category": r.category, "consistency": r.consistency}


def test_A_gated_states_are_feasible():
    m = _model("tgt/A_gated.yaml")
    X = json.load(open(DATA / "tgt/run_A_gated/trajectories.json"))["trajectories"]
    obs = np.load(DATA / "tgt/obs_A_gated.npz")
    for k in (0, 1999):
        rows = m.rows(X[k])
        r = static_check(m, X[k], _z_from_snapshot(rows, obs, k))
        REPORT[f"A_gated_k{k}"] = _summ(r)
        assert (r.level1.verdict, r.level2.verdict, r.level3_verdict, r.category) == ("F", "F", "F", "feasible")
        assert not r.consistency


def test_A_communication_failure_zero_row_is_confirmed_infeasible():
    m = _model("tgt/A_communication.yaml")
    d1, pos = _failure_state("tgt/run_A_communication/failures/step0000001_agent1_infeasible.json")
    rows = m.rows(pos)
    r = static_check(m, pos, _z_from_failure(rows, d1))
    REPORT["A_communication_step1"] = _summ(r)
    assert r.level1.verdict == "F" and r.level2.verdict == "F" and r.category == "C"
    for agent in (1, 2):
        v = r.level3[agent]
        assert v.verdict == "I" and v.certificate["type"] == "zero_row"
        assert v.certificate["floor"] == 1e-9          # tau_0 binds; the cancellation floor is far smaller
    b_rec = d1["degenerate_violated"][0]["b"]          # recorded value -4.314e-8
    assert r.level3[1].certificate["beta"] == pytest.approx(b_rec, rel=1e-12)
    assert r.level2_repository["s_star"] == pytest.approx(d1["centralized"]["z_free"]["s_star"], rel=1e-12)
    assert r.level3[0].verdict == "F" and r.level3[3].verdict == "F"


@pytest.mark.parametrize("rel_cfg, rel_fail", [
    ("tgt/B_gated.yaml", "tgt/run_B_gated/failures/step0002149_agent1_infeasible.json"),
    ("tgt/B_communication.yaml", "tgt/run_B_communication/failures/step0000031_agent0_infeasible.json"),
])
def test_B_failures_are_transient_not_structural(rel_cfg, rel_fail):
    m = _model(rel_cfg)
    d, pos = _failure_state(rel_fail)
    rows = m.rows(pos)
    r = static_check(m, pos, _z_from_failure(rows, d))
    REPORT[Path(rel_fail).stem] = _summ(r) | {"recorded_s_star": d["s_star"], "recorded_class": d["class"]}
    assert r.level1.verdict == "F" and r.level2.verdict == "F"
    assert r.category in ("C", "U")
    assert r.level3[d["agent"]].verdict in ("I", "U")
    assert r.level2_repository["s_star"] == pytest.approx(d["centralized"]["z_free"]["s_star"], rel=1e-9)
    assert not [c for c in r.consistency if c.startswith("INCONSISTENT")]


def test_C_states_feasible_with_and_without_mismatch():
    m = _model("clfc/C_gated.yaml")
    X = json.load(open(DATA / "clfc/run_C_gated/trajectories.json"))["trajectories"]
    obs = np.load(DATA / "clfc/obs_C_gated.npz")
    for k in (0, 1000, 1999):
        rows = m.rows(X[k])
        for name, z in (("actual", _z_from_snapshot(rows, obs, k)), ("zero", np.zeros(rows.n - rows.nw))):
            r = static_check(m, X[k], z, repository_check=(name == "actual"))
            REPORT[f"C_k{k}_{name}"] = _summ(r)
            assert r.category == "feasible", (k, name, r.category)


def test_D3_zero_warm_and_recorded_mismatch():
    m = _model("clfc/D3_run.yaml")
    X = json.load(open(DATA / "clfc/run_D3/trajectories.json"))["trajectories"]
    snaps = pickle.load(open(DATA / "clfc/D3cl_run_obs.pkl", "rb"))["snaps"]
    warm = pickle.load(open(DATA / "clfc/D3cl_warm.pkl", "rb"))["warm"]
    rows0 = m.rows(X[0])
    r0 = static_check(m, X[0], np.zeros(rows0.n - rows0.nw))
    rw = static_check(m, X[0], rows0.to_vec(warm)[0][rows0.nw:])
    REPORT["D3_x0_zero"], REPORT["D3_x0_warm"] = _summ(r0), _summ(rw)
    assert r0.category == "C" and r0.level3[0].verdict == "I" and r0.level3[1].verdict == "F"
    assert set(r0.level3[0].certificate["support"]) == {"pairwise(i=0,j=1)", "obstacle(i=0,k=0)", "clf(i=0,active)"}
    assert rw.category == "feasible"
    phis, cats = [], []
    for k in range(100):
        rows = m.rows(X[k])
        ra = static_check(m, X[k], rows.to_vec(snaps[k])[0][rows.nw:], repository_check=False)
        rz = static_check(m, X[k], np.zeros(rows.n - rows.nw), repository_check=False)
        cats.append((ra.category, rz.category))
        phis.append(rz.level3[0].certificate["phi"])
    REPORT["D3_run"] = {"actual": sorted({c[0] for c in cats}), "zero": sorted({c[1] for c in cats}),
                        "zero_phi_range": [min(phis), max(phis)]}
    assert all(c == ("feasible", "C") for c in cats)
    assert min(phis) == pytest.approx(0.01522, abs=5e-6) and max(phis) == pytest.approx(0.02381, abs=5e-6)


def teardown_module(module):
    path = os.environ.get("STATIC_REGRESSION_REPORT")
    if path and REPORT:
        Path(path).write_text(json.dumps(REPORT, indent=2, default=str))

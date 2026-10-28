"""Automated post-processing of the 48-scenario dataset (diagnostics.study48.postprocess).

A miniature dataset is produced by the repository's 3-step server smoke batch (C09 sparse snapshots, A01 step-0
termination, D03 warm start) in a temporary directory; a strict-guard run is produced with a solver that wrongly
reports success. Data-gated tests re-analyse recorded runs of the earlier experiments (TCNS_RESEARCH_DATA).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import controller.distributed_controller as dcm
import diagnostics.recorder as rec
import simulation.simulator as sm
from controller.local_qp import LocalQPSolveResult
from diagnostics.study48 import run_pilot
from diagnostics.study48.postprocess import Options, analyze_scenario, load_inputs, run_postprocessing

REPO = Path(__file__).resolve().parents[1]
SMOKE_SPEC = REPO / "experiments/feasibility_study/pilot16/pilot/server_smoke_batch.yaml"
DATA = Path(os.environ["TCNS_RESEARCH_DATA"]) if os.environ.get("TCNS_RESEARCH_DATA") else None
needs_data = pytest.mark.skipif(DATA is None or not DATA.exists(), reason="TCNS_RESEARCH_DATA not set")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    base = tmp_path_factory.mktemp("pp")
    batch = base / "batch"
    batch.mkdir()
    r = subprocess.run([sys.executable, "-m", "diagnostics.run_batch", "--spec", str(SMOKE_SPEC), "--workers", "3",
                        "--out", str(batch)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ds = base / "ds"
    (ds / "scenarios").mkdir(parents=True)
    (ds / "superseded").mkdir()
    for sid in ("A01", "C09", "D03"):
        os.symlink(batch / "scenarios" / sid, ds / "scenarios" / sid)
    os.symlink(batch / "scenarios" / "D03", ds / "superseded" / "D03_original")   # must be ignored
    out = base / "out"
    res = run_postprocessing(ds, out)
    return {"base": base, "batch": batch, "ds": ds, "out": out, "res": res,
            "by": {x["id"]: x for x in res["results"]}, "rows": {x["id"]: x for x in res["rows"]}}


def _tree_hashes(root: Path) -> dict:
    out = {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            p = Path(dirpath) / f
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_outputs_and_dataset_selection(dataset):
    out, meta = dataset["out"], dataset["res"]["meta"]
    for f in ("summary.csv", "summary.json", "report.md", "analysis_meta.json", "scenarios/A01.json", "samples/D03.csv"):
        assert (out / f).exists(), f
    assert meta["selected"] == ["A01", "C09", "D03"] and len(meta["missing_from_dataset"]) == 45
    assert not meta["analysis_errors"] and not meta["not_in_manifest"]
    report = (out / "report.md").read_text()
    for phrase in ("not an exact-arithmetic proof", "does not show feasibility for later times",
                   "sufficient conditions", "not a proof that ρ = 0", "computed from the confirmed verdicts"):
        assert phrase in report


def test_step0_termination_is_certified_from_the_recorded_state(dataset):
    r = dataset["by"]["A01"]
    assert r["outcome"]["termination_kind"] == "local_qp_failure" and r["outcome"]["termination_step"] == 0
    assert r["outcome"]["failing_agents"] == [0, 1]
    term = r["key_states"]["termination"]
    assert term["source"].startswith("failure:") and "snapshot agrees with failure file" in term["notes"]
    ti = r["termination_interpretation"]
    assert ti["result"] == "certified_infeasible" and ti["category_at_termination"] == "A"
    assert ti["L1_at_termination"] == "I" and set(ti["failing_static_L3"].values()) == {"I"}
    assert all(c["certificate_type"] in ("farkas", "zero_row") and c["phi"] > 0 for c in ti["certificates"].values())
    assert not ti["agreement_flags"] and len(term["recorder_failures"]) == 2
    assert r["tier0_comparison"]["step0_match"] and r["tier0_comparison"]["category_match"]


def test_completed_warm_start_run_certificates_and_observed_errors(dataset):
    r = dataset["by"]["D03"]
    assert r["integrity"]["all_pass"] and r["outcome"]["termination_kind"] == "completed"
    assert r["termination_interpretation"]["result"] == "finite_horizon_completed"
    for norm in ("spectral", "frobenius"):
        ni = r["certificates"][norm]["results"]["non_increase"]
        assert ni["status"] == "CERT_OK" and ni["certified_states"] == 3
    assert r["certificates"]["spectral"]["k0"]["rho_status"] == "certified_positive"
    obs = r["observed_errors"]
    assert len(obs) == 3 and all(x["within_certified_radius"] for x in obs)
    assert r["trajectory"]["sampled_states"] == 3 and r["trajectory"]["categories"] == {"feasible": 3}
    assert r["key_states"]["last_qp_state"]["classification"]["category"] == "feasible"


def test_sparse_snapshots_are_reported_as_unavailable_not_interpolated(dataset):
    r = dataset["by"]["C09"]
    assert r["integrity"]["snapshot_every"] == 10
    assert r["trajectory"]["unavailable_states"] == [1, 2] and r["trajectory"]["sampled_states"] == 1
    last = r["key_states"]["last_qp_state"]
    assert "classification" not in last and "state unavailable" in last["notes"][0]
    assert "C09 last_qp_state (k=2): state unavailable" in (dataset["out"] / "report.md").read_text()


def test_dataset_is_read_only_and_output_must_be_outside(dataset, tmp_path):
    before = _tree_hashes(dataset["ds"])
    run_postprocessing(dataset["ds"], tmp_path / "again", ids=["A01"], opts=Options(certificates=False))
    assert _tree_hashes(dataset["ds"]) == before
    with pytest.raises(ValueError, match="outside the dataset"):
        run_postprocessing(dataset["ds"], dataset["ds"] / "analysis")
    with pytest.raises(ValueError, match="inside dataset run directory"):
        run_postprocessing(dataset["ds"], dataset["batch"] / "scenarios" / "A01" / "analysis")
    with pytest.raises(ValueError, match="new or empty"):
        run_postprocessing(dataset["ds"], dataset["out"])


def test_missing_termination_state_is_unresolved(dataset, tmp_path):
    sdir = tmp_path / "A01"
    shutil.copytree(dataset["batch"] / "scenarios" / "A01", sdir)
    shutil.rmtree(sdir / "run" / "failures")
    (sdir / "run" / "fast_snapshots.pkl").unlink()
    info = load_inputs()["scenarios"]["A01"]
    r = analyze_scenario(sdir, "A01", info, Options(certificates=False))
    assert r["termination_interpretation"]["result"] == "unresolved_state_unavailable"
    assert "classification" not in r["key_states"]["termination"]
    assert r["trajectory"]["sampled_states"] == 0


def test_changed_configuration_fails_integrity(dataset, tmp_path):
    sdir = tmp_path / "D03"
    shutil.copytree(dataset["batch"] / "scenarios" / "D03", sdir)
    cfg = yaml.safe_load((sdir / "scenario.yaml").read_text())
    cfg["controller"]["clf_rate"] = 0.3
    (sdir / "scenario.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = analyze_scenario(sdir, "D03", load_inputs()["scenarios"]["D03"], Options(certificates=False))
    st = {c["check"]: c["status"] for c in r["integrity"]["checks"]}
    assert st["ran config = repository config (parsed)"] == "FAIL" and not r["integrity"]["all_pass"]


def test_strict_guard_termination(tmp_path, monkeypatch):
    for obj, name in ((sm.Simulator, "initialize_fast_states"), (sm.Simulator, "_record_fast_history_snapshot"),
                      (rec.FeasibilityRecorder, "__call__"), (dcm, "build_local_qp_data")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr("sys.argv", ["pytest"])
    monkeypatch.setattr(dcm, "solve_local_qp_detailed",
                        lambda qp: LocalQPSolveResult((0.0, 0.0), True, "optimal", "optimal", 1, 0.0, 0.0, 0.0, 0.0))
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    raw["simulation"].update(dt=0.001, steps=2)
    raw["agents"].update(num_agents=2, initial_positions=[[-1.25, 0.25], [-1.25, -0.25]], init_mode="manual")
    raw["obstacles"]["obstacles"] = []
    raw["controller"].update(clf_coupling="gated", clf_rate=2.0, u_max=0.05)
    raw["fast_dynamics"].update(substeps=10)
    sdir = tmp_path / "X01"
    sdir.mkdir()
    (sdir / "scenario.yaml").write_text(yaml.safe_dump(raw))
    run_pilot.main(["--config", str(sdir / "scenario.yaml"), "--out", str(sdir / "run"), "--snapshot-every", "1"])
    monkeypatch.undo()
    r = analyze_scenario(sdir, "X01", None, Options(certificates=False))
    assert r["outcome"]["termination_kind"] == "strict_guard" and r["outcome"]["failing_agents"] == [0]
    ti = r["termination_interpretation"]
    assert ti["result"] == "certified_infeasible" and ti["category_at_termination"] == "A"
    assert r["key_states"]["termination"]["recorder_failures"][0]["class"] == "inconsistent"
    assert any("OSQP reported success" in f for f in ti["agreement_flags"])


def _recorded(tmp_path, cfg_rel, run_rel, sid):
    sdir = tmp_path / sid
    sdir.mkdir()
    shutil.copy(DATA / cfg_rel, sdir / "scenario.yaml")
    os.symlink(DATA / run_rel, sdir / "run")
    return sdir


@needs_data
def test_recorded_A_communication_failure_matches_the_verified_analysis(tmp_path):
    sdir = _recorded(tmp_path, "tgt/A_communication.yaml", "tgt/run_A_communication", "Acomm")
    r = analyze_scenario(sdir, "Acomm", None, Options(cert_max_states=2))
    ti = r["termination_interpretation"]
    assert r["outcome"]["termination_step"] == 1 and r["outcome"]["failing_agents"] == [1, 2]
    assert ti["result"] == "certified_infeasible" and ti["category_at_termination"] == "C"
    assert {a: c["certificate_type"] for a, c in ti["certificates"].items()} == {1: "zero_row", 2: "zero_row"}
    st = {c["check"]: c["status"] for c in r["integrity"]["checks"]}
    assert st["strict feasibility guard on"] == "NOT_RECORDED"                  # older run, no pilot runner
    assert r["trajectory"]["sampled_states"] == 0 and r["trajectory"]["unavailable_states"]


@needs_data
def test_recorded_A_gated_completed_run_is_certified(tmp_path):
    sdir = _recorded(tmp_path, "tgt/A_gated.yaml", "tgt/run_A_gated", "Agated")
    r = analyze_scenario(sdir, "Agated", None, Options(cert_max_states=20))
    assert r["outcome"]["termination_kind"] == "completed"
    c = r["certificates"]
    assert c["truncated"] and c["evaluated_states"] == 20
    for norm in ("spectral", "frobenius"):
        assert c[norm]["results"]["non_increase"]["status"] == "CERT_OK"
    assert "observed_errors" in r and r["observed_errors"] == []                # no snapshots recorded

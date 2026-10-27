"""The pilot runner (diagnostics.study48.run_pilot) and the batch runner option (no controller change).

Short in-process runs (2 steps) on synthetic configurations: the strict feasibility guard stops a run before a
control computed from an LP-infeasible QP is applied; the warm start is injected unchanged; the batch spec
resolves runner arguments and classifies a guard stop as a termination.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

import controller.distributed_controller as dcm
import diagnostics.recorder as rec
import simulation.simulator as sm
from config import load_config
from controller.local_qp import LocalQPSolveResult
from diagnostics.certificate import CertModel, zero_fast_state
from diagnostics.run_batch import load_spec, summarize_scenario
from diagnostics.study48 import run_pilot

REPO = Path(__file__).resolve().parents[1]
FAR2 = [[-1.25, 0.25], [-1.25, -0.25]]


def _cfg(tmp_path, positions, steps=2, **ctl):
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    raw["simulation"].update(dt=0.001, steps=steps)
    raw["agents"].update(num_agents=len(positions), initial_positions=positions, init_mode="manual")
    raw["obstacles"]["obstacles"] = []
    raw["controller"].update({"clf_coupling": "gated", **ctl})
    raw["fast_dynamics"].update(substeps=10)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.fixture
def restore(monkeypatch):
    """install_hooks patches classes in-process; register the originals so that they are restored."""
    for obj, name in ((sm.Simulator, "initialize_fast_states"), (sm.Simulator, "_record_fast_history_snapshot"),
                      (rec.FeasibilityRecorder, "__call__")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr("sys.argv", ["pytest"])
    return monkeypatch


def _fake_success(qp_data):
    return LocalQPSolveResult((0.0, 0.0), True, "optimal", "optimal", 1, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("strict", [True, False])
def test_strict_guard_stops_before_applying_a_control_from_an_infeasible_qp(tmp_path, restore, strict):
    cfg = _cfg(tmp_path, FAR2, clf_rate=2.0, u_max=0.05)          # every local QP infeasible (A01 type)
    restore.setattr(dcm, "solve_local_qp_detailed", _fake_success)  # a solver that wrongly reports success
    out = tmp_path / ("strict" if strict else "loose")
    run_pilot.main(["--config", str(cfg), "--out", str(out)] + ([] if strict else ["--no-strict"]))
    s = json.loads((out / "summary.json").read_text())
    traj = json.loads((out / "trajectories.json").read_text())["trajectories"]
    if strict:
        assert s["termination_reason"] == "exception" and s["termination_step"] == 0
        assert s["termination_exception"]["type"] == "StrictFeasibilityGuard"
        assert len(traj) == 1                                       # x_0 only: nothing applied
        row = summarize_scenario(out.parent / "x", {"status": "finished", "returncode": 1}, {})
        assert row["status"] == "error"                             # wrong directory: no summary
        sdir = tmp_path / "batch_like"
        (sdir / "run").mkdir(parents=True)
        (sdir / "run" / "summary.json").write_text((out / "summary.json").read_text())
        row = summarize_scenario(sdir, {"status": "finished", "returncode": 1}, {})
        assert row["status"] == "terminated" and row["failure"].startswith("strict feasibility guard")
    else:                                                           # the hazard the guard removes
        assert s["horizon_completed"] and len(traj) == 3 and s["class_counts"]["inconsistent"] == 4


def test_warm_start_is_injected_unchanged(tmp_path, restore):
    pos = [[-0.75, 0.0], [-0.25, 0.0]]
    cfg = _cfg(tmp_path, pos)
    model = CertModel(load_config(str(cfg)))
    warm = zero_fast_state(model)
    warm[0].z_clf, warm[1].z_clf = 1e-4, -1e-4                      # a recognisable, consistent fast state
    wf = tmp_path / "warm.pkl"
    pickle.dump({"warm": warm, "stop_step": 7}, open(wf, "wb"))
    out = tmp_path / "run"
    run_pilot.main(["--config", str(cfg), "--out", str(out), "--warm", str(wf), "--snapshot-every", "1"])
    snaps = pickle.load(open(out / "fast_snapshots.pkl", "rb"))["snaps"]
    rows = model.rows(pos)
    (p1, y1), (p2, y2) = rows.to_vec(snaps[0]), rows.to_vec(warm)
    assert np.array_equal(p1, p2) and np.array_equal(y1, y2)
    meta = json.loads((out / "warm_start.json").read_text())
    assert meta["stop_step"] == 7 and len(meta["sha256"]) == 64
    assert json.loads((out / "pilot_runner.json").read_text())["warm_start"] is True


def test_batch_spec_runner_and_path_arguments(tmp_path):
    (tmp_path / "w.pkl").write_bytes(b"x")
    spec = {"base_config": "configs/distributed_10_agents.yaml",
            "scenarios": [{"id": "a"},
                          {"id": "b", "runner": "diagnostics.study48.run_pilot",
                           "runner_args": ["--warm", {"path": "w.pkl"}, "--snapshot-every", 1]}]}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    sc = load_spec(p)["scenarios"]
    assert sc[0]["runner"] == "diagnostics.run_feasibility" and sc[0]["runner_args"] == []
    assert sc[1]["runner_args"] == ["--warm", str((tmp_path / "w.pkl").resolve()), "--snapshot-every", "1"]
    for bad, match in (({"runner": "os"}, "not in"), ({"runner_args": ["--warm", {"path": "missing.pkl"}]}, "not found"),
                       ({"runner_args": "--x"}, "must be a list")):
        q = copy.deepcopy(spec)
        q["scenarios"][1].update(bad)
        p.write_text(yaml.safe_dump(q))
        with pytest.raises((ValueError, FileNotFoundError), match=match):
            load_spec(p)

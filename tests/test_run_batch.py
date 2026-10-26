"""Parallel scenario runner: spec handling, isolation, failure handling, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from diagnostics.run_batch import apply_overrides, load_spec, run_batch, summarize_scenario


def _spec(tmp_path: Path, scenarios: list[dict], overrides: dict | None = None, name: str = "spec.yaml") -> Path:
    spec = {
        "base_config": "configs/distributed_10_agents.yaml",
        "overrides": {"simulation.steps": 6, "fast_dynamics.substeps": 2, **(overrides or {})},
        "scenarios": scenarios,
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec))
    return path


def test_apply_overrides_sets_nested_keys_without_touching_the_input():
    base = {"controller": {"safety_distance": 0.2}, "fast_dynamics": {"tau": 0.02}}
    out = apply_overrides(base, {"controller.safety_distance": 0.3, "fast_dynamics.substeps": 10})
    assert out == {"controller": {"safety_distance": 0.3}, "fast_dynamics": {"tau": 0.02, "substeps": 10}}
    assert base["controller"]["safety_distance"] == 0.2 and "substeps" not in base["fast_dynamics"]
    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides(base, {"no_such_section.x": 1})


@pytest.mark.parametrize(
    "scenarios,match",
    [
        ([{"id": "a"}, {"id": "a"}], "duplicate"),
        ([{"id": "bad/id"}], "must match"),
        ([{"id": "."}], "must match"),
        ([{"id": ".."}], "must match"),
        ([{"id": ".hidden"}], "must match"),
        ([{"id": "Case"}, {"id": "case"}], "duplicate"),
        ([{"id": "a", "base_config": "configs/does_not_exist.yaml"}], "not found"),
        ([], "no scenarios"),
    ],
)
def test_load_spec_rejects_invalid_specs(tmp_path, scenarios, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_spec(_spec(tmp_path, scenarios))


def test_batch_isolates_scenarios_and_reports_each_outcome(tmp_path):
    spec = _spec(tmp_path, [
        {"id": "ok", "overrides": {"controller.safety_distance": 0.30}},
        {"id": "infeasible_step0", "base_config": "configs/repro/step0_failure_2026-03-25.yaml",
         "overrides": {"simulation.dt": 0.001, "fast_dynamics.tau": 0.02}},
        {"id": "bad_key", "overrides": {"fast_dynamics.not_a_parameter": 1}},
    ])
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = {r["scenario_id"]: r for r in run_batch(spec, batch, workers=2)}

    assert rows["ok"]["status"] == "completed" and rows["ok"]["qp_solves"] == 6 * 5
    assert rows["ok"]["verified_feasible_full_horizon"] is True
    assert rows["ok"]["safety_distance"] == 0.30 and rows["ok"]["substeps"] == 2
    assert rows["infeasible_step0"]["status"] == "terminated"
    assert rows["infeasible_step0"]["termination_reason"] == "local_qp_failure"
    assert rows["infeasible_step0"]["qp_infeasible"] == 2
    assert rows["infeasible_step0"]["verified_feasible_full_horizon"] is False
    assert rows["bad_key"]["status"] == "error" and "not_a_parameter" in rows["bad_key"]["error_tail"]
    assert rows["bad_key"]["verified_feasible_full_horizon"] is False

    for sid in rows:
        sdir = batch / "scenarios" / sid
        assert {"scenario.yaml", "stdout.log", "stderr.log", "run"} <= {p.name for p in sdir.iterdir()}
    for sid in ("ok", "infeasible_step0"):
        meta = json.loads((batch / "scenarios" / sid / "run" / "run_meta.json").read_text())
        assert meta["config_path"] == str(batch / "scenarios" / sid / "scenario.yaml")
        assert meta["output_dir"] == str(batch / "scenarios" / sid / "run")
    assert (batch / "summary.csv").read_text().count("\n") == 1 + 3
    meta = json.loads((batch / "batch_meta.json").read_text())
    assert meta["status"] == "finished" and meta["workers"] == 2
    assert {s["result"] for s in meta["scenarios"].values()} == {"completed", "terminated", "error"}


def test_parallel_results_equal_sequential_results(tmp_path):
    scenarios = [{"id": "d030", "overrides": {"controller.safety_distance": 0.30}},
                 {"id": "eta0125", "overrides": {"fast_dynamics.eta": 0.125}},
                 {"id": "base"}]
    spec = _spec(tmp_path, scenarios, overrides={"simulation.steps": 12})
    out = {}
    for workers in (1, 3):
        batch = tmp_path / f"w{workers}"
        batch.mkdir()
        run_batch(spec, batch, workers=workers)
        out[workers] = batch
    for sc in scenarios:
        a = np.load(out[1] / "scenarios" / sc["id"] / "run" / "arrays.npz")
        b = np.load(out[3] / "scenarios" / sc["id"] / "run" / "arrays.npz")
        assert all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files)
        ta = json.loads((out[1] / "scenarios" / sc["id"] / "run" / "trajectories.json").read_text())
        tb = json.loads((out[3] / "scenarios" / sc["id"] / "run" / "trajectories.json").read_text())
        assert ta == tb


def test_timeout_kills_only_that_scenario(tmp_path):
    spec = _spec(tmp_path, [{"id": "slow", "overrides": {"simulation.steps": 100000}}, {"id": "fast"}])
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = {r["scenario_id"]: r for r in run_batch(spec, batch, workers=2, timeout=4.0)}
    assert rows["slow"]["status"] == "timeout" and rows["slow"]["returncode"] != 0
    assert rows["fast"]["status"] == "completed"


def test_invalid_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run_batch(_spec(tmp_path, [{"id": "a"}]), tmp_path, workers=0)


def test_run_batch_refuses_a_non_empty_directory(tmp_path):
    spec = _spec(tmp_path, [{"id": "a"}])  # tmp_path now holds the spec file
    with pytest.raises(ValueError, match="empty"):
        run_batch(spec, tmp_path, workers=1)


def _fake_run(tmp_path: Path, name: str, summary: dict | None) -> Path:
    sdir = tmp_path / name
    (sdir / "run").mkdir(parents=True)
    (sdir / "stderr.log").write_text("Traceback ...\nSomeError: boom\n")
    if summary is not None:
        (sdir / "run" / "summary.json").write_text(json.dumps(summary))
    return sdir


def _summary(**kw) -> dict:
    base = {"horizon_completed": True, "termination_reason": "completed", "steps_configured": 10,
            "steps_completed": 10, "qp_solves": 50, "class_counts": {"solved": 50},
            "verification": {"not_verified": 0, "verified_feasible_and_solved": 50},
            "tightest_lp_margin_of_solved_qps": {"s_star": -0.02, "step": 3, "agent": 0, "binding": []}}
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "summary,outcome,status,verified",
    [
        (_summary(), {"returncode": 0}, "completed", True),
        # Full horizon but one QP not verified (e.g. borderline): completed, NOT verified feasible.
        (_summary(class_counts={"solved": 49, "borderline": 1},
                  verification={"not_verified": 1, "verified_feasible_and_solved": 49}),
         {"returncode": 0}, "completed", False),
        # Early termination on an infeasible local QP: a simulation outcome, never "feasible".
        (_summary(horizon_completed=False, termination_reason="local_qp_failure", steps_completed=3,
                  qp_solves=20, class_counts={"solved": 18, "infeasible": 2},
                  verification={"not_verified": 2, "verified_feasible_and_solved": 18}),
         {"returncode": 1}, "terminated", False),
        (_summary(horizon_completed=False, termination_reason="numerical_guard", steps_completed=4, qp_solves=25,
                  verification={"not_verified": 0, "verified_feasible_and_solved": 25}),
         {"returncode": 1}, "terminated", False),
        # Unexpected exception inside the loop, crashed/killed worker, invalid config, timeout: infrastructure.
        (_summary(horizon_completed=False, termination_reason="exception", steps_completed=2, qp_solves=15,
                  termination_exception={"type": "KeyError", "message": "x"}),
         {"returncode": 1}, "error", False),
        (None, {"returncode": -9}, "error", False),
        (None, {"returncode": 1}, "error", False),
        (None, {"returncode": -9, "status": "timeout"}, "timeout", False),
    ],
)
def test_summary_distinguishes_outcomes(tmp_path, summary, outcome, status, verified):
    sdir = _fake_run(tmp_path, "s", summary)
    row = summarize_scenario(sdir, outcome, {"agents": {"num_agents": 5}, "simulation": {"steps": 10}})
    assert row["status"] == status
    assert row["verified_feasible_full_horizon"] is verified
    if status in ("error", "timeout"):
        assert row["failure"] and "SomeError" in row["error_tail"]
    if outcome["returncode"] == -9 and status == "error":
        assert row["failure"] == "process killed by signal 9"

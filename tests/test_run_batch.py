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


# ------------------------------------------------------------------ CPU allocation and workers
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import diagnostics.run_batch as rb  # noqa: E402
from diagnostics.run_batch import default_workers, detect_cpu_allocation  # noqa: E402


def _fs(tmp_path: Path, proc_cgroup: str | None, files: dict[str, str]) -> Path:
    root = tmp_path / "fsroot"
    if proc_cgroup is not None:
        (root / "proc" / "self").mkdir(parents=True)
        (root / "proc" / "self" / "cgroup").write_text(proc_cgroup)
    for rel, text in files.items():
        p = root / "sys" / "fs" / "cgroup" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _detect(root, logical=32, affinity=None):
    return detect_cpu_allocation(cpu_count=lambda: logical, affinity=lambda: affinity, fs_root=root)


@pytest.mark.parametrize(
    "proc_cgroup,files,logical,affinity,allocated,method",
    [
        # No restriction: all logical CPUs.
        (None, {}, 32, None, 32, "logical CPUs"),
        ("0::/user.slice\n", {"user.slice/cpu.max": "max 100000"}, 32, 32, 32, "logical CPUs"),
        # CPU affinity (taskset / numactl).
        (None, {}, 32, 8, 8, "CPU affinity"),
        # cgroup v2 quota on the process's own cgroup, on a parent, and inside a container namespace.
        ("0::/user.slice/job\n", {"user.slice/job/cpu.max": "400000 100000"}, 32, 32, 4, "cgroup v2"),
        ("0::/user.slice/job\n", {"user.slice/job/cpu.max": "max 100000", "user.slice/cpu.max": "250000 100000"},
         32, 32, 2, "cgroup v2"),
        ("0::/\n", {"cpu.max": "1200000 100000"}, 32, 32, 12, "cgroup v2"),
        # cgroup v1 CFS quota; -1 means unlimited.
        ("4:cpu,cpuacct:/docker/abc\n", {"cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "300000",
                                        "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000"}, 32, 32, 3, "cgroup v1"),
        ("4:cpu,cpuacct:/docker/abc\n", {"cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "-1",
                                        "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000"}, 32, 32, 32, "logical CPUs"),
        # The tightest restriction wins.
        ("0::/job\n", {"job/cpu.max": "400000 100000"}, 32, 6, 4, "cgroup v2"),
        ("0::/job\n", {"job/cpu.max": "800000 100000"}, 32, 6, 6, "CPU affinity"),
        # Fractional / sub-CPU quotas round down but never below one CPU.
        ("0::/job\n", {"job/cpu.max": "150000 100000"}, 32, None, 1, "cgroup v2"),
        ("0::/job\n", {"job/cpu.max": "50000 100000"}, 32, None, 1, "cgroup v2"),
        # Malformed files are ignored.
        ("0::/job\n", {"job/cpu.max": "garbage"}, 16, None, 16, "logical CPUs"),
        ("not a cgroup line\n", {}, 16, None, 16, "logical CPUs"),
    ],
)
def test_detect_cpu_allocation(tmp_path, proc_cgroup, files, logical, affinity, allocated, method):
    cpu = _detect(_fs(tmp_path, proc_cgroup, files), logical, affinity)
    assert cpu["allocated_cpus"] == allocated and cpu["method"].startswith(method)
    assert default_workers(cpu) == max(1, allocated - 1)


def test_detect_cpu_allocation_without_any_information(tmp_path):
    cpu = detect_cpu_allocation(cpu_count=lambda: None, affinity=lambda: None, fs_root=tmp_path / "missing")
    assert cpu["allocated_cpus"] == 1 and cpu["method"].startswith("fallback")
    assert default_workers(cpu) == 1


def test_detect_cpu_allocation_on_this_host():
    cpu = detect_cpu_allocation()
    assert 1 <= cpu["allocated_cpus"] <= (os.cpu_count() or 1)
    assert default_workers() >= 1


def test_explicit_workers_above_allocation_warn_but_are_kept(tmp_path, monkeypatch, capsys):
    fake = {"logical_cpus": 2, "affinity_cpus": None, "cgroup_cpu_quota": None, "cgroup_quota_source": None,
            "allocated_cpus": 2, "method": "logical CPUs (os.cpu_count)"}
    monkeypatch.setattr(rb, "detect_cpu_allocation", lambda: fake)
    spec = _spec(tmp_path, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = run_batch(spec, batch, workers=5)
    meta = json.loads((batch / "batch_meta.json").read_text())
    assert meta["workers"] == 5 and meta["workers_source"] == "explicit" and meta["cpu_allocation"] == fake
    assert len(meta["warnings"]) == 1 and "exceeds the detected CPU allocation of 2" in meta["warnings"][0]
    assert "WARNING" in capsys.readouterr().err
    assert {r["status"] for r in rows} == {"completed"}


def test_workers_within_allocation_do_not_warn(tmp_path, monkeypatch):
    fake = dict(logical_cpus=8, affinity_cpus=None, cgroup_cpu_quota=None, cgroup_quota_source=None,
                allocated_cpus=8, method="logical CPUs (os.cpu_count)")
    monkeypatch.setattr(rb, "detect_cpu_allocation", lambda: fake)
    batch = tmp_path / "batch"
    batch.mkdir()
    run_batch(_spec(tmp_path, [{"id": "a"}]), batch, workers=8)
    assert json.loads((batch / "batch_meta.json").read_text())["warnings"] == []


@pytest.mark.parametrize("cli_workers", [None, "3"])
def test_cli_records_automatic_or_explicit_workers(tmp_path, cli_workers):
    spec = _spec(tmp_path, [{"id": "a"}])
    out = tmp_path / "batch"
    cmd = [sys.executable, "-m", "diagnostics.run_batch", "--spec", str(spec), "--out", str(out)]
    if cli_workers:
        cmd += ["--workers", cli_workers]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    meta = json.loads((out / "batch_meta.json").read_text())
    assert meta["cpu_allocation"]["allocated_cpus"] >= 1 and meta["cpu_allocation"]["method"]
    if cli_workers is None:
        assert meta["workers_source"] == "automatic"
        assert meta["workers"] == default_workers(detect_cpu_allocation())
    else:
        assert meta["workers_source"] == "explicit" and meta["workers"] == 3


def test_children_use_builtin_osqp_even_if_parent_env_names_another_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OSQP_ALGEBRA_BACKEND", "mkl")  # not installed: would break `import osqp` in a child
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = run_batch(_spec(tmp_path, [{"id": "a"}]), batch, workers=1)
    assert rows[0]["status"] == "completed"
    run_meta = json.loads((batch / "scenarios" / "a" / "run" / "run_meta.json").read_text())
    assert run_meta["qp_solver_call"]["algebra"] == "builtin"
    assert json.loads((batch / "batch_meta.json").read_text())["child_solver_env"] == {"OSQP_ALGEBRA_BACKEND": "builtin"}

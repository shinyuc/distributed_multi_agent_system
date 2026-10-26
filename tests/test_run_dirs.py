"""Output directories are unique per run and existing results are never overwritten."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

import utils.run_dirs as run_dirs
from controller.local_qp import local_qp_solver_metadata
from utils.init_generator import save_initialization_record
from utils.run_dirs import OutputDirExistsError, make_unique_run_dir, prepare_explicit_dir

REPO = Path(__file__).resolve().parents[1]
STEP0 = "configs/repro/step0_failure_2026-03-25.yaml"
FIXED_NOW = datetime(2026, 10, 20, 12, 0, 0, 123456)


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _make_in_process(root: str) -> str:
    return str(make_unique_run_dir(root, "cfg", now=FIXED_NOW))


# ------------------------------------------------------------------ unit level
def test_same_instant_names_are_unique(tmp_path):
    with ThreadPoolExecutor(max_workers=32) as pool:
        dirs = list(pool.map(lambda _: make_unique_run_dir(tmp_path, "cfg", now=FIXED_NOW), range(300)))
    assert len({d.name for d in dirs}) == 300
    assert all(d.is_dir() and not any(d.iterdir()) for d in dirs)
    assert all(d.name.startswith("20261020_120000_123456_cfg_p") for d in dirs)


def test_concurrent_processes_get_distinct_dirs(tmp_path):
    with ProcessPoolExecutor(max_workers=8) as pool:
        dirs = list(pool.map(_make_in_process, [str(tmp_path)] * 64))
    assert len(set(dirs)) == 64


def test_forced_name_collision_retries(tmp_path, monkeypatch):
    suffixes = iter(["aaaa", "aaaa", "bbbb"])
    monkeypatch.setattr(run_dirs, "_suffix", lambda: next(suffixes))
    first = make_unique_run_dir(tmp_path, "cfg", now=FIXED_NOW)
    (first / "result.txt").write_text("first run")
    second = make_unique_run_dir(tmp_path, "cfg", now=FIXED_NOW)
    assert first.name.endswith("_aaaa") and second.name.endswith("_bbbb")
    assert (first / "result.txt").read_text() == "first run"


def test_label_is_sanitized(tmp_path):
    d = make_unique_run_dir(tmp_path, "configs/repro/step 0 (copy)", now=FIXED_NOW)
    assert d.parent == tmp_path and "/" not in d.name and " " not in d.name


def test_explicit_dir_new_or_empty_is_accepted(tmp_path):
    new, moved = prepare_explicit_dir(tmp_path / "a" / "b")
    assert new.is_dir() and moved is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare_explicit_dir(empty) == (empty, None)


def test_explicit_nonempty_dir_is_refused(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "summary.json").write_text('{"old": true}')
    with pytest.raises(OutputDirExistsError):
        prepare_explicit_dir(d)
    assert (d / "summary.json").read_text() == '{"old": true}'


def test_overwrite_moves_old_results_aside(tmp_path):
    d = tmp_path / "run"
    (d / "failures").mkdir(parents=True)
    (d / "summary.json").write_text("old summary")
    (d / "failures" / "step1_agent0_infeasible.json").write_text("old failure")
    before = _snapshot(d)
    new, moved = prepare_explicit_dir(d, overwrite=True)
    assert new == d and not any(d.iterdir())  # fresh: no stale files can mix in
    assert moved is not None and moved.name.startswith("run.replaced_")
    assert _snapshot(moved) == before  # nothing deleted


def test_init_records_do_not_collide(tmp_path):
    record = {"mode": "manual", "seed": 42}

    class Region:
        save_dir = str(tmp_path)

    paths = {save_initialization_record(record=record, region=Region(), config_path="cfg.yaml") for _ in range(50)}
    assert len(paths) == 50


# ------------------------------------------------------------------ end to end
def _run(module_args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *module_args], cwd=REPO, capture_output=True, text=True)


def _runner(*args: str) -> subprocess.CompletedProcess:
    return _run(["-m", "diagnostics.run_feasibility", "--config", STEP0, "--progress-every", "0", *args])


def test_runner_two_runs_two_dirs_first_preserved(tmp_path):
    p1 = _runner("--output-root", str(tmp_path))
    first = [d for d in tmp_path.iterdir()]
    assert p1.returncode == 1 and len(first) == 1  # step-0 case terminates early (exit 1)
    snap = _snapshot(first[0])
    assert {"summary.json", "run_meta.json", "progress.log", "initial_conditions.json"} <= set(snap)
    p2 = _runner("--output-root", str(tmp_path))
    dirs = sorted(tmp_path.iterdir())
    assert p2.returncode == 1 and len(dirs) == 2
    assert _snapshot(first[0]) == snap
    for d in dirs:
        assert d.name.split("_", 3)[3].startswith("step0_failure_2026-03-25_p")
        meta = json.loads((d / "run_meta.json").read_text())
        assert meta["status"] == "finished" and meta["output_dir"] == str(d)
        assert meta["qp_solver_call"] == json.loads(json.dumps(local_qp_solver_metadata()))


def test_runner_refuses_existing_results(tmp_path):
    p1 = _runner("--out", str(tmp_path / "run"))
    assert p1.returncode == 1
    snap = _snapshot(tmp_path / "run")
    p2 = _runner("--out", str(tmp_path / "run"))
    assert p2.returncode == 2 and "already contains results" in p2.stderr
    assert _snapshot(tmp_path / "run") == snap


def test_runner_overwrite_keeps_previous_copy(tmp_path):
    _runner("--out", str(tmp_path / "run"))
    snap = _snapshot(tmp_path / "run")
    p2 = _runner("--out", str(tmp_path / "run"), "--overwrite")
    assert p2.returncode == 1
    moved = [d for d in tmp_path.iterdir() if d.name.startswith("run.replaced_")]
    assert len(moved) == 1 and _snapshot(moved[0]) == snap
    meta = json.loads((tmp_path / "run" / "run_meta.json").read_text())
    assert meta["previous_contents_moved_to"] == str(moved[0])


def test_main_two_runs_two_dirs_and_refusal(tmp_path):
    p1 = _run(["main.py", "--config", STEP0, "--output-root", str(tmp_path)])
    p2 = _run(["main.py", "--config", STEP0, "--output-root", str(tmp_path)])
    dirs = sorted(tmp_path.iterdir())
    assert p1.returncode == p2.returncode == 1 and len(dirs) == 2
    for d in dirs:
        assert {"run_meta.json", "run_report.json", "trajectories.json", "initial_conditions.json"} <= {
            p.name for p in d.iterdir()}
        meta = json.loads((d / "run_meta.json").read_text())
        assert meta["status"] == "finished"
        assert meta["local_qp_solver"] == json.loads(json.dumps(local_qp_solver_metadata()))
    snap = _snapshot(dirs[0])
    p3 = _run(["main.py", "--config", STEP0, "--out", str(dirs[0])])
    assert p3.returncode == 2 and "already contains results" in p3.stderr
    assert _snapshot(dirs[0]) == snap

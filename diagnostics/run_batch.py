"""Run many independent scenarios in parallel, one OS process per scenario.

Usage:
    python -m diagnostics.run_batch --spec batch.yaml [--workers N] [--timeout SEC]
                                    [--output-root DIR | --out DIR [--overwrite]]

Every scenario runs the unchanged feasibility runner (``python -m diagnostics.run_feasibility``)
in its own child process, so each scenario keeps the full per-QP feasibility diagnostics
(independent LP checks included) and a crash, timeout or error in one scenario only affects
that scenario. At most ``--workers`` scenarios run at a time (default: the detected CPU
allocation - 1; see detect_cpu_allocation). Children are started with BLAS / OpenMP thread
pools limited to one thread so that N workers use about N cores, and with the builtin OSQP
backend.

Batch spec (YAML):

    name: pairwise_sweep                                   # optional, used in the directory name
    base_config: configs/distributed_10_agents.yaml        # default base configuration
    overrides:                                             # optional, applied to every scenario
      simulation.steps: 10000
      fast_dynamics.substeps: 10
    scenarios:
      - id: d030                                           # unique, [A-Za-z0-9._-]
        overrides: {controller.safety_distance: 0.30}
      - id: eta0125
        base_config: configs/default_5_agents.yaml         # optional per-scenario base
        overrides: {fast_dynamics.eta: 0.125}

Overrides use dotted keys; a key may add a new leaf (e.g. ``fast_dynamics.substeps``) but its
parent section must already exist. Scenario overrides are applied after the batch-wide ones.
Relative ``base_config`` paths are resolved against the spec file's directory, then the
repository root.

Output (the batch directory is created exclusively, see utils.run_dirs):

    <batch dir>/batch_spec.yaml       copy of the spec
    <batch dir>/batch_meta.json       workers, thread limits, versions, git commit, timing, statuses
    <batch dir>/summary.json, summary.csv
    <batch dir>/scenarios/<id>/scenario.yaml   merged configuration that was run
                              stdout.log, stderr.log
                              run/            run_feasibility output directory

summary.json / summary.csv hold one row per scenario: status (completed = full horizon simulated;
terminated = the simulation stopped early on a local QP failure or the numerical guard; error =
invalid configuration, crashed or killed worker, or an unexpected exception; timeout, interrupted,
not_started), verified_feasible_full_horizon (full horizon simulated AND every local QP solved and
confirmed feasible by the independent LP; False for every other outcome),
configuration parameters, QP counts by verification class, the tightest LP margin over the
QPs that OSQP solved (infeasible QPs are counted in qp_infeasible instead), safety and
connectivity minima, the largest recorded mismatch coupling terms, runtimes and paths.

Children run with the repository as working directory, exactly like a manual run, so
initialization records still go to the shared outputs/init_conditions directory (file names
are unique and created exclusively); each run copies its record into run/initial_conditions.json.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller.local_qp import local_qp_solver_metadata
from utils.run_dirs import OutputDirExistsError, make_unique_run_dir, prepare_explicit_dir

REPO = Path(__file__).resolve().parents[1]
# One thread per numerical library in every child: N workers -> about N busy cores.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
# OSQP reads OSQP_ALGEBRA_BACKEND at import time; a value naming a backend that is not installed
# makes `import osqp` fail in the child. Children always use the builtin backend that
# controller.local_qp selects explicitly.
CHILD_SOLVER_ENV = {"OSQP_ALGEBRA_BACKEND": "builtin"}
# Starts with a letter or digit, so "." and ".." (and hidden names) are impossible.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Configuration values copied into the summary for every scenario.
SUMMARY_PARAMS = {
    "dt": ("simulation", "dt"),
    "steps": ("simulation", "steps"),
    "n_agents": ("agents", "num_agents"),
    "substeps": ("fast_dynamics", "substeps"),
    "tau": ("fast_dynamics", "tau"),
    "xi": ("fast_dynamics", "xi"),
    "eta": ("fast_dynamics", "eta"),
    "d_c": ("fast_dynamics", "d_c"),
    "epsilon_c": ("fast_dynamics", "epsilon_c"),
    "safety_distance": ("controller", "safety_distance"),
    "cbf_rate": ("controller", "cbf_rate"),
    "clf_rate": ("controller", "clf_rate"),
    "u_max": ("controller", "u_max"),
    "alpha_clf": ("controller", "alpha_clf"),
    "alpha_connectivity": ("controller", "alpha_connectivity"),
    "enable_connectivity_constraint": ("controller", "enable_connectivity_constraint"),
    "clf_coupling": ("controller", "clf_coupling"),
}


# ------------------------------------------------------------------ batch spec
def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with dotted-key ``overrides`` applied."""
    out = copy.deepcopy(config)
    for dotted, value in (overrides or {}).items():
        keys = str(dotted).split(".")
        node = out
        for depth, key in enumerate(keys[:-1]):
            if not isinstance(node.get(key), dict):
                raise ValueError(f"override {dotted!r}: section {'.'.join(keys[:depth + 1])!r} does not exist")
            node = node[key]
        node[keys[-1]] = copy.deepcopy(value)
    return out


def _resolve_config_path(path: str, spec_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (spec_dir, REPO):
        if (base / p).exists():
            return (base / p).resolve()
    raise FileNotFoundError(f"base_config {path!r} not found relative to {spec_dir} or {REPO}")


def load_spec(spec_path: str | Path) -> dict[str, Any]:
    """Read a batch spec and return {name, scenarios: [{id, base_config, overrides, config}]}."""
    spec_path = Path(spec_path)
    raw = yaml.safe_load(spec_path.read_text()) or {}
    scenarios = raw.get("scenarios") or []
    if not scenarios:
        raise ValueError("batch spec has no scenarios")
    seen: set[str] = set()
    out = []
    for k, sc in enumerate(scenarios):
        sid = str(sc.get("id", ""))
        if not _ID_RE.match(sid):
            raise ValueError(f"scenario #{k}: id {sid!r} must match [A-Za-z0-9][A-Za-z0-9._-]*")
        # Compared case-insensitively: on case-insensitive file systems (macOS default)
        # "A" and "a" would be the same scenario directory.
        if sid.lower() in seen:
            raise ValueError(f"duplicate scenario id {sid!r} (ids are compared case-insensitively)")
        seen.add(sid.lower())
        base = sc.get("base_config", raw.get("base_config"))
        if not base:
            raise ValueError(f"scenario {sid!r}: no base_config")
        base_path = _resolve_config_path(base, spec_path.parent)
        overrides = {**(raw.get("overrides") or {}), **(sc.get("overrides") or {})}
        config = apply_overrides(yaml.safe_load(base_path.read_text()) or {}, overrides)
        out.append({"id": sid, "base_config": str(base_path), "overrides": overrides, "config": config})
    return {"name": str(raw.get("name") or spec_path.stem), "scenarios": out}


# ------------------------------------------------------------------ summary
def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _max_abs(a: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))) if a.size else None


def summarize_scenario(sdir: Path, outcome: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Collect one scenario's results from the files its run_feasibility process wrote."""
    run = sdir / "run"
    row: dict[str, Any] = {"scenario_id": sdir.name, "output_dir": str(run)}
    row.update({name: _get(config, *path) for name, path in SUMMARY_PARAMS.items()})
    row["returncode"] = outcome.get("returncode")
    row["batch_wall_time_s"] = outcome.get("wall_time_s")
    row["verified_feasible_full_horizon"] = False
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    report = json.loads((run / "run_report.json").read_text()) if (run / "run_report.json").exists() else None
    if outcome.get("status") in ("timeout", "interrupted", "not_started"):
        row["status"] = outcome["status"]
    elif summary is None or summary.get("run_error") or summary.get("termination_reason") == "exception":
        # No result, or an unexpected exception (not a QP failure or numerical guard) ended the run.
        row["status"] = "error"
    elif summary.get("horizon_completed"):
        row["status"] = "completed"
    else:
        row["status"] = "terminated"
    rc = outcome.get("returncode")
    if row["status"] == "timeout":
        row["failure"] = "killed after exceeding the per-scenario timeout"
    elif row["status"] == "error":
        if isinstance(rc, int) and rc < 0:
            row["failure"] = f"process killed by signal {-rc}"
        elif summary is None:
            row["failure"] = f"exit code {rc}, no simulation summary (see error_tail / stderr.log)"
        elif summary.get("run_error"):
            row["failure"] = "exception outside the simulation loop (see summary run_error)"
        else:
            exc = summary.get("termination_exception") or {}
            row["failure"] = f"unexpected exception inside the simulation loop: {exc.get('type')}: {exc.get('message')}"
    if row["status"] in ("error", "timeout") and (sdir / "stderr.log").exists():
        row["error_tail"] = "\n".join((sdir / "stderr.log").read_text(errors="replace").splitlines()[-15:])
    if summary is None:
        return row
    # Use the values the run resolved (defaults included) where the runner recorded them.
    for name in ("clf_coupling",):
        resolved = _get(summary, "run_meta", name)
        if resolved is not None:
            row[name] = resolved
    cc = summary.get("class_counts", {})
    tight = summary.get("tightest_lp_margin_of_solved_qps") or {}
    first_ns = summary.get("first_non_solved") or {}
    row.update({
        "termination_reason": summary.get("termination_reason"),
        "termination_step": summary.get("termination_step"),
        "verdict": summary.get("verdict"),
        "steps_completed": summary.get("steps_completed"),
        "steps_configured": summary.get("steps_configured"),
        "horizon_completed": summary.get("horizon_completed"),
        "qp_solves": summary.get("qp_solves"),
        **{f"qp_{c}": cc.get(c) for c in ("solved", "solved_inaccurate", "borderline", "infeasible",
                                         "numerical_failure", "inconsistent", "lp_error")},
        "qp_not_verified": _get(summary, "verification", "not_verified"),
        "first_non_solved_step": first_ns.get("step"),
        "first_non_solved_agent": first_ns.get("agent"),
        "first_non_solved_class": first_ns.get("class"),
        "tightest_lp_margin": tight.get("s_star") if tight.get("step") is not None else None,
        "tightest_lp_margin_step": tight.get("step"),
        "tightest_lp_margin_agent": tight.get("agent"),
        "tightest_lp_margin_binding": ";".join(tight.get("binding") or []),
        "max_normalized_violation": _get(summary, "max_violation_of_solved_qps", "normalized"),
        "min_weighted_lambda2": summary.get("min_lambda2"),
        "sim_wall_time_s": summary.get("run_wall_time_s"),
    })
    # True only if the full requested horizon was simulated and every local QP of every step
    # was solved and confirmed feasible by the independent LP.
    n_agents = row.get("n_agents")
    expected_qps = summary.get("steps_configured") * n_agents if isinstance(n_agents, int) and isinstance(
        summary.get("steps_configured"), int) else None
    row["verified_feasible_full_horizon"] = bool(
        row["status"] == "completed"
        and summary.get("horizon_completed") is True
        and summary.get("steps_completed") == summary.get("steps_configured")
        and summary.get("qp_solves") == expected_qps
        and _get(summary, "verification", "not_verified") == 0
        and _get(summary, "verification", "verified_feasible_and_solved") == summary.get("qp_solves")
    )
    if report is not None:
        row.update({
            "min_pairwise_distance": _get(report, "safety", "min_inter_agent_distance"),
            "inter_agent_violation": _get(report, "safety", "inter_agent_violation"),
            "min_obstacle_clearance": _get(report, "safety", "min_obstacle_clearance"),
            "obstacle_violation": _get(report, "safety", "obstacle_violation"),
            "min_lambda2_comm_binary": _get(report, "connectivity", "min_lambda2_comm"),
            "min_lambda2_cons_binary": _get(report, "connectivity", "min_lambda2_cons"),
        })
    # Mismatch coupling terms as recorded by the feasibility recorder (QP right-hand-side terms).
    if (run / "arrays.npz").exists():
        arr = np.load(run / "arrays.npz", allow_pickle=False)
        row["max_abs_conn_coupling"] = _max_abs(arr["conn_mismatch"])
        row["max_abs_clf_coupling"] = _max_abs(arr["clf_mismatch"])
    if (run / "qp_log.csv.gz").exists():
        with gzip.open(run / "qp_log.csv.gz", "rt") as f:
            vals = [float(r["pair_mismatch_maxabs"]) for r in csv.DictReader(f) if r["pair_mismatch_maxabs"] not in ("", "nan")]
        row["max_abs_pair_coupling"] = max(vals) if vals else None
    return row


def write_summary(batch_dir: Path, rows: list[dict[str, Any]], overrides: dict[str, dict]) -> None:
    for r in rows:
        r["overrides"] = json.dumps(overrides.get(r["scenario_id"], {}), sort_keys=True)
    with (batch_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols and c != "error_tail"]
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ CPU allocation
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _ancestors(base: Path, rel: str) -> list[Path]:
    """``base/rel`` and each of its parents up to ``base`` that exist (a cgroup limit applies to all descendants)."""
    parts = [p for p in rel.strip("/").split("/") if p]
    return [d for d in (base.joinpath(*parts[:k]) for k in range(len(parts), -1, -1)) if d.is_dir()]


def _cgroup_cpu_limit(fs_root: Path) -> tuple[float | None, str | None]:
    """Smallest CPU quota (in CPUs) over this process's cgroups (v2 or v1); (None, None) if none applies."""
    base = fs_root / "sys" / "fs" / "cgroup"
    found: list[tuple[float, str]] = []
    for line in (_read(fs_root / "proc" / "self" / "cgroup") or "").splitlines():
        try:
            hierarchy, controllers, rel = line.split(":", 2)
        except ValueError:
            continue
        if hierarchy == "0" and controllers == "":  # cgroup v2 (unified): cpu.max = "<quota> <period>" or "max ..."
            for d in _ancestors(base, rel):
                fields = (_read(d / "cpu.max") or "").split()
                if len(fields) == 2 and fields[0] != "max":
                    try:
                        found.append((int(fields[0]) / int(fields[1]), f"cgroup v2 quota ({d / 'cpu.max'})"))
                    except (ValueError, ZeroDivisionError):
                        pass
        elif "cpu" in controllers.split(","):  # cgroup v1 cpu controller: cfs quota/period, quota -1 = unlimited
            mounts = dict.fromkeys(base / name for name in (controllers, "cpu", "cpu,cpuacct", "cpuacct,cpu"))
            for mount in mounts:
                if not mount.is_dir():
                    continue
                for d in _ancestors(mount, rel):
                    quota, period = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
                    try:
                        if quota is not None and period is not None and int(quota) > 0:
                            found.append((int(quota) / int(period), f"cgroup v1 quota ({d / 'cpu.cfs_quota_us'})"))
                    except (ValueError, ZeroDivisionError):
                        pass
    return min(found) if found else (None, None)


def _affinity_cpus() -> int | None:
    if hasattr(os, "sched_getaffinity"):  # Linux
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            return None
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13 (affinity-aware where the OS supports it)
        return os.process_cpu_count()
    return None


def detect_cpu_allocation(*, cpu_count=os.cpu_count, affinity=_affinity_cpus, fs_root: str | Path = "/") -> dict[str, Any]:
    """CPUs this process may use: the smallest of logical CPUs, CPU affinity and cgroup quota.

    Each source is optional (e.g. no affinity API on macOS, no cgroups outside Linux); with no
    information at all the allocation is 1. A fractional quota is rounded down, but never below 1.
    """
    logical = cpu_count()
    aff = affinity()
    quota, quota_source = _cgroup_cpu_limit(Path(fs_root))
    candidates = [(float(v), method) for v, method in (
        (logical, "logical CPUs (os.cpu_count)"),
        (aff, "CPU affinity"),
        (quota, quota_source),
    ) if v]
    if candidates:
        limit, method = min(candidates, key=lambda c: c[0])
        allocated = max(1, math.floor(limit))
    else:
        allocated, method = 1, "fallback: no CPU information available"
    return {
        "logical_cpus": logical,
        "affinity_cpus": aff,
        "cgroup_cpu_quota": quota,
        "cgroup_quota_source": quota_source,
        "allocated_cpus": allocated,
        "method": method,
    }


def default_workers(cpu: dict[str, Any] | None = None) -> int:
    """One worker per allocated CPU, leaving one for the system and the batch process (at least 1)."""
    cpu = cpu if cpu is not None else detect_cpu_allocation()
    return max(1, cpu["allocated_cpus"] - 1)


# ------------------------------------------------------------------ execution
def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.update(CHILD_SOLVER_ENV)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), *filter(None, [env.get("PYTHONPATH")])])
    return env


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_batch(spec_path: str | Path, batch_dir: Path, workers: int, timeout: float | None = None,
              poll_interval: float = 0.2, workers_source: str = "explicit") -> list[dict[str, Any]]:
    """Run every scenario of ``spec_path`` into ``batch_dir`` (must exist and be empty).

    ``workers`` is used as given. If it exceeds the detected CPU allocation a warning is printed
    and recorded in batch_meta.json, but the value is not changed.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    batch_dir = Path(batch_dir)
    if not batch_dir.is_dir() or any(batch_dir.iterdir()):
        raise ValueError(f"batch directory must be an existing, empty directory: {batch_dir}")
    spec = load_spec(spec_path)
    (batch_dir / "batch_spec.yaml").write_text(Path(spec_path).read_text())
    env = _child_env()
    cpu = detect_cpu_allocation()
    batch_warnings: list[str] = []
    if workers > cpu["allocated_cpus"]:
        batch_warnings.append(
            f"workers={workers} exceeds the detected CPU allocation of {cpu['allocated_cpus']} "
            f"({cpu['method']}); scenarios will share CPUs. The requested value is used unchanged.")
        print(f"[batch] WARNING: {batch_warnings[-1]}", file=sys.stderr, flush=True)
    meta: dict[str, Any] = {
        "status": "running",
        "spec": str(Path(spec_path).resolve()),
        "batch_name": spec["name"],
        "workers": workers,
        "workers_source": workers_source,
        "cpu_allocation": cpu,
        "warnings": batch_warnings,
        "timeout_s": timeout,
        "child_thread_env": THREAD_ENV,
        "child_solver_env": CHILD_SOLVER_ENV,
        "python": platform.python_version(),
        "local_qp_solver": local_qp_solver_metadata(),
        "git_commit": _git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "scenarios": {},
    }

    def write_meta() -> None:
        with (batch_dir / "batch_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    for sc in spec["scenarios"]:
        sdir = batch_dir / "scenarios" / sc["id"]
        sdir.mkdir(parents=True, exist_ok=False)
        with (sdir / "scenario.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(sc["config"], f, sort_keys=False)
        meta["scenarios"][sc["id"]] = {"status": "pending", "base_config": sc["base_config"],
                                       "overrides": sc["overrides"]}
    write_meta()

    pending = deque(spec["scenarios"])
    running: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    t_batch = time.perf_counter()
    try:
        while pending or running:
            while pending and len(running) < workers:
                sc = pending.popleft()
                sdir = batch_dir / "scenarios" / sc["id"]
                out_f = open(sdir / "stdout.log", "w", encoding="utf-8")
                err_f = open(sdir / "stderr.log", "w", encoding="utf-8")
                cmd = [sys.executable, "-m", "diagnostics.run_feasibility",
                       "--config", str(sdir / "scenario.yaml"), "--out", str(sdir / "run")]
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out_f, stderr=err_f,
                                        stdin=subprocess.DEVNULL)
                running[sc["id"]] = {"proc": proc, "t0": time.perf_counter(), "files": (out_f, err_f)}
                meta["scenarios"][sc["id"]].update(status="running", pid=proc.pid,
                                                   started_at=datetime.now().isoformat(timespec="seconds"))
                write_meta()
                print(f"[batch] started {sc['id']} (pid {proc.pid}); running={len(running)} pending={len(pending)}",
                      file=sys.stderr, flush=True)
            time.sleep(poll_interval)
            for sid in list(running):
                job = running[sid]
                rc = job["proc"].poll()
                elapsed = time.perf_counter() - job["t0"]
                timed_out = rc is None and timeout is not None and elapsed > timeout
                if rc is None and not timed_out:
                    continue
                if timed_out:
                    job["proc"].kill()
                    rc = job["proc"].wait()
                for f in job["files"]:
                    f.close()
                outcomes[sid] = {"returncode": rc, "wall_time_s": elapsed,
                                 "status": "timeout" if timed_out else "finished"}
                meta["scenarios"][sid].update(status=outcomes[sid]["status"], returncode=rc,
                                              wall_time_s=elapsed)
                del running[sid]
                write_meta()
                print(f"[batch] finished {sid}: returncode={rc} wall={elapsed:.1f}s"
                      + (" (TIMEOUT, killed)" if timed_out else ""), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        for sid, job in running.items():
            job["proc"].kill()
            job["proc"].wait()
            for f in job["files"]:
                f.close()
            outcomes[sid] = {"returncode": None, "wall_time_s": None, "status": "interrupted"}
            meta["scenarios"][sid].update(status="interrupted")
        meta["status"] = "interrupted"
        write_meta()
        raise
    finally:
        rows = []
        for sc in spec["scenarios"]:
            outcome = outcomes.get(sc["id"], {"status": "not_started"})
            rows.append(summarize_scenario(batch_dir / "scenarios" / sc["id"], outcome, sc["config"]))
            meta["scenarios"][sc["id"]]["result"] = rows[-1]["status"]
        write_summary(batch_dir, rows, {sc["id"]: sc["overrides"] for sc in spec["scenarios"]})
        if meta["status"] == "running":
            meta["status"] = "finished"
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["batch_wall_time_s"] = time.perf_counter() - t_batch
        write_meta()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="batch spec YAML")
    parser.add_argument("--workers", type=int, default=None,
                        help="maximum number of scenarios running at once (default: detected CPU allocation "
                             "- 1, considering logical CPUs, CPU affinity and cgroup quotas; benchmark to tune)")
    parser.add_argument("--timeout", type=float, default=None, help="per-scenario wall-clock limit in seconds")
    parser.add_argument("--output-root", default="outputs/batches",
                        help="parent of the unique batch directory used when --out is omitted")
    parser.add_argument("--out", default=None, help="explicit batch directory (must be new or empty unless --overwrite)")
    parser.add_argument("--overwrite", action="store_true",
                        help="with --out: move an existing non-empty directory aside (renamed, never deleted)")
    args = parser.parse_args()
    load_spec(args.spec)  # validate before creating anything
    try:
        if args.out is not None:
            batch_dir, _ = prepare_explicit_dir(args.out, overwrite=args.overwrite)
        else:
            batch_dir = make_unique_run_dir(args.output_root, Path(args.spec).stem)
    except OutputDirExistsError as exc:
        print(f"[OUTPUT ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"[batch] directory: {batch_dir}", file=sys.stderr, flush=True)
    if args.workers is None:
        workers, source = default_workers(), "automatic"
    else:
        workers, source = args.workers, "explicit"
    print(f"[batch] workers: {workers} ({source})", file=sys.stderr, flush=True)
    rows = run_batch(args.spec, batch_dir, workers=workers, timeout=args.timeout, workers_source=source)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[batch] done: {counts}; summary: {batch_dir / 'summary.csv'}", file=sys.stderr, flush=True)
    sys.exit(0 if counts.get("error", 0) == 0 and counts.get("timeout", 0) == 0 else 1)


if __name__ == "__main__":
    main()

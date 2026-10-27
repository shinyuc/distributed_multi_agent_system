"""Pilot runner: the unchanged ``diagnostics.run_feasibility`` pipeline with optional hooks (no controller change).

    python -m diagnostics.study48.run_pilot --config CFG --out DIR [--warm PKL] [--no-strict]
                                            [--snapshot-every K] [--log-first-qps S]
                                            [any run_feasibility option, e.g. --max-steps 3]

Hooks (the Stage-2 protocol of the validated D3 scratch runner, clfc/run_d3.py, generalized):

* ``--warm PKL``: ``Simulator.initialize_fast_states`` returns a deep copy of the stored warm fast state
  (``{"warm": {i: FastState}}``, as written by diagnostics.study48.scenarios); the plant starts at the YAML x_0.
  The warm file's SHA-256 and metadata are written to ``DIR/warm_start.json``.
* strict guard (default; ``--no-strict`` disables it): after the recorder has classified a local QP, a QP that OSQP
  reports solved but the independent LP confirms infeasible (recorder class ``inconsistent``) stops the run with
  ``StrictFeasibilityGuard`` before that step's controls are applied (the simulator records termination
  ``exception`` at that step, state not advanced). Without it such a step would apply a control from an
  infeasible QP. QPs that OSQP reports infeasible already stop the run (simulator behaviour, no fallback).
* ``--snapshot-every K``: a deep copy of the full fast state at every K-th history snapshot of the simulator
  (``DIR/fast_snapshots.pkl``: {"every": K, "snaps": [...]}), for the a-posteriori certificate.
* ``--log-first-qps S``: the mismatch variables (z_pair, z_c, z_clf of every agent) passed to every local-QP build
  of the first S steps (``DIR/first_qp_inputs.json``; read-only observer of build_local_qp_data).

``DIR/pilot_runner.json`` records the hook settings, the SHA-256 of the configuration file that was run, and the
git commit and working-tree state of the repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class StrictFeasibilityGuard(RuntimeError):
    """A local QP was reported solved although the independent LP confirms it is infeasible."""


def _git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                    # noqa: BLE001 - provenance only
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_files_modified": None if status is None else bool(status)}


def install_hooks(warm=None, strict=True, snapshot_every=0, log_first_qps=0):
    """Patch the simulator/recorder in this process; returns (snapshot list, first-QP input log), both filled
    during the run."""
    import controller.distributed_controller as dcm
    import simulation.simulator as sm
    import diagnostics.recorder as rec
    from diagnostics.feasibility import CLASS_CODE

    snaps: list = []
    qp_inputs: list = []
    if warm is not None:
        sm.Simulator.initialize_fast_states = lambda self: copy.deepcopy(warm)
    if strict:
        orig_call = rec.FeasibilityRecorder.__call__
        bad = CLASS_CODE["inconsistent"]

        def strict_call(self, **kw):
            orig_call(self, **kw)
            step = kw.get("step_idx")
            if isinstance(step, int) and self.a_class[step, self.col[kw["agent_id"]]] == bad:
                raise StrictFeasibilityGuard(
                    f"step {step}, agent {kw['agent_id']}: solver status {kw['solve_result'].status} on a local QP "
                    f"that the independent LP confirms infeasible; the run is stopped before the step's controls are "
                    f"applied")

        rec.FeasibilityRecorder.__call__ = strict_call
    if snapshot_every:
        orig_rec = sm.Simulator._record_fast_history_snapshot
        count = [0]

        def record(self, history):
            if count[0] % snapshot_every == 0:
                snaps.append(copy.deepcopy(self.fast_states))
            count[0] += 1
            return orig_rec(self, history)

        sm.Simulator._record_fast_history_snapshot = record
    if log_first_qps:
        orig_build = dcm.build_local_qp_data

        def build_logged(**kw):
            step = kw.get("step_idx")
            if isinstance(step, int) and step < log_first_qps:
                allf = kw["all_fast_states"]
                qp_inputs.append({"step": step, "agent": kw["i"],
                                  "z_clf": {str(j): f.z_clf for j, f in allf.items()},
                                  "z_c": {str(j): f.z_c for j, f in allf.items()},
                                  "z_pair": {str(j): {str(m): e.self_component for m, e in f.z_pair.items()}
                                             for j, f in allf.items()}})
            return orig_build(**kw)

        dcm.build_local_qp_data = build_logged
    return snaps, qp_inputs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", default=None)
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--log-first-qps", type=int, default=0)
    a, rest = ap.parse_known_args(argv)
    warm, warm_meta = None, None
    if a.warm:
        raw = Path(a.warm).read_bytes()
        d = pickle.loads(raw)
        warm = d["warm"]
        warm_meta = {"file": str(Path(a.warm).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                     **{k: v for k, v in d.items() if k != "warm"}}
    snaps, qp_inputs = install_hooks(warm, strict=not a.no_strict, snapshot_every=a.snapshot_every,
                                     log_first_qps=a.log_first_qps)
    cfg_sha = hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    import diagnostics.run_feasibility as rf
    sys.argv = ["run_feasibility", "--config", a.config, "--out", a.out, *rest]
    code = 0
    try:
        rf.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        out = Path(a.out)
        if out.is_dir():
            (out / "pilot_runner.json").write_text(json.dumps(
                {"strict_guard": not a.no_strict, "warm_start": warm_meta is not None,
                 "snapshot_every": a.snapshot_every, "log_first_qps": a.log_first_qps, "extra_args": rest,
                 "config": str(Path(a.config).resolve()), "config_sha256": cfg_sha, "git": _git_state()},
                indent=2, default=str))
            if a.log_first_qps:
                (out / "first_qp_inputs.json").write_text(json.dumps(qp_inputs, indent=2))
            if warm_meta is not None:
                (out / "warm_start.json").write_text(json.dumps(warm_meta, indent=2, default=str))
            if a.snapshot_every:
                with open(out / "fast_snapshots.pkl", "wb") as f:
                    pickle.dump({"every": a.snapshot_every, "snaps": snaps}, f)
    return code or 0


if __name__ == "__main__":
    sys.exit(main())

"""Check a server smoke batch (pilot/server_smoke_batch.yaml) against the expected execution paths.

    python -m diagnostics.study48.verify_smoke --bundle BUNDLE_DIR --batch SMOKE_OUT [--expect-commit SHA]

Checks (each reported as PASS/FAIL, exit code 1 if any fails):
  * every scenario ran the pilot runner with the strict guard on, and the configuration it ran is identical (as
    parsed YAML) to the bundle configuration, whose SHA-256 equals the manifest's; config SHA-256 recorded;
  * the recorded git commit equals the expected one and no tracked file is modified;
  * C09 (zero obstacles): completed 3/3 steps, every QP solved, no inconsistent QP, clearance reported empty;
  * A01: terminated at step 0 by a local QP failure, trajectory = x_0 only, no QP solved, no inconsistent QP;
  * D03: completed, warm start SHA-256 equals the bundle file, the first fast-state snapshot equals the warm
    state, and every step-0 QP received exactly the warm mismatch (z_pair, z_c, z_clf of all agents);
  * no Traceback in any stderr.log and batch_meta lists the builtin OSQP backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import yaml


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(bundle: Path, batch: Path, expect_commit: str | None = None,
           require_clean: bool = True) -> list[tuple[str, bool, str]]:
    res: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        res.append((name, bool(ok), str(detail)))

    man = {e["id"]: e for e in json.loads((bundle / "manifest_pilot16.json").read_text())["scenarios"]}
    rows = {r["scenario_id"]: r for r in csv.DictReader(open(batch / "summary.csv"))}
    meta = json.loads((batch / "batch_meta.json").read_text())
    check("batch: child OSQP backend builtin", meta.get("child_solver_env") == {"OSQP_ALGEBRA_BACKEND": "builtin"},
          meta.get("child_solver_env"))
    check("batch: scenarios", sorted(rows) == ["A01", "C09", "D03"], sorted(rows))
    for sid, row in rows.items():
        sdir = batch / "scenarios" / sid
        run = sdir / "run"
        err = (sdir / "stderr.log").read_text(errors="replace") if (sdir / "stderr.log").exists() else ""
        check(f"{sid}: no Traceback in stderr", "Traceback" not in err)
        pr = json.loads((run / "pilot_runner.json").read_text())
        rm = json.loads((run / "run_meta.json").read_text())
        e = man[sid]
        bcfg = bundle / e["config"]
        check(f"{sid}: bundle config SHA-256 = manifest", _sha(bcfg) == e["config_sha256"])
        check(f"{sid}: ran config = bundle config (parsed)",
              yaml.safe_load((sdir / "scenario.yaml").read_text()) == yaml.safe_load(bcfg.read_text()))
        check(f"{sid}: config SHA-256 recorded", pr.get("config_sha256") == _sha(sdir / "scenario.yaml"))
        check(f"{sid}: strict guard on", pr.get("strict_guard") is True)
        git = pr.get("git") or {}
        if expect_commit:
            check(f"{sid}: git commit", git.get("commit") == expect_commit == rm.get("git_commit"),
                  f"{git.get('commit')} / run_meta {rm.get('git_commit')}")
        if require_clean:
            check(f"{sid}: no modified tracked files", git.get("tracked_files_modified") is False, git)
        traj = json.loads((run / "trajectories.json").read_text())["trajectories"]
        if sid == "C09":
            check("C09: completed 3/3", row["status"] == "completed" and row["steps_completed"] == "3", row["status"])
            check("C09: all QPs solved", row["qp_solved"] == row["qp_solves"] == "9", (row["qp_solved"], row["qp_solves"]))
            check("C09: no inconsistent QP", row["qp_inconsistent"] == "0")
            check("C09: clearance empty (no obstacles)", row["min_obstacle_clearance"] == "", row["min_obstacle_clearance"])
        if sid == "A01":
            check("A01: terminated at step 0 by local QP failure",
                  row["status"] == "terminated" and row["termination_reason"] == "local_qp_failure"
                  and row["termination_step"] == "0", (row["status"], row["termination_reason"], row["termination_step"]))
            check("A01: trajectory = x_0 only (no control applied)", len(traj) == 1, len(traj))
            check("A01: no QP solved, none inconsistent", row["qp_solved"] == "0" and row["qp_inconsistent"] == "0")
        if sid == "D03":
            check("D03: completed 3/3", row["status"] == "completed" and row["steps_completed"] == "3", row["status"])
            wfile = bundle / e["warm_start"]["file"]
            ws = json.loads((run / "warm_start.json").read_text())
            check("D03: warm SHA-256 = bundle file", ws["sha256"] == _sha(wfile))
            warm = pickle.loads(wfile.read_bytes())["warm"]
            snaps = pickle.loads((run / "fast_snapshots.pkl").read_bytes())["snaps"]
            check("D03: first snapshot = warm state", bool(snaps) and snaps[0] == warm)
            first = json.loads((run / "first_qp_inputs.json").read_text())
            want = {"z_clf": {str(j): f.z_clf for j, f in warm.items()}, "z_c": {str(j): f.z_c for j, f in warm.items()},
                    "z_pair": {str(j): {str(m): v.self_component for m, v in f.z_pair.items()} for j, f in warm.items()}}
            step0 = [q for q in first if q["step"] == 0]
            check("D03: every step-0 QP logged", sorted(q["agent"] for q in step0) == sorted(warm), len(step0))
            check("D03: step-0 QPs received the warm mismatch",
                  all(all(q[k] == want[k] for k in want) for q in step0))
            check("D03: warm mismatch is nonzero", any(v != 0 for v in want["z_clf"].values())
                  or any(v != 0 for d in want["z_pair"].values() for v in d.values()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the server smoke batch")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="do not require a clean tracked working tree")
    a = ap.parse_args(argv)
    res = verify(Path(a.bundle), Path(a.batch), a.expect_commit, require_clean=not a.allow_dirty)
    for name, ok, detail in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
    bad = [r for r in res if not r[1]]
    print(f"{len(res) - len(bad)}/{len(res)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

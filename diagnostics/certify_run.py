"""Certify a recorded run with the R6.1 Method B certificate (post-processing only; no simulation).

Usage:
    python -m diagnostics.certify_run --config CFG.yaml --run-dir RUN_DIR [--init zero | --init WARM.pkl]
                                      [--num-states K] [--norm spectral|frobenius] [--out OUT.json]
                                      [--check-model-at 0,last]

The QP states are trajectories[0..K-1] from RUN_DIR/trajectories.json. By default K = steps_completed for a
completed run and steps_completed + 1 for a run that terminated at a QP (the failing QP state is included).
WARM.pkl is either a dict {agent_id: FastState} or a dict with key "warm" holding one.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from config import load_config
from diagnostics.certificate import CertModel, CertTolerances, certify, zero_fast_state


def load_initial_state(model: CertModel, init: str):
    if init == "zero":
        return zero_fast_state(model)
    obj = pickle.load(open(init, "rb"))
    return obj["warm"] if isinstance(obj, dict) and "warm" in obj else obj


def default_num_states(run_dir: Path) -> int:
    traj = json.load(open(run_dir / "trajectories.json"))
    k = int(traj["steps_completed"])
    term = traj.get("termination_reason") or json.load(open(run_dir / "summary.json")).get("termination_reason")
    return k if term == "completed" else min(k + 1, len(traj["trajectories"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", default="zero")
    ap.add_argument("--num-states", type=int, default=None)
    ap.add_argument("--norm", choices=("spectral", "frobenius"), default="spectral")
    ap.add_argument("--check-model-at", default="0")
    ap.add_argument("--out", default=None, help="JSON summary path (per-step arrays go to the same stem .npz)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    cfg = load_config(a.config)
    model = CertModel(cfg)
    positions = json.load(open(run_dir / "trajectories.json"))["trajectories"]
    K = a.num_states if a.num_states is not None else default_num_states(run_dir)
    positions = positions[:K]
    chk = tuple(int(t) if t != "last" else K - 1 for t in a.check_model_at.split(",") if t)
    t0 = time.time()
    cert = certify(model, positions, load_initial_state(model, a.init), CertTolerances(norm=a.norm), check_model_at=chk,
                   progress=lambda k, n: print(f"[certify] {k}/{n}", file=sys.stderr, flush=True))
    summ = cert.summary()
    summ.update(config=str(a.config), run_dir=str(run_dir), init=a.init, num_states=K, norm=a.norm,
                wall_time_s=time.time() - t0)
    for v, r in cert.results.items():
        print(f"[certify] {v}: {r.status} reason={r.reason} ({r.reason_kind}) k_fail={r.k_fail} "
              f"certified_states={r.certified_states}/{K} :: {r.detail}", file=sys.stderr)
    if a.out:
        out = Path(a.out)
        out.write_text(json.dumps(summ, indent=2, default=str))
        np.savez_compressed(out.with_suffix(".npz"), **{k: np.asarray(v, dtype=object if k in
                            ("active", "rho_agents", "C1_label", "C2_label", "kkt_status") else float)
                            for k, v in cert.steps.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

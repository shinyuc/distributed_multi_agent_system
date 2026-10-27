"""Build the pilot transfer bundle (only the files the 16-scenario pilot needs) from a generated study directory.

    python -m diagnostics.study48.bundle --study STUDY_DIR --out BUNDLE_DIR [--expect-commit SHA]

BUNDLE_DIR (must not exist) receives

    configs/<ID>.yaml            the 16 pilot configurations, byte-identical to the study
    warm/D03_warm.pkl            the frozen warm start(s) of the pilots that use one
    manifest_pilot16.json        the 16 pilot entries of manifest.json, verbatim (config SHA-256 included)
    expected_outcomes.json       Tier-0 predictions of the pilots
    pilot/pilot_batch.yaml       the pilot batch spec (paths relative to this file; no machine-specific paths)
    pilot/server_smoke_batch.yaml  3 scenarios x 3 steps: zero obstacles (C09), step-0 termination (A01), warm D03
    BUNDLE_INFO.json             source study, full-manifest SHA-256, expected commit, file list
    SHA256SUMS                   `sha256sum -c` format, every file above
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

SMOKE = (("C09", "zero obstacles, proceeds"), ("A01", "terminates at step 0 (OSQP infeasible), no control applied"),
         ("D03", "warm start; first QPs must receive the warm mismatch"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(study: Path, out: Path, expect_commit: str | None = None) -> dict:
    study, out = Path(study), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    man = json.loads((study / "manifest.json").read_text())
    pil = [e for e in man["scenarios"] if e["pilot"]]
    if len(pil) != 16:
        raise ValueError(f"expected 16 pilots, found {len(pil)}")
    for sub in ("configs", "warm", "pilot"):
        (out / sub).mkdir()
    for e in pil:
        src = study / e["config"]
        if _sha(src) != e["config_sha256"]:
            raise ValueError(f"{e['id']}: config hash differs from the manifest")
        shutil.copy2(src, out / e["config"])
        if e.get("warm_start"):
            shutil.copy2(study / e["warm_start"]["file"], out / e["warm_start"]["file"])
    (out / "manifest_pilot16.json").write_text(json.dumps({k: v for k, v in man.items() if k != "scenarios"}
                                                          | {"scenarios": pil}, indent=2))
    shutil.copy2(study / "pilot/expected_outcomes.json", out / "expected_outcomes.json")
    spec = yaml.safe_load((study / "pilot/pilot_batch.yaml").read_text())
    ids = sorted(s["id"] for s in spec["scenarios"])
    if ids != sorted(e["id"] for e in pil):
        raise ValueError("pilot spec and manifest pilots differ")
    shutil.copy2(study / "pilot/pilot_batch.yaml", out / "pilot/pilot_batch.yaml")
    by_id = {s["id"]: s for s in spec["scenarios"]}
    smoke = {"name": "study48_server_smoke", "scenarios": []}
    for sid, _ in SMOKE:
        s = json.loads(json.dumps(by_id[sid]))
        s["runner_args"] = s["runner_args"] + ["--max-steps", "3", "--log-first-qps", "1"]
        smoke["scenarios"].append(s)
    (out / "pilot/server_smoke_batch.yaml").write_text(
        "# Server smoke test only: 3 outer steps each. " + "; ".join(f"{i}: {d}" for i, d in SMOKE) + "\n"
        + yaml.safe_dump(smoke, sort_keys=False))
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    info = {"source_study": str(study.resolve()), "full_manifest_sha256": _sha(study / "manifest.json"),
            "expected_commit": expect_commit, "pilots": [e["id"] for e in pil], "files": files}
    (out / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=2))
    summed = sorted(files + ["BUNDLE_INFO.json"])
    (out / "SHA256SUMS").write_text("".join(f"{_sha(out / f)}  {f}\n" for f in summed))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the pilot transfer bundle")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-commit", default=None)
    a = ap.parse_args(argv)
    info = build(Path(a.study), Path(a.out), a.expect_commit)
    print(f"[bundle] {len(info['files'])} files for pilots {info['pilots']} in {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

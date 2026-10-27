"""Tests of the 48-scenario study generator (diagnostics.study48) and its Tier-0 static screening.

Generation runs without warm starts (static, ~2 s); the warm-start reproduction and the replica comparisons with
the saved experiment configurations are skipped unless TCNS_RESEARCH_DATA points to the research data folder.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from config import load_config
from diagnostics.certificate import CertModel
from diagnostics.static_feasibility import StaticTolerances, classify_polytope, graphs, level1_problem
from diagnostics.study48 import tier0
from diagnostics.study48.geometry import RULE_TOL, head_on_obstacle, lambda2, lattice_with_cutoff_rule, physical_validity
from diagnostics.study48.scenarios import PILOTS, _flatten, build_raw, family, generate, scenario_specs
from diagnostics.study48.warmstart import frozen_warmup

DATA = Path(os.environ["TCNS_RESEARCH_DATA"]) if os.environ.get("TCNS_RESEARCH_DATA") else None
needs_data = pytest.mark.skipif(DATA is None or not DATA.exists(), reason="TCNS_RESEARCH_DATA not set")

EXEMPTED = {"A07": {"lambda2_ge_eta_plus_0.01"}, "B09": {"lambda2_ge_eta_plus_0.01"},
            "C11": {"lambda2_ge_eta_plus_0.01"}, "D05": {"pair_ge_d0_plus_0.05"},
            "D06": {"lambda2_ge_eta_plus_0.01"},
            "A12": {"pair_ge_d0_plus_0.05", "obstacle_clearance_ge_0.05", "lambda2_ge_eta_plus_0.01",
                    "cutoff_margin_ge_0.02"}}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    out = tmp_path_factory.mktemp("study48")
    return out, generate(out, warm=False)


def _entry(doc, sid):
    return next(e for e in doc["scenarios"] if e["id"] == sid)


def _level1(cfg, u):
    rows = CertModel(cfg).rows(cfg.agents.initial_positions)
    M, b, lab = level1_problem(rows)
    return classify_polytope(M, b, -u * np.ones(rows.nw), u * np.ones(rows.nw), lab, StaticTolerances())


# ------------------------------------------------------------------------------------- counts and identifiers
def test_exactly_48_unique_ids_12_per_group_16_pilots(study):
    _, doc = study
    sc = doc["scenarios"]
    ids = [e["id"] for e in sc]
    assert len(ids) == 48 and len(set(ids)) == 48
    for g in "ABCD":
        assert [e["id"] for e in sc if e["group"] == g] == [f"{g}{k:02d}" for k in range(1, 13)]
        assert sum(e["pilot"] for e in sc if e["group"] == g) == 4
    assert {e["id"] for e in sc if e["pilot"]} == PILOTS and len(PILOTS) == 16


def test_comparison_pairs_reference_existing_scenarios(study):
    _, doc = study
    ids = {e["id"] for e in doc["scenarios"]}
    for e in doc["scenarios"]:
        for other, kind in e["pairs"]:
            assert other in ids and other != e["id"], (e["id"], other)
            assert kind in ("single-parameter", "design-factor", "contrast", "replicate")


# ------------------------------------------------------------------------------------- reproducibility, loading
def test_generation_is_reproducible(study, tmp_path):
    out, doc = study
    doc2 = generate(tmp_path, warm=False)
    for e1, e2 in zip(doc["scenarios"], doc2["scenarios"]):
        assert e1["config_sha256"] == e2["config_sha256"], e1["id"]
        assert (out / e1["config"]).read_bytes() == (tmp_path / e2["config"]).read_bytes()


def test_random_families_are_seeded_and_record_rejections():
    for name, seed in (("FA", 101), ("FB", 201), ("FD", 401)):
        X1, o1, f1 = family(name, seed)
        X2, o2, f2 = family(name, seed)
        assert X1 == X2 and o1 == o2 and f1 == f2
        assert f1["seed"] == seed and f1["family"] == name and f1["rejections"] >= 0


def test_configs_load_and_num_agents_is_resolved_from_contents(study):
    out, doc = study
    for e in doc["scenarios"]:
        cfg = load_config(str(out / e["config"]))
        assert cfg.agents.num_agents == len(cfg.agents.initial_positions) == e["num_agents"], e["id"]
        assert len(cfg.obstacles.obstacles) == e["num_obstacles"]
        assert cfg.controller.alpha_clf == 1.0 and not cfg.controller.debug_disable_clf_coupling
        if e["source"] != "repro":
            assert cfg.fast_dynamics.substeps >= 1
            assert cfg.simulation.steps == max(1, round(e["horizon_s"] / cfg.simulation.dt))
        CertModel(cfg)                                   # refuses unsupported settings


def test_changed_from_base_lists_exactly_the_differences(study):
    out, doc = study
    from diagnostics.study48.scenarios import BASE
    base = _flatten(yaml.safe_load(BASE.read_text()))
    for e in doc["scenarios"]:
        if e["source"] == "repro":
            continue
        raw = _flatten(yaml.safe_load((out / e["config"]).read_text()))
        assert {k for k in raw if base.get(k) != raw[k]} == set(e["changed_from_base"]), e["id"]


# ------------------------------------------------------------------------------------- geometry validation
def test_geometry_rules_hold_except_designated_exemptions(study):
    _, doc = study
    for e in doc["scenarios"]:
        failed = {k for k, ok in e["physical_validity"]["rules"].items() if not ok}
        if e["id"] == "A12":
            continue                                     # reported construction failure, tested separately
        assert failed <= EXEMPTED.get(e["id"], set()), (e["id"], failed)
        assert set(e["exempt"]) == EXEMPTED.get(e["id"], set()), e["id"]
        assert e["physical_validity"]["rules"]["pair_safe_H_ge_0"], e["id"]
        assert e["physical_validity"]["rules"]["lambda2_ge_eta"], e["id"]


def test_repro_initial_state_is_reported_not_repaired(study):
    _, doc = study
    a12 = _entry(doc, "A12")
    assert a12["source"] == "repro" and a12["changed_from_base"] == {}
    assert a12["physical_validity"]["lambda2_minus_eta"] < 0        # lambda2 < eta at the repro's x_0
    assert not a12["physical_validity"]["rules"]["lambda2_ge_eta"]
    assert a12["construction_rule_failures"] == ["lambda2_ge_eta"] and not a12["construction_ok"]
    assert [e["id"] for e in doc["scenarios"] if not e["construction_ok"]] == ["A12"]


def test_physical_validity_detects_violations():
    cfg = dict(Sigma=[[2, 0], [0, 0.25]], center=[0, 0], epsilon=0.5, d_c=0.9, epsilon_c=0.1, eta=0.1, d0=0.2)
    ok = physical_validity([[-1.0, 0.0], [-1.0, 0.5]], [], cfg)["rules"]
    assert all(ok.values())
    r = physical_validity([[-1.0, 0.0], [-1.0, 0.19]], [], cfg)["rules"]
    assert not r["pair_safe_H_ge_0"] and not r["pair_ge_d0_plus_0.05"]
    r = physical_validity([[-1.0, 0.0], [-1.0, 0.5]], [[[-1.3, 0.0], 0.27]], cfg)["rules"]
    assert not r["obstacle_clearance_ge_0.05"]
    r = physical_validity([[-1.0, 0.0], [-1.0, 0.81]], [], cfg)["rules"]
    assert not r["cutoff_margin_ge_0.02"]
    r = physical_validity([[-1.0, 0.0], [-1.0, 0.95]], [], cfg)["rules"]
    assert not r["lambda2_ge_eta"]                                  # disconnected


def test_head_on_obstacle_lies_on_the_descent_ray():
    x = np.array([-0.55, 1.35])
    c, r = head_on_obstacle(x, 0.15, 0.05, [[2, 0], [0, 0.25]], [0, 0])
    d = -np.diag([2, 0.25]) @ x
    assert np.linalg.norm(np.asarray(c) - x) == pytest.approx(0.2, abs=1e-15)
    v = np.asarray(c) - x
    assert v[0] * d[1] - v[1] * d[0] == pytest.approx(0.0, abs=1e-15) and np.dot(v, d) > 0


def test_lattice_cutoff_rule():
    X, s = lattice_with_cutoff_rule(2, 4, 0.25, -1.5, 0.9, 0.1)
    D = [np.linalg.norm(np.subtract(a, b)) for i, a in enumerate(X) for b in X[i + 1:]]
    assert all(min(abs(d - 0.9), abs(d - 0.8)) >= 0.02 - RULE_TOL for d in D if d < 0.95)
    assert s >= 0.25


# ------------------------------------------------------------------------------------- A01 / A02 threshold
def test_a02_threshold_is_bracketed_with_verified_evidence(study):
    out, doc = study
    cal = _entry(doc, "A02")["calibration_result"]
    assert cal["finite_threshold"] and (cal["hi"] - cal["lo"]) / cal["hi"] <= 1e-6
    assert cal["evidence_hi"]["verdict"] == "F" and cal["evidence_hi"]["witness_slack"] <= -1e-9
    assert cal["evidence_lo"]["verdict"] == "I" and cal["evidence_lo"]["phi"] > 0
    assert cal["threshold"] == pytest.approx(0.760872, abs=2e-6)
    spec = next(s for s in scenario_specs() if s["id"] == "A02")
    tmp = out / "chk"
    tmp.mkdir(exist_ok=True)
    for u, want in ((cal["lo"], "I"), (cal["hi"], "F")):
        p = tmp / "a02.yaml"
        p.write_text(yaml.safe_dump(build_raw(spec, u), sort_keys=False))
        assert _level1(load_config(str(p)), u).verdict == want
    a02 = load_config(str(out / _entry(doc, "A02")["config"]))
    assert a02.controller.u_max == pytest.approx(1.05 * cal["threshold"], rel=1e-15)
    assert _level1(a02, a02.controller.u_max).verdict == "F"


def test_a01_is_below_the_threshold_and_differs_from_a02_only_in_u_max_and_horizon(study):
    out, doc = study
    a01, a02 = (_flatten(yaml.safe_load((out / _entry(doc, s)["config"]).read_text())) for s in ("A01", "A02"))
    assert {k for k in a01 if a01[k] != a02[k]} == {"controller.u_max", "simulation.steps"}
    assert a01["controller.u_max"] < _entry(doc, "A02")["calibration_result"]["lo"]


def test_b03_margin_calibration(study):
    _, doc = study
    cal = _entry(doc, "B03")["calibration_result"]
    assert cal["ok"] and cal["verdict"] == "F" and cal["lp_margin"] == pytest.approx(-5e-4, rel=1e-6)


# ------------------------------------------------------------------------------------- B06 connectivity
def test_b06_connectivity_and_components(study):
    out, doc = study
    e = _entry(doc, "B06")
    cfg = load_config(str(out / e["config"]))
    assert cfg.agents.num_agents == 8 and cfg.fast_dynamics.eta == 0.05
    pv = e["physical_validity"]
    assert pv["lambda2"] >= 0.05 + 0.01 and pv["lambda2"] < 0.1 + 0.01     # why the eta fallback is needed
    assert pv["lambda2"] == pytest.approx(lambda2(cfg.agents.initial_positions, 0.9, 0.1), rel=1e-12)
    rows = CertModel(cfg).rows(cfg.agents.initial_positions)
    assert rows.lam2 == pytest.approx(pv["lambda2"], rel=1e-9)
    q = pv["q"]
    assert [v <= 0 for v in q] == [False, False, False, True, True, False, False, False]
    g = graphs(rows)
    assert len(g["h"]) == 1
    assert sorted(map(sorted, g["v"])) == [[0, 1, 2], [3], [4], [5, 6, 7]]
    assert pv["cutoff_margin"] >= 0.02 and pv["min_obstacle_clearance"] >= 0.05 - RULE_TOL


# ------------------------------------------------------------------------------------- Tier-0 consistency
@pytest.mark.parametrize("sid, cat, cert", [("D01", "feasible", "CERT_OK"), ("D02", "feasible", "CERT_OK"),
                                            ("C03", "C", "CERT_FAIL")])
def test_tier0_known_states(study, sid, cat, cert):
    out, doc = study
    r = tier0.screen_scenario(out, _entry(doc, sid))
    assert r["category"] == cat and not [c for c in r["consistency"] if c.startswith("INCONSISTENT")]
    l4 = r["level4"]
    assert l4["kkt_status"] == "ok" and l4["rho_status"] == "certified_positive"
    assert 0 < l4["rho_lower"] <= l4["rho_upper"]
    for norm in ("spectral", "frobenius"):
        assert l4["frozen"][norm]["results"]["non_increase"]["status"] == cert
    assert l4["frozen"]["spectral"]["a"] <= l4["frozen"]["frobenius"]["a"]
    assert l4["frozen"]["spectral"]["E0"] <= l4["frozen"]["frobenius"]["E0"] * (1 + 1e-12)
    if sid == "D01":
        assert l4["rho_lower"] == pytest.approx(0.133218, abs=1e-6) and l4["frozen"]["spectral"]["thm1_continuous"]
        assert r["step0_preflight"]["prediction"] == "proceeds"
    if sid == "C03":                                   # F2: 0 not in Z_feas implies rho <= |z*| < |e_0|
        assert not l4["z0_in_Zfeas"] and not l4["frozen"]["spectral"]["thm1_continuous"]
        assert l4["rho_lower"] <= l4["z_star_norm"]
        assert l4["frozen"]["spectral"]["results"]["non_increase"]["reason"] == "RHO"   # E_0 >= the upper bound
        assert r["step0_preflight"]["prediction"] == "terminates_at_step0"
        assert r["step0_preflight"]["osqp_failed_agents"] == [0]


def test_tier0_skips_kkt_when_level2_infeasible(study):
    out, doc = study
    r = tier0.screen_scenario(out, _entry(doc, "A01"))
    assert r["levels"]["level1"]["verdict"] == "I" and r["category"] == "A"
    assert "kkt_status" not in r["level4"] and "skipped" in r["level4"]
    pf = r["step0_preflight"]                         # the simulator will stop at step 0 without applying controls
    assert pf["prediction"] == "terminates_at_step0" and pf["osqp_failed_agents"] == [0, 1]
    assert not pf["inconsistent_agents"] and all(a["lp_verdict"] == "infeasible" for a in pf["agents"])


@pytest.mark.parametrize("sid, status", [("C01", "upper_bound_below_tol"), ("B02", "upper_bound_below_tol"),
                                         ("D07", "certified_positive")])
def test_tier0_margin_reporting_is_never_negative(study, sid, status):
    out, doc = study
    l4 = tier0.screen_scenario(out, _entry(doc, sid))["level4"]
    assert l4["rho_status"] == status and l4["rho_lower"] >= 0.0 and l4["rho_lower"] <= l4["rho_upper"] + 1e-15
    if status == "upper_bound_below_tol":
        assert l4["rho_lower"] == 0.0 and l4["rho_upper"] <= 1e-9
    if sid == "D07":                                  # C0 holds for the spectral bound only
        fz = l4["frozen"]
        assert fz["spectral"]["c0_status"] == "ok" and fz["frobenius"]["c0_status"] == "fail"
        assert fz["spectral"]["results"]["non_increase"]["status"] == "CERT_OK"


def test_a12_is_a_flagged_legacy_case(study):
    out, doc = study
    a12 = _entry(doc, "A12")
    assert a12["legacy"] and not a12["initial_connectivity_ok"]
    assert [e["id"] for e in doc["scenarios"] if not e["initial_connectivity_ok"]] == ["A12"]
    assert [e["id"] for e in doc["scenarios"] if e["legacy"]] == ["A12"] and not a12["pilot"]
    r = tier0.screen_scenario(out, a12)
    assert r["legacy_case"] and not r["initial_connectivity_ok"]
    # lambda2 < eta does not by itself make the centralized QP infeasible: here Level 1 and Level 2 are F
    assert r["levels"]["level1"]["verdict"] == "F" and r["levels"]["level2"]["verdict"] == "F"


def test_b04_disagreement_is_mathematical_feasibility_with_solver_failure(study):
    from diagnostics.study48.disagreement import diagnose
    out, doc = study
    cfg = load_config(str(out / _entry(doc, "B04")["config"]))
    d = diagnose(CertModel(cfg), cfg.agents.initial_positions)
    c = d["classification"]
    assert c["mathematical_level2"] == "F" and d["lifted"]["exact_verified"] and d["column_scaled_lp"]["witness_verified"]
    lb, zl = c["required_mismatch_inf_norm"]
    assert 1e9 < lb <= zl and lb <= d["column_scaled_lp"]["witness_z_inf_norm"]
    assert d["coupling"]["min_v"] == pytest.approx(1.617e-11, rel=1e-3)          # weak, not zero
    assert c["solver_disagreement_explained"]                                      # HiGHS drops entries < 1e-9
    assert d["solver_mechanism"]["lp_margin_full_matrix"] == pytest.approx(d["levels"]["repository"]["s_star"], rel=1e-12)
    assert d["kkt"]["status"] == "L2_INFEASIBLE" and d["kkt"]["interpretation"].startswith("solver failure")


@needs_data
@pytest.mark.parametrize("sid, ref", [("D01", "tgt/A_gated.yaml"), ("C01", "tgt/A_communication.yaml"),
                                      ("D02", "clfc/C_gated.yaml"), ("C02", "tgt/B_gated.yaml"),
                                      ("C08", "tgt/B_communication.yaml"), ("D03", "clfc/D3_run.yaml"),
                                      ("C03", "clfc/D3_run.yaml"), ("D11", "clfc/D2.yaml")])
def test_calibration_replicas_equal_saved_configs(study, sid, ref):
    out, doc = study
    a = _flatten(yaml.safe_load((out / _entry(doc, sid)["config"]).read_text()))
    b = _flatten(yaml.safe_load((DATA / ref).read_text()))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"simulation.steps"}


@needs_data
def test_d03_warm_start_reproduces_saved_d3_state(study):
    out, doc = study
    cfg = load_config(str(out / _entry(doc, "D03")["config"]))
    fs, info = frozen_warmup(CertModel(cfg), cfg.agents.initial_positions, {"type": "E_abs", "value": 0.0056})
    ref = pickle.load(open(DATA / "clfc/D3cl_warm.pkl", "rb"))["warm"]
    assert info["stop_step"] == 863
    rows = CertModel(cfg).rows(cfg.agents.initial_positions)
    (p1, y1), (p2, y2) = rows.to_vec(fs), rows.to_vec(ref)
    assert np.array_equal(p1, p2) and np.array_equal(y1, y2)       # bit-identical


@needs_data
def test_pilot_batch_spec_is_ready():
    import hashlib
    import json
    from diagnostics.run_batch import load_spec
    study = DATA / "study48"
    man = {e["id"]: e for e in json.load(open(study / "manifest.json"))["scenarios"]}
    spec = load_spec(study / "pilot/pilot_batch.yaml")
    assert sorted(s["id"] for s in spec["scenarios"]) == sorted(PILOTS)
    for s in spec["scenarios"]:
        e = man[s["id"]]
        assert s["runner"] == "diagnostics.study48.run_pilot" and s["overrides"] == {}
        assert hashlib.sha256(Path(s["base_config"]).read_bytes()).hexdigest() == e["config_sha256"]
        assert ("--warm" in s["runner_args"]) == bool(e.get("warm_start")) == (s["id"] == "D03")
        assert "--max-steps" not in s["runner_args"] and "--no-strict" not in s["runner_args"]
        assert s["config"]["simulation"]["steps"] == e["steps"]


@needs_data
def test_pilot_bundle_and_server_smoke_verification(tmp_path):
    import subprocess
    import sys
    from diagnostics.study48.bundle import build
    from diagnostics.study48.verify_smoke import verify
    b = tmp_path / "bundle"
    info = build(DATA / "study48", b)
    assert sorted(p.stem for p in (b / "configs").glob("*.yaml")) == sorted(PILOTS)      # nothing beyond the 16
    assert [p.name for p in (b / "warm").iterdir()] == ["D03_warm.pkl"]
    for line in (b / "SHA256SUMS").read_text().splitlines():
        h, f = line.split("  ")
        assert hashlib_sha256(b / f) == h
    assert len(info["files"]) == len((b / "SHA256SUMS").read_text().splitlines()) - 1
    out = tmp_path / "smoke"
    out.mkdir()
    r = subprocess.run([sys.executable, "-m", "diagnostics.run_batch", "--spec", str(b / "pilot/server_smoke_batch.yaml"),
                        "--workers", "3", "--out", str(out)], cwd=Path(__file__).resolve().parents[1],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = verify(b, out, require_clean=False)        # the developer tree may be dirty; the server must not
    assert all(ok for _, ok, _ in res), [x for x in res if not x[1]]


def hashlib_sha256(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


PILOT16 = Path(__file__).resolve().parents[1] / "experiments/feasibility_study/pilot16"


def test_repository_pilot_inputs_are_complete_portable_and_unchanged(study):
    import hashlib
    import json
    import subprocess
    from diagnostics.run_batch import load_spec
    out, doc = study
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sums = [ln.split("  ") for ln in (PILOT16 / "SHA256SUMS").read_text().splitlines()]
    assert all(sha(PILOT16 / f) == h for h, f in sums)
    listed = {f for _, f in sums} | {"SHA256SUMS"}
    present = {p.relative_to(PILOT16).as_posix() for p in PILOT16.rglob("*") if p.is_file()}
    assert present == listed                                              # nothing unlisted, nothing missing
    assert sorted(p.stem for p in (PILOT16 / "configs").glob("*.yaml")) == sorted(PILOTS)
    man = {e["id"]: e for e in json.loads((PILOT16 / "manifest_pilot16.json").read_text())["scenarios"]}
    gen = {e["id"]: e for e in doc["scenarios"]}
    for sid in PILOTS:                                                    # byte-identical to a fresh generation
        assert sha(PILOT16 / "configs" / f"{sid}.yaml") == man[sid]["config_sha256"] == gen[sid]["config_sha256"]
        assert (PILOT16 / "configs" / f"{sid}.yaml").read_bytes() == (out / "configs" / f"{sid}.yaml").read_bytes()
    cfg = load_config(str(PILOT16 / "configs/D03.yaml"))                  # warm start recomputed bit-identically
    model = CertModel(cfg)
    fs, info = frozen_warmup(model, cfg.agents.initial_positions, man["D03"]["init"])
    ref = pickle.load(open(PILOT16 / "warm/D03_warm.pkl", "rb"))
    rows = model.rows(cfg.agents.initial_positions)
    (p1, y1), (p2, y2) = rows.to_vec(fs), rows.to_vec(ref["warm"])
    assert np.array_equal(p1, p2) and np.array_equal(y1, y2) and info["stop_step"] == ref["stop_step"] == 863
    for name, n in (("pilot_batch", 16), ("server_smoke_batch", 3)):
        text = (PILOT16 / "pilot" / f"{name}.yaml").read_text()
        assert "/Users/" not in text and "/home/" not in text and "/private/" not in text
        sc = load_spec(PILOT16 / "pilot" / f"{name}.yaml")["scenarios"]
        assert len(sc) == n
        for s in sc:
            assert Path(s["base_config"]).is_relative_to(PILOT16.resolve())
            assert all(Path(a).is_relative_to(PILOT16.resolve()) for a in s["runner_args"] if a.endswith(".pkl"))
            assert ("--max-steps" in s["runner_args"]) == (name == "server_smoke_batch")
            assert "--no-strict" not in s["runner_args"]
    r = subprocess.run(["git", "check-ignore", "--no-index", *sorted(present)], cwd=PILOT16, capture_output=True, text=True)
    assert r.stdout == ""                                                 # no file excluded by a .gitignore rule
    assert "run_batch --spec" in (PILOT16 / "SERVER_STEPS.sh").read_text()
    assert not any(ln.strip().startswith("$PY -m diagnostics.run_batch --spec \"$STUDY_DIR/pilot/pilot_batch")
                   for ln in (PILOT16 / "SERVER_STEPS.sh").read_text().splitlines())   # the pilot is only printed


REMAINING32 = Path(__file__).resolve().parents[1] / "experiments/feasibility_study/remaining32"
STUDY48_META = Path(__file__).resolve().parents[1] / "experiments/feasibility_study/study48"


def test_repository_remaining32_inputs_are_complete_and_unchanged(study):
    import hashlib
    import json
    from diagnostics.run_batch import load_spec
    out, doc = study
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sums = [ln.split("  ") for ln in (REMAINING32 / "SHA256SUMS").read_text().splitlines()]
    assert all(sha(REMAINING32 / f) == h for h, f in sums)
    present = {p.relative_to(REMAINING32).as_posix() for p in REMAINING32.rglob("*") if p.is_file()}
    assert present == {f for _, f in sums} | {"SHA256SUMS"}
    gen = {e["id"]: e for e in doc["scenarios"]}
    rest = sorted(i for i, e in gen.items() if not e["pilot"])
    assert len(rest) == 32 and sorted(p.stem for p in (REMAINING32 / "configs").glob("*.yaml")) == rest
    man = {e["id"]: e for e in json.loads((REMAINING32 / "manifest_remaining32.json").read_text())["scenarios"]}
    full = {e["id"]: e for e in json.loads((STUDY48_META / "manifest.json").read_text())["scenarios"]}
    assert sorted(man) == rest and len(full) == 48
    for sid in rest:                                                      # byte-identical to a fresh generation
        assert (REMAINING32 / "configs" / f"{sid}.yaml").read_bytes() == (out / "configs" / f"{sid}.yaml").read_bytes()
        assert man[sid]["config_sha256"] == full[sid]["config_sha256"] == gen[sid]["config_sha256"]
    for sid in ("C04", "D11"):                                            # warm starts recomputed bit-identically
        cfg = load_config(str(REMAINING32 / "configs" / f"{sid}.yaml"))
        model = CertModel(cfg)
        fs, info = frozen_warmup(model, cfg.agents.initial_positions, man[sid]["init"], max_steps=6000)
        ref = pickle.load(open(REMAINING32 / "warm" / f"{sid}_warm.pkl", "rb"))
        rows = model.rows(cfg.agents.initial_positions)
        (p1, y1), (p2, y2) = rows.to_vec(fs), rows.to_vec(ref["warm"])
        assert np.array_equal(p1, p2) and np.array_equal(y1, y2) and info["stop_step"] == ref["stop_step"]
    pilot = {p.stem for p in (PILOT16 / "configs").glob("*.yaml")}
    assert pilot.isdisjoint(rest) and pilot | set(rest) == set(full)
    spec = load_spec(REMAINING32 / "batch/remaining32_batch.yaml")["scenarios"]
    smoke = load_spec(REMAINING32 / "batch/remaining32_smoke_batch.yaml")["scenarios"]
    assert sorted(s["id"] for s in spec) == rest and {s["id"] for s in smoke} <= set(rest)
    for s in spec + smoke:
        a = s["runner_args"]
        assert s["runner"] == "diagnostics.study48.run_pilot" and s["overrides"] == {}
        assert sha(Path(s["base_config"])) == man[s["id"]]["config_sha256"]
        assert "--no-strict" not in a and a[a.index("--snapshot-every") + 1] == "1"
        assert ("--warm" in a) == (s["id"] in ("C04", "D11"))
        assert ("--max-steps" in a) == (s in smoke)
    for p in (REMAINING32 / "batch").glob("*.yaml"):
        assert "/Users/" not in p.read_text() and "/home/" not in p.read_text()

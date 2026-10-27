"""Regression of the general certifier against the saved reference certificates of Experiments A, C and D3.

Read-only: uses the recorded trajectories, fast-state snapshots and reference outputs of the earlier
experiments. Skipped unless TCNS_RESEARCH_DATA points to the research data folder (the copy of the session
scratchpad, containing tgt/ and clfc/). Optionally writes a JSON comparison report to CERT_REGRESSION_REPORT.

References:
* A gated: tgt/t2_cert.txt (printed values; spectral-norm variant of the recursion).
* C gated: clfc/cert_C_gated.npz (full-precision arrays; Frobenius).
* D3:      clfc/D3cl_cert.npz (full-precision arrays; Frobenius) with the warm state clfc/D3cl_warm.pkl.
"""

from __future__ import annotations

import json
import os
import pickle
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from diagnostics.certificate import (CERT_OK, CertModel, CertTolerances, certify, observed_errors,
                                     zero_fast_state)
from models.fast_state import EdgeDualState, EdgeMismatchState, FastState

DATA = Path(os.environ["TCNS_RESEARCH_DATA"]) if os.environ.get("TCNS_RESEARCH_DATA") else None
pytestmark = pytest.mark.skipif(DATA is None or not DATA.exists(), reason="TCNS_RESEARCH_DATA not set")
REPORT: dict = {}


def _printed_close(value: float, ref: str) -> tuple[bool, float]:
    """|value - ref| <= half a unit in the last printed digit of ref."""
    d = Decimal(ref)
    tol = 0.5 * 10.0 ** d.as_tuple().exponent
    return abs(value - float(d)) <= tol * (1 + 1e-9), tol


def _fs_from_obs(obs, k, n_obs):
    N = obs["snap_w"].shape[1]
    return {i: FastState(w=tuple(map(float, obs["snap_w"][k, i])),
                         z_pair={j: EdgeMismatchState(float(obs["snap_z_pair"][k, i, j]), 0.0) for j in range(N) if j != i},
                         z_c=float(obs["snap_z_c"][k, i]), z_clf=float(obs["snap_z_clf"][k, i]),
                         y_pair={j: EdgeDualState(float(obs["snap_y_pair"][k, i, j]), 0.0) for j in range(N) if j != i},
                         y_c=float(obs["snap_y_c"][k, i]), y_clf=float(obs["snap_y_clf"][k, i]),
                         y_obs=[0.0] * n_obs, y_u=[0.0] * 4) for i in range(N)}


def _run(cfg_path, run_dir, chi0_fn, K=None, norm="frobenius"):
    cfg = load_config(str(cfg_path))
    model = CertModel(cfg)
    traj = json.load(open(run_dir / "trajectories.json"))
    K = K or traj["steps_completed"]
    pos = traj["trajectories"][:K]
    cert = certify(model, pos, chi0_fn(model), CertTolerances(norm=norm), check_model_at=(0, K // 2, K - 1))
    return model, pos, cert


def _compare_arrays(name, got, ref, rtol, atol):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    assert got.shape == ref.shape, (name, got.shape, ref.shape)
    err = np.abs(got - ref)
    lim = atol + rtol * np.abs(ref)
    REPORT.setdefault("arrays", {})[name] = {"max_abs": float(err.max()), "max_rel": float(np.max(err / np.maximum(np.abs(ref), 1e-300))),
                                             "rtol": rtol, "atol": atol, "ok": bool(np.all(err <= lim))}
    assert np.all(err <= lim), (name, float(err.max()))


@pytest.fixture(scope="module")
def cert_A():
    return _run(DATA / "tgt/A_gated.yaml", DATA / "tgt/run_A_gated", zero_fast_state, norm="spectral")


@pytest.fixture(scope="module")
def cert_C():
    return _run(DATA / "clfc/C_gated.yaml", DATA / "clfc/run_C_gated", zero_fast_state)


@pytest.fixture(scope="module")
def cert_D3():
    warm = pickle.load(open(DATA / "clfc/D3cl_warm.pkl", "rb"))["warm"]
    return _run(DATA / "clfc/D3_run.yaml", DATA / "clfc/run_D3", lambda m: warm)


def test_A_gated_against_printed_reference(cert_A):
    model, pos, cert = cert_A
    s, ini = cert.steps, cert.initial
    assert all(c["fast_p"] < 1e-9 and c["fast_g"] < 1e-9 and c["local_qp"] < 1e-12 for c in ini["model_check"])
    assert ini["degenerate"] == ["pair(0,2)", "pair(0,3)", "pair(1,3)", "clf(1)", "pair(2,0)", "clf(2)", "pair(3,0)",
                                 "pair(3,1)"]
    assert ini["active"] == ["clf(0)", "clf(3)"] and all(a == ["clf(0)", "clf(3)"] for a in s["active"])
    E, Eni, C1, C2, rho = (np.asarray(s[k]) for k in ("E", "E_ni", "C1", "C2", "rho"))
    ksw = np.prod(np.sqrt(np.asarray(s["kappa_sw"])))
    checks = {
        "sqrtU0": (ini["sqrtU0"], "3.803214e-03"), "e0": (ini["e0"], "3.817927e-03"),
        "variation": (float(np.sum(s["delta"])), "8.884096e-05"), "max_delta": (float(np.max(s["delta"])), "4.519e-08"),
        "a_min": (float(np.min(s["a"])), "1.9859"), "a_max": (float(np.max(s["a"])), "2.0311"),
        "kappa_min": (float(np.min(s["kappa_p"])), "4.6645e-03"), "kappa_max": (float(np.max(s["kappa_p"])), "4.6782e-03"),
        "c_min": (float(np.min(s["c"])), "2.924e-05"), "c_max": (float(np.max(s["c"])), "3.069e-05"),
        "E_max": (float(E.max()), "3.822675e-03"), "E_last": (float(E[-1]), "2.911e-03"),
        "Eni_max": (float(Eni.max()), "3.912412e-03"), "Eni_last": (float(Eni[-1]), "3.912e-03"),
        "switch_product": (float(ksw), "1.000114165"),
        "C1_over_E": (float(np.min(C1 / E)), "9.0"), "C2_over_E": (float(np.min(C2 / E)), "34.0"),
        "C2_over_Eni": (float(np.min(C2 / Eni)), "33.8"),
        "rho_min": (float(rho.min()), "1.33218e-01"), "rho_max": (float(rho.max()), "1.33375e-01"),
        "E_over_rho": (float(np.max(E / rho)), "2.8695e-02"), "Eni_over_rho": (float(np.max(Eni / rho)), "2.9334e-02"),
    }
    rep = {}
    for name, (val, ref) in checks.items():
        ok, t = _printed_close(val, ref)
        rep[name] = {"value": val, "reference": ref, "tol": t, "ok": ok}
    REPORT["A_gated"] = {"scalars": rep, "results": {v: r.as_dict() for v, r in cert.results.items()}}
    bad = {k: v for k, v in rep.items() if not v["ok"]}
    assert not bad, bad
    assert int(np.argmin(rho)) == 0
    assert all(r.status == CERT_OK for r in cert.results.values())
    obs = np.load(DATA / "tgt/obs_A_gated.npz")
    snaps = [_fs_from_obs(obs, k, 1) for k in range(len(pos))]
    oe = observed_errors(model, pos, snaps, len(pos), CertTolerances(norm="spectral"))
    for name, val, ref in (("obs_max", max(oe["err"]), "3.8179e-03"), ("obs_last", oe["err"][-1], "4.703e-07")):
        ok, t = _printed_close(val, ref)
        rep[name] = {"value": val, "reference": ref, "tol": t, "ok": ok}
        assert ok, (name, val, ref)
    assert int(np.argmax(oe["err"])) == 0
    assert np.all(np.asarray(oe["radius"]) <= np.asarray(cert.steps["E"]) * (1 + 1e-9))


def test_A_gated_frobenius_variant_is_reported(cert_A):
    model, pos, _ = cert_A
    cert = certify(model, pos, zero_fast_state(model), CertTolerances(norm="frobenius"))
    s = cert.steps
    REPORT["A_gated_frobenius"] = {"results": {v: r.as_dict() for v, r in cert.results.items()},
                                   "E_max": float(np.max(s["E"])), "Eni_max": float(np.max(s["E_ni"])),
                                   "Gamma_ni": float(np.max(np.asarray(s["E_ni"]) / np.asarray(s["E_adm"])))}
    assert all(r.status == CERT_OK for r in cert.results.values())


def test_C_gated_against_reference_arrays(cert_C):
    model, pos, cert = cert_C
    ref = np.load(DATA / "clfc/cert_C_gated.npz")
    s = cert.steps
    assert all(c["fast_p"] < 1e-9 and c["fast_g"] < 1e-9 and c["local_qp"] < 1e-12 for c in cert.initial["model_check"])
    assert all(a == ["clf(0)", "clf(1)", "clf(2)"] for a in s["active"]) and not cert.initial["degenerate"]
    _compare_arrays("C.E", s["E"], ref["E"], 1e-9, 1e-15)
    _compare_arrays("C.E_ni", s["E_ni"], ref["E_ni"], 1e-9, 1e-15)
    _compare_arrays("C.rho", s["rho"], ref["rho"], 1e-9, 1e-14)
    _compare_arrays("C.C1", s["C1"], ref["slk"], 1e-10, 1e-15)
    _compare_arrays("C.C2", s["C2"], ref["clip"], 1e-10, 1e-12)
    _compare_arrays("C.delta", s["delta"][1:], ref["dchi"], 1e-8, 1e-15)
    REPORT["C_gated"] = {"results": {v: r.as_dict() for v, r in cert.results.items()}, "summary": cert.summary()}
    assert all(r.status == CERT_OK for r in cert.results.values())
    obs = np.load(DATA / "clfc/obs_C_gated.npz")
    snaps = [_fs_from_obs(obs, k, 1) for k in range(len(pos))]
    oe = observed_errors(model, pos, snaps, len(pos), CertTolerances(norm="frobenius"))   # saved convention
    _compare_arrays("C.observed_err", oe["err"], ref["act_err"], 1e-8, 1e-15)


def test_D3_against_reference_arrays(cert_D3):
    model, pos, cert = cert_D3
    ref = np.load(DATA / "clfc/D3cl_cert.npz")
    s = cert.steps
    assert all(c["fast_p"] < 1e-9 and c["fast_g"] < 1e-9 and c["local_qp"] < 1e-12 for c in cert.initial["model_check"])
    assert all(a == ["pair(0,1)", "clf(0)", "conn(1)", "clf(1)"] for a in s["active"]) and not cert.initial["degenerate"]
    for name, got, key, rtol, atol in (("E", s["E"], "E", 1e-9, 1e-15), ("E_ni", s["E_ni"], "E_ni", 1e-9, 1e-15),
                                       ("E_adm", s["E_adm"], "Eadm", 1e-9, 1e-14), ("C1", s["C1"], "C1", 1e-10, 1e-15),
                                       ("C2", s["C2"], "C2", 1e-10, 1e-12), ("rho", s["rho"], "rho2", 1e-9, 1e-14),
                                       ("delta", s["delta"][1:], "dchi", 1e-8, 1e-15)):
        _compare_arrays(f"D3.{name}", got, ref[key], rtol, atol)
    REPORT["D3"] = {"results": {v: r.as_dict() for v, r in cert.results.items()}, "summary": cert.summary()}
    assert all(r.status == CERT_OK for r in cert.results.values())
    snaps = pickle.load(open(DATA / "clfc/D3cl_run_obs.pkl", "rb"))["snaps"]
    oe = observed_errors(model, pos, snaps, len(pos), CertTolerances(norm="frobenius"))   # saved convention
    _compare_arrays("D3.observed_err", oe["err"], ref["obs_e"], 1e-8, 1e-15)
    _compare_arrays("D3.observed_radius", oe["radius"], ref["obs_U"], 1e-8, 1e-15)


def teardown_module(module):
    path = os.environ.get("CERT_REGRESSION_REPORT")
    if path and REPORT:
        Path(path).write_text(json.dumps(REPORT, indent=2, default=str))

"""The direct-OSQP local QP solve must reproduce the former cvxpy(solver=OSQP) solve exactly."""

from __future__ import annotations

import contextlib
import io
import math
import time
import warnings
from typing import Any

import numpy as np
import pytest

import controller.local_qp as lq
from config import load_config
from controller.local_qp import LocalQPData, LocalQPSolveResult, build_local_qp_data, solve_local_qp_detailed
from models.agent import AgentState
from models.environment import CircularObstacle
from models.fast_state import init_fast_state_complete_graph

cp = pytest.importorskip("cvxpy")
osqp = pytest.importorskip("osqp")

I2 = [[1.0, 0.0], [0.0, 1.0]]
COMPARED = ("u", "success", "reason", "status", "num_iters", "pri_res", "dua_res")


def _cvxpy_reference(qp_data: LocalQPData) -> LocalQPSolveResult:
    """The solve_local_qp_detailed implementation that used cvxpy (verbatim logic)."""
    try:
        P = np.asarray(qp_data.P, dtype=float)
        q = np.asarray(qp_data.q, dtype=float)
        A = np.asarray(qp_data.A, dtype=float)
        b = np.asarray(qp_data.b, dtype=float)
        if P.shape != (2, 2) or q.shape != (2,):
            return LocalQPSolveResult((0.0, 0.0), False, f"invalid objective dimensions P={P.shape}, q={q.shape}", None)
        if A.size == 0:
            A = np.zeros((0, 2), dtype=float)
            b = np.zeros((0,), dtype=float)
        if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            return LocalQPSolveResult((0.0, 0.0), False, f"invalid inequality dimensions A={A.shape}, b={b.shape}", None)
        u = cp.Variable(2)
        objective = 0.5 * cp.quad_form(u, P) + q @ u
        constraints: list[Any] = []
        if A.shape[0] > 0:
            constraints.append(A @ u <= b)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        t0 = time.perf_counter()
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, polish=False)
        wall = time.perf_counter() - t0
        status = str(problem.status)
        stats = getattr(problem, "solver_stats", None)
        num_iters = getattr(stats, "num_iters", None)
        solve_time = getattr(stats, "solve_time", None)
        info = getattr(getattr(stats, "extra_stats", None), "info", None)
        pri = None if info is None else getattr(info, "prim_res", None)
        dua = None if info is None else getattr(info, "dual_res", None)
        details = dict(
            status=status,
            num_iters=int(num_iters) if num_iters is not None else None,
            solve_time=float(solve_time) if solve_time is not None else None,
            wall_time=wall,
            pri_res=float(pri) if pri is not None else None,
            dua_res=float(dua) if dua is not None else None,
        )
        if problem.status not in {"optimal", "optimal_inaccurate"} or u.value is None:
            return LocalQPSolveResult((0.0, 0.0), False, f"solver status={problem.status}", **details)
        return LocalQPSolveResult((float(u.value[0]), float(u.value[1])), True, status, **details)
    except Exception as exc:
        return LocalQPSolveResult((0.0, 0.0), False, f"exception: {exc}", "exception")


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _assert_identical(qp: LocalQPData) -> LocalQPSolveResult:
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        ref, new = _cvxpy_reference(qp), solve_local_qp_detailed(qp)
    for field in COMPARED:
        assert _same(getattr(ref, field), getattr(new, field)), (field, getattr(ref, field), getattr(new, field))
    return new


def _scaled(A: np.ndarray, b: np.ndarray) -> LocalQPData:
    s = np.maximum(np.maximum(np.linalg.norm(A, axis=1), np.abs(b)), 1e-8)  # _append_row scaling
    return LocalQPData(I2, [0.0, 0.0], (A / s[:, None]).tolist(), (b / s).tolist(), [f"r{k}" for k in range(len(b))])


def _random_qps(seed: int, n: int, eps_values=(1e-1, 1e-3, 1e-6, 0.0, -1e-6, -1e-3)):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        eps = eps_values[k % len(eps_values)]
        c = rng.normal(size=2) * rng.choice([1e-3, 0.05, 1.0])
        m = int(rng.integers(3, 9))
        ang = rng.uniform(0, 2 * np.pi, size=m)
        A = np.column_stack([np.cos(ang), np.sin(ang)]) * rng.uniform(0.1, 10, size=(m, 1))
        b = A @ c + eps * np.linalg.norm(A, axis=1)
        out.append(_scaled(A, b))
    return out


@pytest.mark.parametrize("qp", _random_qps(seed=1, n=120))
def test_matches_cvxpy_on_feasible_boundary_and_infeasible_qps(qp):
    _assert_identical(qp)


@pytest.mark.parametrize(
    "A,b,status",
    [
        ([[1.0, 0.0], [-1.0, 0.0]], [-1.0, -1.0], "infeasible"),
        ([[0.0, 0.0], [1.0, 0.0]], [-1.0, 1.0], "infeasible"),  # violated all-zero row
        ([[0.0, 0.0], [1.0, 0.0]], [0.0, -0.5], "optimal"),  # inactive CLF row 0^T u <= 0
        ([], [], "optimal"),  # no constraints
        ([[1.0, 0.0]], [math.inf], "optimal"),
        ([[1.0, 0.0]], [-math.inf], "exception"),
        ([[math.nan, 0.0]], [1.0], "exception"),
        ([[1.0, 0.0]], [math.nan], "exception"),
    ],
)
def test_matches_cvxpy_on_edge_cases(A, b, status):
    new = _assert_identical(LocalQPData(I2, [0.0, 0.0], A, b, [f"r{k}" for k in range(len(b))]))
    assert new.status == status


def test_matches_cvxpy_on_invalid_objective_data():
    _assert_identical(LocalQPData([[math.nan, 0.0], [0.0, 1.0]], [0.0, 0.0], [[1.0, 0.0]], [1.0], ["r"]))
    _assert_identical(LocalQPData([[1.0, 0.5], [0.0, 1.0]], [0.0, 0.0], [[1.0, 0.0]], [1.0], ["r"]))
    _assert_identical(LocalQPData(I2, [math.nan, 0.0], [[1.0, 0.0]], [1.0], ["r"]))
    _assert_identical(LocalQPData(I2, [0.3, -0.7], [[1.0, 1.0], [-1.0, 0.2]], [0.1, 0.4], ["a", "b"]))
    _assert_identical(LocalQPData([[1.0]], [0.0, 0.0], [[1.0, 0.0]], [1.0], ["r"]))


def test_matches_cvxpy_on_built_qps_with_zero_rows():
    cfg = load_config("configs/distributed_10_agents.yaml")
    c, f = cfg.controller, cfg.fast_dynamics
    obstacles = [CircularObstacle(tuple(o["center"]), float(o["radius"])) for o in cfg.obstacles.obstacles]
    # Agent 0 sits inside the target (inactive CLF row = 0); agent 3 is out of range (h_ij = 0 rows).
    positions = [(0.0, 0.0), (0.25, 0.0), (-0.3, 0.1), (2.0, 2.0), (0.1, -0.3)]
    agents = {i: AgentState(i, p, (0.0, 0.0)) for i, p in enumerate(positions)}
    fast = {i: init_fast_state_complete_graph(i, 5, len(obstacles)) for i in range(5)}
    for i in range(5):
        with contextlib.redirect_stdout(io.StringIO()):
            qp = build_local_qp_data(
                i=i, state=fast[i], all_fast_states=fast, all_agent_states=agents, obstacles=obstacles,
                u_max=c.u_max, gamma_1=c.cbf_rate, d_0=c.safety_distance, gamma_2=c.cbf_rate, eta=f.eta,
                d_c=f.d_c, epsilon_c=f.epsilon_c, gamma_3=c.clf_rate, target_center=tuple(cfg.target.center),
                sigma=cfg.target.Sigma, epsilon=cfg.target.epsilon, enable_connectivity_constraint=False,
            )
        if i == 0:
            assert [0.0, 0.0] in qp.A
        _assert_identical(qp)


@pytest.mark.parametrize("forced", [2, 4, 6, 7, 8, 10, 11, 99])
def test_status_mapping_matches_cvxpy(monkeypatch, forced):
    """Force each OSQP status code; cvxpy and the direct path must report it identically."""
    original = osqp.OSQP.solve

    def solve(self, *args, **kwargs):
        res = original(self, *args, **kwargs)
        res.info.status_val = forced
        return res

    monkeypatch.setattr(osqp.OSQP, "solve", solve)
    for qp in _random_qps(seed=2, n=4, eps_values=(1e-2,)):
        _assert_identical(qp)


def test_effective_osqp_settings_match_cvxpy(monkeypatch):
    """The settings OSQP actually runs with must be identical on both paths."""
    seen: list[dict] = []
    original = osqp.OSQP.setup

    def setup(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        st = self.settings
        seen.append({k: getattr(st, k) for k in dir(st) if not k.startswith("_") and not callable(getattr(st, k))})
        return out

    monkeypatch.setattr(osqp.OSQP, "setup", setup)
    qp = _random_qps(seed=3, n=1, eps_values=(1e-2,))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _cvxpy_reference(qp)
        solve_local_qp_detailed(qp)
    assert len(seen) == 2
    assert {k: str(v) for k, v in seen[0].items()} == {k: str(v) for k, v in seen[1].items()}
    assert seen[1]["polishing"] in (0, False) and seen[1]["eps_abs"] == 1e-5 and seen[1]["max_iter"] == 10000
    # The run metadata must describe exactly the settings OSQP ran with.
    meta = lq.local_qp_solver_metadata()
    assert meta["osqp_version"] == osqp.__version__
    for key, value in meta["settings"].items():
        assert seen[1][key] == value, key


def test_status_table_matches_cvxpy_osqp_interface():
    from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as CvxpyOSQP

    assert {k: str(v) for k, v in CvxpyOSQP.STATUS_MAP.items()} == lq._OSQP_STATUS

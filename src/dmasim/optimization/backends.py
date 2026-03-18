from __future__ import annotations

import numpy as np

from dmasim.optimization.qp_interface import QPProblem, QPSolver


class CVXPYSolver(QPSolver):
    def solve(self, problem: QPProblem) -> np.ndarray:
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError("cvxpy is required to solve QPs. Install dependencies first.") from exc

        n = problem.H.shape[0]
        z = cp.Variable(n)
        obj = 0.5 * cp.quad_form(z, problem.H) + problem.f @ z
        constraints = []
        if problem.A.size > 0:
            constraints.append(problem.A @ z <= problem.b)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, polish=False)
        if z.value is None:
            raise RuntimeError("QP solver failed to produce a solution")
        return np.asarray(z.value).reshape(-1)

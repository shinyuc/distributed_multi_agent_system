from __future__ import annotations

import numpy as np

from dmasim.config.schema import ControllerConfig
from dmasim.controllers.base import BaseCentralizedController
from dmasim.env.world import World
from dmasim.models.system_state import SystemState
from dmasim.optimization.backends import CVXPYSolver
from dmasim.optimization.qp_interface import QPProblem, QPSolver


class CentralizedQPController(BaseCentralizedController):
    """
    Global QP controller scaffold for baseline comparison.

    Implement your exact paper equations in _build_global_qp.
    """

    def __init__(self, config: ControllerConfig, solver: QPSolver | None = None):
        self.config = config
        self.solver = solver or CVXPYSolver()

    def compute_controls(self, state: SystemState, world: World, t: float) -> np.ndarray:
        qp = self._build_global_qp(state=state, world=world, t=t)
        z_star = self.solver.solve(qp)
        return z_star.reshape(state.n_agents, 2)

    def _build_global_qp(self, state: SystemState, world: World, t: float) -> QPProblem:
        # TODO(paper-eq): Replace this placeholder QP with the exact centralized QP from your paper.
        n_dec = 2 * state.n_agents
        H = np.eye(n_dec) * self.config.qp.weight_u
        f = np.zeros(n_dec)
        A = np.zeros((0, n_dec))
        b = np.zeros(0)
        return QPProblem(H=H, f=f, A=A, b=b)

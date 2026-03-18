from __future__ import annotations

import numpy as np

from dmasim.config.schema import ControllerConfig
from dmasim.controllers.base import BaseDistributedController
from dmasim.env.world import World
from dmasim.models.system_state import SystemState
from dmasim.optimization.backends import CVXPYSolver
from dmasim.optimization.qp_interface import QPProblem, QPSolver


class DistributedQPController(BaseDistributedController):
    """
    Local QP controller scaffold.

    Implement your exact paper equations in:
    - _build_local_qp
    - (optionally) helper methods in constraints.py / objectives.py
    """

    def __init__(self, config: ControllerConfig, solver: QPSolver | None = None):
        self.config = config
        self.solver = solver or CVXPYSolver()

    def compute_control(self, agent_id: int, state: SystemState, world: World, t: float) -> np.ndarray:
        qp = self._build_local_qp(agent_id=agent_id, state=state, world=world, t=t)
        z_star = self.solver.solve(qp)
        return z_star[:2]

    def _build_local_qp(self, agent_id: int, state: SystemState, world: World, t: float) -> QPProblem:
        # TODO(paper-eq): Replace this placeholder QP with the exact local QP from your paper.
        # Decision vector z = [u_x, u_y, slack? ...]
        n_dec = 2
        H = np.eye(n_dec) * self.config.qp.weight_u
        f = np.zeros(n_dec)
        A = np.zeros((0, n_dec))
        b = np.zeros(0)
        return QPProblem(H=H, f=f, A=A, b=b)

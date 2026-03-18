from dmasim.config.loader import load_config
from dmasim.controllers.distributed_qp import DistributedQPController
from dmasim.optimization.qp_interface import QPProblem, QPSolver
from dmasim.sim.simulator import Simulator


class ZeroSolver(QPSolver):
    def solve(self, problem: QPProblem):
        return [0.0, 0.0]


def test_simulator_runs_with_placeholder_qp():
    cfg = load_config("configs/base.yaml")
    ctrl = DistributedQPController(cfg.controller, solver=ZeroSolver())
    sim = Simulator(config=cfg, controller=ctrl)
    log = sim.run()
    assert len(log.times) == cfg.simulation.steps + 1

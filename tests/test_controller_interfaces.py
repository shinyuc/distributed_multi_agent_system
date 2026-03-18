from dmasim.config.loader import load_config
from dmasim.controllers.centralized_qp import CentralizedQPController
from dmasim.controllers.distributed_qp import DistributedQPController


def test_controller_construction():
    cfg = load_config("configs/base.yaml")
    assert isinstance(DistributedQPController(cfg.controller), DistributedQPController)
    assert isinstance(CentralizedQPController(cfg.controller), CentralizedQPController)

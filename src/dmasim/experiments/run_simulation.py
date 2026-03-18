from __future__ import annotations

import argparse
from pathlib import Path

from dmasim.config.loader import load_config
from dmasim.controllers.centralized_qp import CentralizedQPController
from dmasim.controllers.distributed_qp import DistributedQPController
from dmasim.sim.simulator import Simulator
from dmasim.viz.plotter import plot_trajectories



def _build_controller(cfg):
    method = cfg.controller.method.lower()
    if method == "distributed":
        return DistributedQPController(cfg.controller)
    if method == "centralized":
        return CentralizedQPController(cfg.controller)
    raise ValueError(f"Unknown controller method: {cfg.controller.method}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    controller = _build_controller(cfg)
    sim = Simulator(config=cfg, controller=controller)
    world = sim.build_world()
    log = sim.run()

    out = args.out
    if out is None:
        out = Path(cfg.simulation.output_dir) / "figures" / f"{cfg.controller.method}_trajectories.png"
    plot_trajectories(log=log, world=world, out_path=out)


if __name__ == "__main__":
    main()

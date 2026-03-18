from __future__ import annotations

import argparse
from pathlib import Path

from dmasim.config.loader import load_config
from dmasim.controllers.centralized_qp import CentralizedQPController
from dmasim.controllers.distributed_qp import DistributedQPController
from dmasim.sim.metrics import final_distance_to_target
from dmasim.sim.simulator import Simulator



def _run(cfg_path: str) -> float:
    cfg = load_config(cfg_path)
    method = cfg.controller.method.lower()
    if method == "distributed":
        controller = DistributedQPController(cfg.controller)
    elif method == "centralized":
        controller = CentralizedQPController(cfg.controller)
    else:
        raise ValueError(f"Unsupported method: {cfg.controller.method}")

    sim = Simulator(config=cfg, controller=controller)
    log = sim.run()
    final_positions = log.positions[-1]
    target = sim.build_world().target_region.center
    return final_distance_to_target(final_positions, target)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-config", required=True)
    parser.add_argument("--centralized-config", required=True)
    parser.add_argument("--out", default="outputs/logs/comparison.txt")
    args = parser.parse_args()

    dist_metric = _run(args.distributed_config)
    cent_metric = _run(args.centralized_config)

    lines = [
        f"distributed_final_mean_distance: {dist_metric:.6f}",
        f"centralized_final_mean_distance: {cent_metric:.6f}",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

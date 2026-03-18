from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmasim.env.world import World
from dmasim.sim.logger import TrajectoryLog
from dmasim.viz.styles import OBSTACLE_COLOR, TARGET_COLOR, TRAJ_COLOR



def plot_trajectories(log: TrajectoryLog, world: World, out_path: str | Path | None = None) -> None:
    positions = np.array(log.positions)
    n_agents = positions.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(n_agents):
        traj = positions[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], color=TRAJ_COLOR, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color="black", s=18)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=TRAJ_COLOR, s=18)

    for obs in world.obstacles:
        c = plt.Circle(obs.center, obs.radius, color=OBSTACLE_COLOR, alpha=0.25)
        ax.add_patch(c)

    tr = world.target_region
    target_circle = plt.Circle(tr.center, tr.radius, color=TARGET_COLOR, alpha=0.25)
    ax.add_patch(target_circle)

    x_min, x_max, y_min, y_max = world.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.set_title("Agent trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

"""Static trajectory plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from models.environment import Environment


def plot_trajectories(
    env: Environment,
    trajectories: list[list[tuple[float, float]]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot multi-agent trajectories in a 2D environment.

    Args:
        env: Environment containing agents, obstacles, and target region.
        trajectories: trajectories[t][i] = (x, y) position of agent i at time t.
    """

    if not trajectories:
        raise ValueError("trajectories must not be empty.")

    num_agents = env.num_agents()
    if any(len(step) != num_agents for step in trajectories):
        raise ValueError("Each trajectory time step must contain one position per agent.")

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(num_agents):
        xs = [trajectories[t][i][0] for t in range(len(trajectories))]
        ys = [trajectories[t][i][1] for t in range(len(trajectories))]
        ax.plot(xs, ys, linewidth=2, label=f"Agent {i}")
        ax.scatter(xs[0], ys[0], marker="o", s=35, color=ax.lines[-1].get_color())
        ax.scatter(xs[-1], ys[-1], marker="x", s=55, color=ax.lines[-1].get_color())

    for obs in env.obstacles:
        ax.add_patch(
            Circle(
                obs.center,
                obs.radius,
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
            )
        )

    sigma = getattr(env.target, "Sigma", None)
    if sigma is not None:
        sigma_np = np.asarray(sigma, dtype=float)
        if sigma_np.shape == (2, 2):
            evals, evecs = np.linalg.eigh(sigma_np)
            if np.all(evals > 0):
                # (x-c)^T Sigma (x-c) <= 1 => ellipse axes = 1/sqrt(lambda_i).
                axes = 1.0 / np.sqrt(evals)
                angle_deg = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
                ax.add_patch(
                    Ellipse(
                        xy=env.target.center,
                        width=2.0 * float(axes[0]),
                        height=2.0 * float(axes[1]),
                        angle=angle_deg,
                        facecolor="tab:green",
                        edgecolor="tab:green",
                        alpha=0.25,
                    )
                )
            else:
                ax.add_patch(
                    Circle(
                        env.target.center,
                        env.target.radius,
                        facecolor="tab:green",
                        edgecolor="tab:green",
                        alpha=0.25,
                    )
                )
        else:
            ax.add_patch(
                Circle(
                    env.target.center,
                    env.target.radius,
                    facecolor="tab:green",
                    edgecolor="tab:green",
                    alpha=0.25,
                )
            )
    else:
        ax.add_patch(
            Circle(
                env.target.center,
                env.target.radius,
                facecolor="tab:green",
                edgecolor="tab:green",
                alpha=0.25,
            )
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Agent Trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

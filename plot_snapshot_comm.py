from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from matplotlib.patches import Patch

# ============================================================
# User settings
# ============================================================

TRAJ_PATH = Path("outputs/20260326_165802/trajectories.json")
SAVE_PATH = Path("figs/trajectory_snapshots_comm_single_revised_5.png")

# manually selected steps
# SNAPSHOT_STEPS = [0, 75000, 150000]
SNAPSHOT_STEPS = [0, 50000, 75000, 250000]

# communication threshold
D_C = 0.9

# target region: (x-c)^T Sigma (x-c) <= 1
TARGET_CENTER = np.array([0.0, 0.0], dtype=float)
TARGET_SIGMA = np.array([
    [2.0, 0.0],
    [0.0, 0.25],
], dtype=float)

# obstacle list: (center, radius)
OBSTACLES = [
    ([-1.5, 0.6], 0.16),
    ([-1.1, 0.72], 0.22),
    ([-0.65, 1.0], 0.2),
    ([-1.25, 0.05], 0.1),
    ([-1.5, -0.6], 0.16),
    ([-1.1, -0.72], 0.22),
    ([-0.65, -1.0], 0.2),
]

# do not use blue for agents because blue is reserved for communication edges
AGENT_COLORS = [
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

COMM_COLOR = "tab:blue"


# ============================================================
# Helpers
# ============================================================

def load_trajectories(path: Path) -> np.ndarray:
    """
    Load trajectories from JSON.

    Expected JSON structure:
    {
        ...,
        "trajectories": [
            [[x1, y1], [x2, y2], ..., [xN, yN]],
            ...
        ]
    }

    Returns
    -------
    np.ndarray
        Shape (T, N, 2)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object to be a dict, got {type(data)}.")

    if "trajectories" not in data:
        raise ValueError("Key 'trajectories' not found in JSON file.")

    traj = np.asarray(data["trajectories"], dtype=float)

    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(
            f"Expected trajectory array with shape (T, N, 2), got {traj.shape}."
        )

    return traj


def add_target_region(ax: plt.Axes, center: np.ndarray, sigma: np.ndarray) -> None:
    """Draw ellipse defined by (x-c)^T Sigma (x-c) <= 1."""
    if sigma.shape != (2, 2):
        return

    evals, evecs = np.linalg.eigh(sigma)
    if not np.all(evals > 0):
        return

    axes = 1.0 / np.sqrt(evals)
    angle_deg = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))

    ax.add_patch(
        Ellipse(
            xy=tuple(center),
            width=2.0 * float(axes[0]),
            height=2.0 * float(axes[1]),
            angle=angle_deg,
            facecolor="tab:green",
            edgecolor="tab:green",
            alpha=0.20,
            linewidth=1.5,
            zorder=0,
        )
    )


def add_obstacles(ax: plt.Axes, obstacles: list[tuple[list[float], float]]) -> None:
    for center, radius in obstacles:
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
                zorder=0,
            )
        )

def get_comm_edges(
    positions: np.ndarray,  # shape (N, 2)
    d_c: float,
) -> list[tuple[int, int]]:
    """Return list of communication edges (i, j) with distance < d_c."""
    num_agents = positions.shape[0]
    edges = []

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < d_c:
                edges.append((i, j))

    return edges

def draw_comm_edges(
    ax: plt.Axes,
    positions: np.ndarray,   # shape (N, 2)
    d_c: float,
    color: str = "tab:blue",
    linewidth: float = 2.0,
) -> None:
    """Draw communication edges for pairs with distance < d_c."""
    num_agents = positions.shape[0]
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < d_c:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    zorder=1,
                )


def compute_axis_limits(
    trajectories: np.ndarray,
    obstacles: list[tuple[list[float], float]],
    target_center: np.ndarray,
    target_sigma: np.ndarray,
    padding: float = 0.25,
) -> tuple[float, float, float, float]:
    xs = trajectories[:, :, 0].reshape(-1)
    ys = trajectories[:, :, 1].reshape(-1)

    extra_x = [target_center[0]]
    extra_y = [target_center[1]]

    for center, radius in obstacles:
        cx, cy = center
        extra_x.extend([cx - radius, cx + radius])
        extra_y.extend([cy - radius, cy + radius])

    if target_sigma.shape == (2, 2):
        evals, _ = np.linalg.eigh(target_sigma)
        if np.all(evals > 0):
            axes = 1.0 / np.sqrt(evals)
            extra_x.extend([target_center[0] - axes[0], target_center[0] + axes[0]])
            extra_y.extend([target_center[1] - axes[1], target_center[1] + axes[1]])

    all_x = np.concatenate([xs, np.asarray(extra_x)])
    all_y = np.concatenate([ys, np.asarray(extra_y)])

    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))

    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


# ============================================================
# Main plotting
# ============================================================

def plot_snapshot_figure(
    trajectories: np.ndarray,
    snapshot_steps: list[int],
    d_c: float,
    target_center: np.ndarray,
    target_sigma: np.ndarray,
    obstacles: list[tuple[list[float], float]],
    save_path: Path,
) -> None:
    T, N, dim = trajectories.shape
    if dim != 2:
        raise ValueError("This plotting script only supports 2D trajectories.")

    if N > len(AGENT_COLORS):
        raise ValueError(
            f"Not enough preset agent colors for {N} agents. "
            f"Current color count = {len(AGENT_COLORS)}."
        )

    snapshot_steps = [int(s) for s in snapshot_steps]
    for s in snapshot_steps:
        if s < 0 or s >= T:
            raise ValueError(f"Snapshot step {s} is out of range [0, {T-1}].")

    fig, ax = plt.subplots(figsize=(8, 8))

    xmin, xmax, ymin, ymax = compute_axis_limits(
        trajectories=trajectories,
        obstacles=obstacles,
        target_center=target_center,
        target_sigma=target_sigma,
        padding=0.3,
    )

    # background
    add_target_region(ax, target_center, target_sigma)
    add_obstacles(ax, obstacles)

    # 1. full trajectories as dashed lines
    for i in range(N):
        color = AGENT_COLORS[i]
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"Agent {i}",
            zorder=2,
        )

    # 2. final positions as x
    final_positions = trajectories[-1]
    for i in range(N):
        # color = AGENT_COLORS[i]
        # ax.scatter(
        #     final_positions[i, 0],
        #     final_positions[i, 1],
        #     marker="x",
        #     s=90,
        #     color=color,
        #     linewidths=2.0,
        #     zorder=5,
        # )
        ax.plot(
        final_positions[i, 0],
        final_positions[i, 1],
        marker='x',
        markersize=10,        
        markeredgewidth=1.5, 
        color='black',
        linestyle='None',
        zorder=10,
        )

    # 3. selected snapshots, all using circular markers
    for step in snapshot_steps:
        current_positions = trajectories[step]

        # ===== NEW: print edges =====
        edges = get_comm_edges(current_positions, d_c)

        print(f"\n[Step {step}] Communication edges:")
        if not edges:
            print("  None")
        else:
            for (i, j) in edges:
                print(f"  ({i}, {j})")

        draw_comm_edges(
            ax=ax,
            positions=current_positions,
            d_c=d_c,
            color=COMM_COLOR,
            linewidth=2.0,
        )

        for i in range(N):
            color = AGENT_COLORS[i]
            ax.scatter(
                current_positions[i, 0],
                current_positions[i, 1],
                marker="o",
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

    ax.set_xlabel("$x_{i,1}$", fontsize=22)
    ax.set_ylabel("$x_{i,2}$", fontsize=22)
    # ax.set_title("Trajectories of Agents with Communication Edges", fontsize=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=20)

    # # legend
    # agent_handles = []
    # agent_labels = []
    # for i in range(N):
    #     h, = ax.plot([], [], linestyle="--", color=AGENT_COLORS[i], linewidth=2.0)
    #     agent_handles.append(h)
    #     agent_labels.append(f"Agent {i}")

    # comm_handle, = ax.plot([], [], linestyle="-", color=COMM_COLOR, linewidth=2.0)
    # final_handle = ax.scatter([], [], marker="x", s=90, color="black")

    # handles = agent_handles + [comm_handle, final_handle]
    # labels = agent_labels + ["Communication edge", "Final position"]

    # ax.legend(handles, labels, loc="best", frameon=True)
    from matplotlib.patches import Patch

    # agent legend
    agent_handles = []
    agent_labels = []
    for i in range(N):
        h, = ax.plot([], [], linestyle="--", color=AGENT_COLORS[i], linewidth=2.0)
        agent_handles.append(h)
        agent_labels.append(f"Agent {i}")

    # communication + final
    comm_handle, = ax.plot([], [], linestyle="-", color=COMM_COLOR, linewidth=2.0)
    # final_handle = ax.scatter([], [], marker="x", s=90, color="black")
    final_handle, = ax.plot(
                            [], [],
                            marker='x',
                            markersize=10,
                            markeredgewidth=1.5,
                            color='black',
                            linestyle='None'
                            )

    # NEW: obstacle + target (use color only, no description words)
    obstacle_patch = Patch(facecolor="tab:red", edgecolor="tab:red", alpha=0.25)
    target_patch = Patch(facecolor="tab:green", edgecolor="tab:green", alpha=0.20)

    handles = (
        agent_handles
        + [comm_handle, final_handle, obstacle_patch, target_patch]
    )

    labels = (
        agent_labels
        + ["Communication edge", "Final position", "Obstacle", "Target region"]
    )

    # ax.legend(handles, labels, loc="best", frameon=True)
    ax.legend(handles, labels, loc="best", frameon=True, fontsize=12, framealpha=0.9)
    

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")


def main() -> None:
    trajectories = load_trajectories(TRAJ_PATH)
    plot_snapshot_figure(
        trajectories=trajectories,
        snapshot_steps=SNAPSHOT_STEPS,
        d_c=D_C,
        target_center=TARGET_CENTER,
        target_sigma=TARGET_SIGMA,
        obstacles=OBSTACLES,
        save_path=SAVE_PATH,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

TRAJ_PATH = Path("outputs/20260326_165802/trajectories.json")
SAVE_PATH = Path("figs/lambda2_binary_comm_revised_3.png")

# communication threshold
D_C = 0.9

# ✅ Use physical time (seconds)
USE_PHYSICAL_TIME = True


# ============================================================
# Helpers
# ============================================================

def load_sim_data(path: Path) -> tuple[np.ndarray, float | None]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trajectories = np.asarray(data["trajectories"], dtype=float)

    dt = data.get("dt", None)
    if dt is not None:
        dt = float(dt)

    return trajectories, dt


def build_binary_adjacency(positions: np.ndarray, d_c: float) -> np.ndarray:
    num_agents = positions.shape[0]
    A = np.zeros((num_agents, num_agents), dtype=float)

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < d_c:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A


def build_laplacian(A: np.ndarray) -> np.ndarray:
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A


def compute_lambda2_from_positions(positions: np.ndarray, d_c: float) -> float:
    A = build_binary_adjacency(positions, d_c)
    L = build_laplacian(A)

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

    eigvals[np.abs(eigvals) < 1e-12] = 0.0

    return float(eigvals[1])


def compute_lambda2_series(trajectories: np.ndarray, d_c: float) -> np.ndarray:
    num_steps = trajectories.shape[0]
    lambda2_values = np.zeros(num_steps)

    for t in range(num_steps):
        lambda2_values[t] = compute_lambda2_from_positions(
            trajectories[t], d_c
        )

    return lambda2_values


def plot_lambda2(
    lambda2_values: np.ndarray,
    save_path: Path,
    dt: float | None = None,
    use_physical_time: bool = False,
) -> None:
    num_steps = len(lambda2_values)

    # ✅ Time axis instead of step
    if use_physical_time and dt is not None:
        x = np.arange(num_steps) * dt
        xlabel = "Time (s)"
    else:
        x = np.arange(num_steps)
        xlabel = "Step"

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(x, lambda2_values, linewidth=2.0)

    # ax.set_xlabel(xlabel, fontsize=14)
    # ax.set_ylabel(r"$\overline{\lambda}_2$", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(r"$\overline{\lambda}_2$", fontsize=22)
    # ax.set_title(r"Evolution of $\overline{\lambda}_2$ for the Communication Graph", fontsize=16)
        
    # ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', labelsize=20)


    # ✅ Make sure axis starts at 0
    ax.set_xlim(0, x[-1])

    # ✅ Optional: nicer ticks for paper
    if use_physical_time and dt is not None:
        ax.set_xticks(np.linspace(0, x[-1], 6))  # e.g., 0,60,120,...,300

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    trajectories, dt = load_sim_data(TRAJ_PATH)

    print(f"Loaded trajectories from: {TRAJ_PATH}")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"dt = {dt}")

    lambda2_values = compute_lambda2_series(trajectories, D_C)

    print(f"Computed lambda_2 for {len(lambda2_values)} steps.")
    print(f"Min lambda_2: {np.min(lambda2_values):.6f}")
    print(f"Max lambda_2: {np.max(lambda2_values):.6f}")
    print(f"Final lambda_2: {lambda2_values[-1]:.6f}")

    plot_lambda2(
        lambda2_values=lambda2_values,
        save_path=SAVE_PATH,
        dt=dt,
        use_physical_time=USE_PHYSICAL_TIME,
    )


if __name__ == "__main__":
    main()
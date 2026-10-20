from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

TRAJ_PATH = Path("outputs/20260326_165802/trajectories.json")
SAVE_PATH = Path("figs/lambda2_binary_comm.png")

# communication threshold
D_C = 0.9

# if True, use physical time t = step * dt on x-axis when dt is available
USE_PHYSICAL_TIME = False


# ============================================================
# Helpers
# ============================================================

def load_sim_data(path: Path) -> tuple[np.ndarray, float | None]:
    """
    Load trajectories and dt from JSON.

    Expected JSON structure:
    {
        ...,
        "dt": ...,
        "trajectories": [
            [[x1, y1], [x2, y2], ..., [xN, yN]],
            ...
        ]
    }

    Returns
    -------
    trajectories : np.ndarray
        Shape (T, N, 2)
    dt : float | None
        Time step if available
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object to be a dict, got {type(data)}.")

    if "trajectories" not in data:
        raise ValueError("Key 'trajectories' not found in JSON file.")

    trajectories = np.asarray(data["trajectories"], dtype=float)

    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError(
            f"Expected trajectory array with shape (T, N, 2), got {trajectories.shape}."
        )

    dt = data.get("dt", None)
    if dt is not None:
        dt = float(dt)

    return trajectories, dt


def build_binary_adjacency(positions: np.ndarray, d_c: float) -> np.ndarray:
    """
    Construct binary adjacency matrix A for a communication graph:
        a_ij = 1, if ||x_i - x_j|| < d_c and i != j
        a_ij = 0, otherwise

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 2)
    d_c : float
        Communication threshold

    Returns
    -------
    A : np.ndarray
        Shape (N, N)
    """
    num_agents = positions.shape[0]
    A = np.zeros((num_agents, num_agents), dtype=float)

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < d_c:  # strictly less than
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A


def build_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Build graph Laplacian L = D - A.
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A
    return L


def compute_lambda2_from_positions(positions: np.ndarray, d_c: float) -> float:
    """
    Compute lambda_2 of the binary communication graph at one time step.
    """
    A = build_binary_adjacency(positions, d_c)
    L = build_laplacian(A)

    # L is symmetric, so use eigvalsh
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

    # Remove tiny numerical negatives
    eigvals[np.abs(eigvals) < 1e-12] = 0.0

    if len(eigvals) < 2:
        raise ValueError("Need at least 2 agents to compute lambda_2.")

    return float(eigvals[1])


def compute_lambda2_series(trajectories: np.ndarray, d_c: float) -> np.ndarray:
    """
    Compute lambda_2 for all time steps.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (T, N, 2)

    Returns
    -------
    lambda2_values : np.ndarray
        Shape (T,)
    """
    num_steps = trajectories.shape[0]
    lambda2_values = np.zeros(num_steps, dtype=float)

    for t in range(num_steps):
        positions = trajectories[t]
        lambda2_values[t] = compute_lambda2_from_positions(positions, d_c)

    return lambda2_values


def plot_lambda2(
    lambda2_values: np.ndarray,
    save_path: Path,
    dt: float | None = None,
    use_physical_time: bool = False,
) -> None:
    """
    Plot lambda_2 over time.
    """
    num_steps = len(lambda2_values)

    if use_physical_time and dt is not None:
        x = np.arange(num_steps) * dt
        xlabel = "Time (s)"
    else:
        x = np.arange(num_steps)
        xlabel = "Step"

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(x, lambda2_values, linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\overline{\lambda}_2$")
    ax.set_title(r"Evolution of $\overline{\lambda}_2$ for the Communication Graph")
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

    lambda2_values = compute_lambda2_series(trajectories, D_C)

    print(f"Computed lambda_2 for {len(lambda2_values)} time steps.")
    print(f"Minimum lambda_2: {np.min(lambda2_values):.6f}")
    print(f"Maximum lambda_2: {np.max(lambda2_values):.6f}")
    print(f"Final lambda_2:   {lambda2_values[-1]:.6f}")

    plot_lambda2(
        lambda2_values=lambda2_values,
        save_path=SAVE_PATH,
        dt=dt,
        use_physical_time=USE_PHYSICAL_TIME,
    )


if __name__ == "__main__":
    main()
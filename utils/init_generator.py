"""Initial-condition generation and validation utilities.

This module centralizes initialization logic for agent positions so config
loading can support both manual and generated initial conditions while keeping
the rest of the simulation code unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InitRegion:
    """Ellipsoidal initialization region.

    Points satisfy ``(x - center)^T Sigma (x - center) <= radius^2``.
    """

    center: list[float]
    Sigma: list[list[float]]
    radius: float
    check_manual_positions: bool = False
    save_dir: str = "outputs/init_conditions"
    max_sampling_attempts: int = 2000
    max_global_restarts: int = 50


def _as_position_array(positions: list[list[float]]) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "agents.initial_positions must be a list of 2D points with shape (N, 2)."
        )
    return array


def _region_arrays(region: InitRegion) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(region.center, dtype=float)
    sigma = np.asarray(region.Sigma, dtype=float)
    if center.shape != (2,):
        raise ValueError("agents.init_region.center must be a length-2 list.")
    if sigma.shape != (2, 2):
        raise ValueError("agents.init_region.Sigma must be a 2x2 matrix.")
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("agents.init_region.Sigma must be symmetric.")
    eigvals = np.linalg.eigvalsh(sigma)
    if np.min(eigvals) <= 0.0:
        raise ValueError("agents.init_region.Sigma must be positive definite.")
    if region.radius <= 0.0:
        raise ValueError(f"agents.init_region.radius must be > 0, got {region.radius}.")
    return center, sigma


def point_in_region(point: np.ndarray, region: InitRegion, tol: float = 1e-12) -> bool:
    """Return whether a 2D point lies in the ellipsoidal region."""

    center, sigma = _region_arrays(region)
    delta = point - center
    value = float(delta.T @ sigma @ delta)
    return value <= float(region.radius * region.radius) + tol


def _sample_in_unit_disk(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    radius = np.sqrt(rng.uniform(0.0, 1.0))
    return radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)


def sample_point_in_region(region: InitRegion, rng: np.random.Generator) -> np.ndarray:
    """Sample a point inside the ellipsoidal region."""

    center, sigma = _region_arrays(region)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    transform = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    ball_point = region.radius * _sample_in_unit_disk(rng)
    return center + transform @ ball_point


def _connectivity_threshold(d_c: float, epsilon_c: float) -> float:
    threshold = d_c - epsilon_c
    if threshold <= 0.0:
        raise ValueError(
            "Initialization connectivity threshold must satisfy d_c - epsilon_c > 0, "
            f"got d_c={d_c}, epsilon_c={epsilon_c}."
        )
    return threshold


def adjacency_matrix(positions: np.ndarray, d_c: float, epsilon_c: float) -> list[list[int]]:
    """Build the undirected adjacency matrix for Graph B using ``d <= d_c - epsilon_c``."""

    n_agents = positions.shape[0]
    threshold = _connectivity_threshold(d_c=d_c, epsilon_c=epsilon_c)
    adjacency = [[0 for _ in range(n_agents)] for _ in range(n_agents)]
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist <= threshold:
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    return adjacency


def is_connected(adjacency: list[list[int]]) -> bool:
    """Check connectivity of an undirected graph given its adjacency matrix."""

    n_agents = len(adjacency)
    if n_agents == 0:
        return True

    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor, has_edge in enumerate(adjacency[node]):
            if has_edge and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == n_agents


def _minimum_pairwise_distance(positions: np.ndarray) -> float | None:
    if positions.shape[0] < 2:
        return None

    min_distance: float | None = None
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if min_distance is None or distance < min_distance:
                min_distance = distance
    return min_distance


def summarize_initialization_record(record: dict[str, Any]) -> dict[str, float | int | bool | None]:
    """Summarize graph and spacing statistics from an initialization record.

    The summary is derived from the stored agent positions and adjacency matrix.
    Edge statistics only use edges present in ``adjacency_matrix``. If the graph
    has no edges, the edge-distance fields are returned as ``None``.
    """

    positions = np.asarray(record["positions"], dtype=float)
    adjacency = record["adjacency_matrix"]
    num_agents = int(record["num_agents"])

    pairwise_distances: list[float] = []
    edge_distances: list[float] = []
    degree_sum = 0
    number_of_edges = 0
    for i in range(num_agents):
        row_degree = int(sum(adjacency[i]))
        degree_sum += row_degree
        for j in range(i + 1, num_agents):
            pairwise_distances.append(float(np.linalg.norm(positions[i] - positions[j])))
            if adjacency[i][j]:
                number_of_edges += 1
                edge_distances.append(float(np.linalg.norm(positions[i] - positions[j])))

    return {
        "num_agents": num_agents,
        "connected": bool(record["is_connected"]),
        "minimum_pairwise_distance": min(pairwise_distances) if pairwise_distances else None,
        "number_of_edges": number_of_edges,
        "average_degree": (degree_sum / num_agents) if num_agents > 0 else 0.0,
        "min_edge_distance": min(edge_distances) if edge_distances else None,
        "max_edge_distance": max(edge_distances) if edge_distances else None,
    }


def _build_record(
    *,
    mode: str,
    seed: int,
    positions: np.ndarray,
    region: InitRegion | None,
    safety_distance: float,
    d_c: float,
    epsilon_c: float,
) -> dict[str, Any]:
    adjacency = adjacency_matrix(positions, d_c=d_c, epsilon_c=epsilon_c)
    connected = is_connected(adjacency)
    min_distance = _minimum_pairwise_distance(positions)
    return {
        "mode": mode,
        "seed": seed,
        "num_agents": int(positions.shape[0]),
        "positions": positions.tolist(),
        "region": None
        if region is None
        else {
            "center": list(region.center),
            "Sigma": [list(row) for row in region.Sigma],
            "radius": float(region.radius),
            "check_manual_positions": bool(region.check_manual_positions),
        },
        "safety_distance": float(safety_distance),
        "d_c": float(d_c),
        "epsilon_c": float(epsilon_c),
        "connectivity_threshold": float(_connectivity_threshold(d_c=d_c, epsilon_c=epsilon_c)),
        "adjacency_matrix": adjacency,
        "is_connected": connected,
        "minimum_pairwise_distance": min_distance,
    }


def validate_initial_positions(
    *,
    positions: list[list[float]],
    num_agents: int,
    safety_distance: float,
    d_c: float,
    epsilon_c: float,
    mode: str,
    seed: int,
    region: InitRegion | None = None,
) -> dict[str, Any]:
    """Validate initial positions and return a debug record."""

    if safety_distance <= 0.0:
        raise ValueError(
            f"controller.safety_distance must be > 0, got {safety_distance}."
        )
    if d_c <= 0.0:
        raise ValueError(f"fast_dynamics.d_c must be > 0, got {d_c}.")
    if epsilon_c < 0.0:
        raise ValueError(f"fast_dynamics.epsilon_c must be >= 0, got {epsilon_c}.")
    _connectivity_threshold(d_c=d_c, epsilon_c=epsilon_c)

    position_array = _as_position_array(positions)
    if position_array.shape[0] != num_agents:
        raise ValueError(
            "agents.num_agents must match len(agents.initial_positions): "
            f"{num_agents} != {position_array.shape[0]}"
        )

    # should_check_region = region is not None and (
    #     mode == "random_region" or region.check_manual_positions
    # )
    should_check_region = region is not None and mode == "random_region" 
    if should_check_region:
        for idx, point in enumerate(position_array):
            if not point_in_region(point, region):
                raise ValueError(
                    f"agents.initial_positions[{idx}] lies outside agents.init_region."
                )

    min_distance = _minimum_pairwise_distance(position_array)
    if min_distance is not None and min_distance <= safety_distance:
        raise ValueError(
            "Initial positions must satisfy pairwise distance > "
            f"controller.safety_distance={safety_distance}, got minimum distance {min_distance:.6f}."
        )

    record = _build_record(
        mode=mode,
        seed=seed,
        positions=position_array,
        region=region,
        safety_distance=safety_distance,
        d_c=d_c,
        epsilon_c=epsilon_c,
    )
    if not bool(record["is_connected"]):
        raise ValueError(
            "Initial communication graph under Graph B (d <= d_c - epsilon_c) must be connected."
        )
    return record


def generate_initial_positions(
    *,
    num_agents: int,
    region: InitRegion,
    safety_distance: float,
    d_c: float,
    epsilon_c: float,
    seed: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Generate connected, collision-safe initial positions inside the region."""

    if num_agents <= 0:
        raise ValueError(f"agents.num_agents must be > 0, got {num_agents}.")

    _region_arrays(region)
    threshold = _connectivity_threshold(d_c=d_c, epsilon_c=epsilon_c)
    rng = np.random.default_rng(seed)

    for _restart_idx in range(region.max_global_restarts):
        accepted: list[np.ndarray] = []
        failed = False

        for agent_idx in range(num_agents):
            placed = False
            for _attempt_idx in range(region.max_sampling_attempts):
                candidate = sample_point_in_region(region, rng)
                if not accepted:
                    accepted.append(candidate)
                    placed = True
                    break

                distances = [float(np.linalg.norm(candidate - point)) for point in accepted]
                if any(distance <= safety_distance for distance in distances):
                    continue
                if not any(distance <= threshold for distance in distances):
                    continue

                accepted.append(candidate)
                placed = True
                break

            if not placed:
                failed = True
                break

            if len(accepted) != agent_idx + 1:
                failed = True
                break

        if failed:
            continue

        positions = np.vstack(accepted)
        record = validate_initial_positions(
            positions=positions.tolist(),
            num_agents=num_agents,
            safety_distance=safety_distance,
            d_c=d_c,
            epsilon_c=epsilon_c,
            mode="random_region",
            seed=seed,
            region=region,
        )
        return positions.tolist(), record

    raise ValueError(
        "Failed to generate valid initial positions in agents.init_region after "
        f"{region.max_global_restarts} restarts with "
        f"{region.max_sampling_attempts} samples per agent."
    )


def save_initialization_record(
    *,
    record: dict[str, Any],
    region: InitRegion | None,
    config_path: str | Path,
) -> str:
    """Persist initialization metadata to disk and return the JSON path."""

    save_dir = Path(region.save_dir if region is not None else "outputs/init_conditions")
    save_dir.mkdir(parents=True, exist_ok=True)

    config_stem = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{config_stem}_{record['mode']}_seed{record['seed']}_{timestamp}_p{os.getpid()}"
    # Exclusive create ("x"): two processes can never write the same record file.
    for attempt in range(100):
        output_path = save_dir / (f"{base}.json" if attempt == 0 else f"{base}_{attempt}.json")
        try:
            with output_path.open("x", encoding="utf-8") as file_obj:
                json.dump(record, file_obj, indent=2)
        except FileExistsError:
            continue
        return str(output_path)
    raise RuntimeError(f"Could not create a unique initialization record in {save_dir}.")

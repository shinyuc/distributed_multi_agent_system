from pathlib import Path

from config import load_config
from utils.init_generator import (
    InitRegion,
    generate_initial_positions,
    summarize_initialization_record,
    validate_initial_positions,
)


def test_generate_initial_positions_satisfies_constraints():
    region = InitRegion(
        center=[0.0, 0.0],
        Sigma=[[1.0, 0.0], [0.0, 1.0]],
        radius=1.0,
        check_manual_positions=True,
        max_sampling_attempts=500,
        max_global_restarts=10,
    )

    positions, record = generate_initial_positions(
        num_agents=4,
        region=region,
        safety_distance=0.3,
        d_c=0.9,
        seed=7,
    )

    assert len(positions) == 4
    assert record["is_connected"] is True
    assert record["minimum_pairwise_distance"] > 0.3
    assert sum(sum(row) for row in record["adjacency_matrix"]) > 0


def test_validate_initial_positions_rejects_disconnected_graph():
    region = InitRegion(
        center=[0.0, 0.0],
        Sigma=[[1.0, 0.0], [0.0, 1.0]],
        radius=5.0,
        check_manual_positions=True,
    )

    try:
        validate_initial_positions(
            positions=[[0.0, 0.0], [2.0, 0.0], [4.5, 0.0]],
            num_agents=3,
            safety_distance=0.3,
            d_c=1.0,
            mode="manual",
            seed=42,
            region=region,
        )
    except ValueError as exc:
        assert "connected" in str(exc)
    else:
        raise AssertionError("Expected disconnected positions to fail validation.")


def test_load_config_random_region_generates_positions_and_saves_log():
    cfg = load_config("configs/distributed_10_agents.yaml")

    assert cfg.agents.init_mode == "random_region"
    assert len(cfg.agents.initial_positions) == cfg.agents.num_agents
    assert cfg.agents.initialization_log_path is not None
    assert Path(cfg.agents.initialization_log_path).is_file()


def test_generate_initial_positions_fails_gracefully_for_impossible_geometry():
    region = InitRegion(
        center=[0.0, 0.0],
        Sigma=[[1.0, 0.0], [0.0, 1.0]],
        radius=0.1,
        max_sampling_attempts=20,
        max_global_restarts=3,
    )

    try:
        generate_initial_positions(
            num_agents=2,
            region=region,
            safety_distance=0.5,
            d_c=0.9,
            seed=11,
        )
    except ValueError as exc:
        assert "Failed to generate valid initial positions" in str(exc)
    else:
        raise AssertionError("Expected impossible random-region geometry to fail.")


def test_summarize_initialization_record_handles_edge_statistics():
    record = {
        "num_agents": 3,
        "positions": [[0.0, 0.0], [0.6, 0.0], [0.6, 0.8]],
        "adjacency_matrix": [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        "minimum_pairwise_distance": 0.6,
        "is_connected": True,
    }

    summary = summarize_initialization_record(record)

    assert summary["num_agents"] == 3
    assert summary["connected"] is True
    assert summary["minimum_pairwise_distance"] == 0.6
    assert summary["number_of_edges"] == 2
    assert summary["average_degree"] == 4 / 3
    assert summary["min_edge_distance"] == 0.6
    assert summary["max_edge_distance"] == 0.8

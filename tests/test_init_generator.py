from pathlib import Path
import textwrap

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
        epsilon_c=0.1,
        seed=7,
    )

    assert len(positions) == 4
    assert record["is_connected"] is True
    assert record["minimum_pairwise_distance"] > 0.3
    assert record["connectivity_threshold"] == 0.8
    assert sum(sum(row) for row in record["adjacency_matrix"]) > 0


def test_validate_initial_positions_rejects_graph_b_disconnected_graph():
    region = InitRegion(
        center=[0.0, 0.0],
        Sigma=[[1.0, 0.0], [0.0, 1.0]],
        radius=5.0,
        check_manual_positions=True,
    )

    try:
        validate_initial_positions(
            positions=[[0.0, 0.0], [0.85, 0.0], [1.70, 0.0]],
            num_agents=3,
            safety_distance=0.3,
            d_c=1.0,
            epsilon_c=0.2,
            mode="manual",
            seed=42,
            region=region,
        )
    except ValueError as exc:
        assert "Graph B" in str(exc)
    else:
        raise AssertionError("Expected Graph B disconnected positions to fail validation.")


def test_load_config_random_region_generates_positions_and_saves_log(tmp_path):
    config_path = tmp_path / "random_region.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            simulation:
              dt: 0.001
              steps: 10
              seed: 42

            agents:
              num_agents: 4
              init_mode: random_region
              initial_positions:
                - [0.0, 0.0]
                - [0.1, 0.0]
                - [0.2, 0.0]
                - [0.3, 0.0]
              init_region:
                center: [0.0, 0.0]
                Sigma:
                  - [1.0, 0.0]
                  - [0.0, 1.0]
                radius: 1.0
                save_dir: outputs/init_conditions
                max_sampling_attempts: 200
                max_global_restarts: 10

            obstacles:
              obstacles: []

            target:
              center: [0.0, 0.0]
              radius: 1.0
              Sigma:
                - [1.0, 0.0]
                - [0.0, 1.0]
              epsilon: 0.2

            controller:
              mode: distributed
              u_max: 1.0
              safety_distance: 0.2
              clf_rate: 0.1
              cbf_rate: 1.0
              nominal_gain: 1.0

            fast_dynamics:
              tau: 0.01
              xi: 0.5
              d_c: 0.9
              eta: 0.1
              epsilon_c: 0.1
            """
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

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
            epsilon_c=0.1,
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

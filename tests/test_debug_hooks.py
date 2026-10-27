"""Hard-coded diagnostic prints in the distributed controller must not depend on the number of agents.

The ``debug_pair_12`` block of DistributedController.compute_control is active at steps 207-208 for agents 1 and 2
and prints the pair (1, 2). With two agents (ids 0, 1), agent 2 does not exist (pilot D03 stopped with KeyError: 2).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import yaml

from config import load_config
from diagnostics.run_feasibility import build

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, positions, steps):
    raw = yaml.safe_load((REPO / "configs/distributed_10_agents.yaml").read_text())
    raw["simulation"].update(dt=0.001, steps=steps)
    raw["agents"].update(num_agents=len(positions), initial_positions=positions, init_mode="manual")
    raw["obstacles"]["obstacles"] = []
    raw["controller"].update({"clf_coupling": "gated"})
    raw["fast_dynamics"].update(substeps=10)
    p = tmp_path / f"cfg_{len(positions)}.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _control(sim, agent_id, step_idx):
    """One compute_control call with exactly the simulator's keyword set (simulation/simulator.py)."""
    agents = sim._agent_state_map()
    _, _, lam2, v2 = sim._compute_connectivity_runtime_spectrum(agents)
    fs = sim.initialize_fast_states()
    agent = next(a for a in sim.agents if a.agent_id == agent_id)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        u = sim.controller.compute_control(
            agent, sim.env, step_idx=step_idx, fast_state=fs[agent_id], all_fast_states=fs, all_agent_states=agents,
            d_c=sim.fast_d_c, gamma_1=sim.fast_gamma_1, d_0=sim.fast_d_0, target_center=sim.fast_target_center,
            sigma=sim.fast_sigma, epsilon=sim.fast_epsilon, n_agents=len(sim.agents), eta=sim.fast_eta,
            epsilon_c=sim.fast_epsilon_c, lambda_2=lam2, v2=v2)
    return u, out.getvalue()


@pytest.mark.parametrize("positions, prints", [([[-0.75, 0.0], [-0.25, 0.0]], False),
                                               ([[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.05]], True)])
def test_pair12_debug_block_is_print_only_and_safe_for_any_n(tmp_path, positions, prints):
    _, _, sim = build(_cfg(tmp_path, positions, steps=1))
    for step in (207, 208):                                 # the debug block is active only at these steps
        u_dbg, text = _control(sim, 1, step)
        u_ref, _ = _control(sim, 1, 206)                    # same state, debug block inactive
        assert u_dbg == u_ref                               # the debug block never changes the control
        assert ("[DEBUG pair12 cbf]" in text) == prints     # printed only when agent 2 exists


def test_two_agent_simulation_passes_steps_207_and_208(tmp_path):
    _, _, sim = build(_cfg(tmp_path, [[-0.75, 0.0], [-0.25, 0.0]], steps=210))
    with contextlib.redirect_stdout(io.StringIO()):
        res = sim.run()
    assert res.termination_exception is None, res.termination_exception
    assert res.termination_reason == "completed" and res.horizon_completed and len(res.trajectories) == 211

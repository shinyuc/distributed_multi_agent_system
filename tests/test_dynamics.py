import numpy as np

from dmasim.models.agent import Agent


def test_single_integrator_step():
    a = Agent(agent_id=0, position=np.array([0.0, 0.0]), radius=0.1, max_speed=10.0)
    a.step(np.array([1.0, -2.0]), dt=0.1)
    assert np.allclose(a.position, np.array([0.1, -0.2]))

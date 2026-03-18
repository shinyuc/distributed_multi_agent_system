from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dmasim.config.schema import RootConfig
from dmasim.controllers.base import BaseCentralizedController, BaseDistributedController
from dmasim.env.obstacle import CircularObstacle
from dmasim.env.target_region import CircularTargetRegion
from dmasim.env.world import World
from dmasim.models.agent import Agent
from dmasim.models.system_state import SystemState
from dmasim.sim.logger import TrajectoryLog


@dataclass
class Simulator:
    config: RootConfig
    controller: BaseDistributedController | BaseCentralizedController

    def build_world(self) -> World:
        obstacles = [
            CircularObstacle(center=np.array(o.center, dtype=float), radius=o.radius)
            for o in self.config.environment.obstacles
        ]
        target_cfg = self.config.environment.target_region
        if target_cfg is None:
            raise ValueError("target_region must be provided")
        target = CircularTargetRegion(center=np.array(target_cfg.center, dtype=float), radius=target_cfg.radius)
        return World(
            bounds=np.array(self.config.environment.bounds, dtype=float),
            obstacles=obstacles,
            target_region=target,
        )

    def _build_agents(self) -> list[Agent]:
        if len(self.config.agents.initial_positions) != self.config.agents.n_agents:
            raise ValueError("agents.initial_positions length must match agents.n_agents")
        return [
            Agent(
                agent_id=i,
                position=np.array(self.config.agents.initial_positions[i], dtype=float),
                radius=self.config.agents.radius,
                max_speed=self.config.agents.max_speed,
            )
            for i in range(self.config.agents.n_agents)
        ]

    def run(self) -> TrajectoryLog:
        np.random.seed(self.config.seed)
        dt = self.config.simulation.dt
        steps = self.config.simulation.steps

        world = self.build_world()
        agents = self._build_agents()
        log = TrajectoryLog()

        for k in range(steps):
            t = k * dt
            positions = np.array([a.position for a in agents])
            state = SystemState(t=t, positions=positions)
            log.add(t=t, positions=positions)

            if hasattr(self.controller, "compute_controls"):
                u = self.controller.compute_controls(state=state, world=world, t=t)
            else:
                u = np.array(
                    [
                        self.controller.compute_control(agent_id=i, state=state, world=world, t=t)
                        for i in range(state.n_agents)
                    ]
                )

            for i, agent in enumerate(agents):
                agent.step(u=np.array(u[i], dtype=float), dt=dt)
                agent.position = world.clip_to_bounds(agent.position)

        log.add(steps * dt, np.array([a.position for a in agents]))
        return log

"""Agent state and single-integrator update utilities."""

from __future__ import annotations

from dataclasses import dataclass

Vector2 = tuple[float, float]


def clamp_control_inf_norm(u: Vector2, u_max: float) -> Vector2:
    """Clamp each control component to [-u_max, u_max] (infinity norm bound)."""
    if u_max < 0:
        raise ValueError(f"u_max must be non-negative, got {u_max}.")
    ux, uy = u
    return (max(-u_max, min(u_max, ux)), max(-u_max, min(u_max, uy)))


@dataclass
class AgentState:
    """Minimal agent state for 2D single-integrator simulations."""

    agent_id: int
    position: Vector2
    velocity: Vector2

    def euler_step(self, u_k: Vector2, dt: float) -> None:
        """Apply one Euler integration step: x_{k+1} = x_k + dt * u_k."""
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        px, py = self.position
        ux, uy = u_k
        self.position = (px + dt * ux, py + dt * uy)
        # Stored explicitly for logging/debugging, even under single-integrator dynamics.
        self.velocity = u_k

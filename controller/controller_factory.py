"""Controller selection helper that preserves distributed mode as default."""

from __future__ import annotations

from config import ControllerConfig
from controller.base_controller import Controller
from controller.centralized_controller import CentralizedController
from controller.distributed_controller import DistributedController


def build_controller(config: ControllerConfig) -> Controller:
    """Build the controller selected by config.mode."""
    mode = config.mode.lower()
    if mode == "distributed":
        return DistributedController(config)
    if mode == "centralized":
        return CentralizedController(config)
    raise ValueError(f"Unsupported controller.mode: {config.mode!r}")

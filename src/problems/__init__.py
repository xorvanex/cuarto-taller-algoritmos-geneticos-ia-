"""Módulo de problemas de optimización."""

from src.problems.vrp import VRPModel, VRPInstance, VRPSolution, create_default_vrp_instance
from src.problems.pcb import PCBModel, PCBInstance, PCBSolution, create_default_pcb_instance
from src.problems.markowitz import (
    MarkowitzModel,
    PortfolioInstance,
    PortfolioSolution,
    Asset,
    create_default_portfolio_instance,
)

__all__ = [
    "VRPModel",
    "VRPInstance",
    "VRPSolution",
    "create_default_vrp_instance",
    "PCBModel",
    "PCBInstance",
    "PCBSolution",
    "create_default_pcb_instance",
    "MarkowitzModel",
    "PortfolioInstance",
    "PortfolioSolution",
    "Asset",
    "create_default_portfolio_instance",
]

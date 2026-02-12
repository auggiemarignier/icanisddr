"""Entry point for the icprem package."""

from .love import prem as PREM
from .model import average

PREM_IC_RHO = average("density", radius=1221.5)
PREM_IC_VP = average("vp", radius=1221.5)
PREM_IC_VS = average("vs", radius=1221.5)

__all__ = ["PREM", "PREM_IC_RHO", "PREM_IC_VP", "PREM_IC_VS"]

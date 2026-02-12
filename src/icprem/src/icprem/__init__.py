"""Entry point for the icprem package."""

from .love import prem as PREM
from .model import PREMInnerCore, average

_prem_ic_model = PREMInnerCore()
PREM_IC_RHO = average("density", radius=1221.5, model=_prem_ic_model)
PREM_IC_VP = average("vp", radius=1221.5, model=_prem_ic_model)
PREM_IC_VS = average("vs", radius=1221.5, model=_prem_ic_model)

__all__ = ["PREM", "PREM_IC_RHO", "PREM_IC_VP", "PREM_IC_VS"]

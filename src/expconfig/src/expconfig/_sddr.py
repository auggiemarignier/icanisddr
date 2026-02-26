"""
Lightweight local copies of SDDR config models so importing `expconfig` does not pull in `sddr` (which imports heavy backends like `jax`).

These match the expected YAML schema used by experiments and are intentionally minimal.
If downstream code requires the real `sddr` types, they can be converted explicitly by the caller.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class FlowModelConfig(BaseModel):
    """Lightweight local copy of the SDDR FlowModelConfig model, just the parameters we need for YAML config."""

    model_config = ConfigDict(frozen=True)


class RQSplineConfig(FlowModelConfig):
    """Lightweight local copy of the SDDR RQSplineConfig model, just the parameters we need for YAML config."""

    n_layers: int = 8
    n_bins: int = 8
    hidden_size: Sequence[int] = Field(default=(64, 64))
    spline_range: Sequence[float] = Field(default=(-10.0, 10.0))


class RealNVPConfig(FlowModelConfig):
    """Lightweight local copy of the SDDR RealNVPConfig model, just the parameters we need for YAML config."""

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4


class FlowConfig(BaseModel):
    """Lightweight local copy of the SDDR FlowConfig model, just the parameters we need for YAML config."""

    flow_type: str
    flow_model_config: FlowModelConfig
    standardize: bool = True
    learning_rate: float | None = None
    momentum: float | None = None


class TrainConfig(BaseModel):
    """Lightweight local copy of the SDDR TrainConfig model, just the parameters we need for YAML config."""

    batch_size: int
    epochs: int
    verbose: bool

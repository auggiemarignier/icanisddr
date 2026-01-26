"""Configuration SDDR IC anisotropy experiment."""

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field
from pydantic_yaml import to_yaml_str

from .geometry import GeometryConfig


class TrueBulkICConfig(BaseModel):
    """True bulk IC model parameters.

    These parameters define a single elastic tensor for the entire inner core.
    This is cylindrically anisotropic with symmetry axis along the Earth's rotation axis.
    The Love parameters are relative perturbations, not absolute values.
    The values for A, C, F are from Brett et al. (2024), while L and N are set to 0 (i.e. the absolute values are equal to the reference model).
    The angles eta1 and eta2 are in degrees.
    """

    A: float = 0.0143
    C: float = 0.0909
    F: float = -0.0858
    L: float = 0.0
    N: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return the parameters as a numpy array."""
        return np.array([self.A, self.C, self.F, self.L, self.N, self.eta1, self.eta2])


class GaussianComponentConfig(BaseModel):
    """Configuration for a Gaussian prior component."""

    type: str = Field("gaussian", frozen=True)
    mean: list[float]
    inv_covar: list[list[float]]
    indices: list[int]


class UniformComponentConfig(BaseModel):
    """Configuration for a Uniform prior component."""

    type: str = Field("uniform", frozen=True)
    lower_bounds: float | list[float]
    upper_bounds: float | list[float]
    indices: list[int]


class PriorsConfig(BaseModel):
    """Overall configuration for prior distributions."""

    components: list[dict]


class SamplingConfig(BaseModel):
    """Configuration for MCMC sampling."""

    nwalkers: int
    nsteps: int
    burn_in: int
    thin: int
    progress: bool
    parallel: bool


class TrainConfig(BaseModel):
    """Configuration for training the flow model."""

    batch_size: int = 256
    epochs: int = 1000
    verbose: bool = True


class RealNVPConfig(BaseModel):
    """Configuration for the RealNVP flow model."""

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4
    learning_rate: float = 0.001  # this is here rather than in TrainConfig because of how `harmonic` uses it
    momentum: float = 0.9
    standardize: bool = False
    temperature: float = 0.95


class HypothesisConfig(BaseModel):
    """Configuration for a single hypothesis test."""

    name: str
    indices: list[int]
    nu: list[float]


class DataConfig(BaseModel):
    """Configuration for synthetic data generation."""

    noise_level: float = Field(
        0.05,
        description="Noise level.",
    )


class Config(BaseModel):
    """Overall configuration for synthetic bulk IC experiment."""

    sampling: SamplingConfig
    priors: PriorsConfig
    training: TrainConfig
    realnvp: RealNVPConfig
    hypotheses: list[HypothesisConfig]
    truth: TrueBulkICConfig = Field(default_factory=TrueBulkICConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    geometry: GeometryConfig = Field(
        default_factory=GeometryConfig.earth_inner_core,
        description="Geometric configuration of regions.",
    )


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""

    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    """Dump configuration to YAML file."""

    with open(path, "w") as f:
        f.write(to_yaml_str(cfg))

"""Configuration for synthetic bulk IC experiment."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


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


class Config(BaseModel):
    """Overall configuration for synthetic bulk IC experiment."""

    sampling: SamplingConfig
    priors: PriorsConfig
    training: TrainConfig
    realnvp: RealNVPConfig
    hypotheses: list[HypothesisConfig]


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""

    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

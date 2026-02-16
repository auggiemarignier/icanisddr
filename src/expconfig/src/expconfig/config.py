"""Configuration SDDR IC anisotropy experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field
from sddr.sddr import FlowConfig, TrainConfig

from .geometry import GeometryConfig


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
    vectorise: bool
    parallel: bool


class HypothesisConfig(BaseModel):
    """Configuration for a single hypothesis test."""

    name: str
    indices: list[int]
    nu: list[float]


class ExpConfig(BaseModel):
    """Overall configuration for synthetic bulk IC experiment."""

    sampling: SamplingConfig
    priors: PriorsConfig
    training: TrainConfig
    flow: FlowConfig
    hypotheses: list[HypothesisConfig]
    geometry: GeometryConfig = Field(
        default_factory=GeometryConfig.earth_inner_core,
        description="Geometric configuration of regions.",
    )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load configuration from YAML file."""

        return load_config(path, model=cls)

    def dump(self: Self, path: str | Path) -> None:
        """Dump configuration to YAML file."""

        dump_config(self, path)


def _lists_to_tuples(obj):
    """Recursively convert lists in an object to tuples.

    This walks nested dict/list structures produced by YAML and ensures
    any list is converted to a tuple. Useful when downstream models or
    dataclasses expect tuples (for hashing/immutability) rather than lists.
    """
    if isinstance(obj, list):
        return tuple(_lists_to_tuples(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _lists_to_tuples(v) for k, v in obj.items()}
    return obj


def load_config[T](path: str | Path, model: type[T]) -> T:
    """Load configuration from YAML file."""

    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _lists_to_tuples(raw)
    return model(**raw)


def dump_config[T](cfg: T, path: str | Path) -> None:
    """Dump configuration to YAML file."""

    with open(path, "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)

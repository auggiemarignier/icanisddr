"""Testing basic experiment configuration loading and dumping."""

from pathlib import Path

import pytest
from sddr.sddr import FlowConfig, RQSplineConfig, TrainConfig

from expconfig import ExpConfig, dump_config, load_config
from expconfig.config import HypothesisConfig, PriorsConfig, SamplingConfig


@pytest.fixture
def valid_config() -> ExpConfig:
    """Return a valid configuration for testing.

    There is no default configuration, so we construct one here for testing purposes.
    """

    return ExpConfig(
        sampling=SamplingConfig(
            nwalkers=10,
            nsteps=100,
            burn_in=10,
            thin=2,
            progress=False,
            vectorise=True,
            parallel=False,
        ),
        priors=PriorsConfig(components=[{}]),
        training=TrainConfig(
            epochs=100,
            batch_size=32,
            verbose=True,
        ),
        flow=FlowConfig(flow_type="RQSpline", flow_model_config=RQSplineConfig()),
        hypotheses=[
            HypothesisConfig(
                name="H0",
                indices=[0],
                nu=[0.0],
            ),
        ],
    )


def test_load_dump(valid_config: ExpConfig, tmp_path: Path) -> None:
    """Test loading and dumping a configuration."""

    path = tmp_path / "test_config.yaml"
    dump_config(valid_config, path)
    loaded = load_config(path, ExpConfig)
    assert loaded == valid_config


def test_load_dump_classmethod(valid_config: ExpConfig, tmp_path: Path) -> None:
    """Test loading a configuration using the classmethod."""

    path = tmp_path / "test_config.yaml"
    valid_config.dump(path)
    loaded = ExpConfig.load(path)
    assert loaded == valid_config

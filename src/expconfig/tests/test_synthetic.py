"""Tests for synthetic data creation and noise models."""

from pathlib import Path

import numpy as np
import pytest
from sddr.sddr import RealNVPConfig, TrainConfig

from expconfig.config import HypothesisConfig, PriorsConfig, SamplingConfig
from expconfig.synthetic import (
    DataConfig,
    SynthConfig,
    TrueBulkICConfig,
    create_synthetic_data,
    gaussian_noise_data_max,
    noise_models,
)


class TestGaussianNoiseDataMax:
    """Tests for gaussian_noise_data_max noise model."""

    def test_returns_correct_shape(self):
        """Test that noise has the same shape as input data."""
        rng = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data)

        assert noise.shape == data.shape

    def test_raises_when_data_is_none(self):
        """Test that ValueError is raised when data is None."""
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="Data must be provided"):
            gaussian_noise_data_max(noise_level=0.1, rng=rng, data=None)

    def test_noise_scale_relative_to_data_max(self):
        """Test that noise scale is proportional to data max and noise_level."""
        rng = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_level = 0.1

        noise = gaussian_noise_data_max(noise_level=noise_level, rng=rng, data=data)

        # Expected scale is max(|data|) * noise_level = 4.0 * 0.1 = 0.4
        expected_scale = np.abs(data).max() * noise_level

        # Check that noise standard deviation is approximately equal to expected scale
        # Using a loose tolerance since it's random
        assert np.isclose(np.std(noise), expected_scale, rtol=0.3)

    def test_noise_centred_around_zero(self):
        """Test that noise has mean approximately zero."""
        rng = np.random.default_rng(42)
        data = np.ones(1000)
        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data)

        # With many samples, mean should be close to zero
        assert np.isclose(np.mean(noise), 0.0, atol=0.05)

    def test_larger_noise_level_produces_larger_noise(self):
        """Test that larger noise_level produces larger noise magnitude."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0, 4.0])

        noise_small = gaussian_noise_data_max(noise_level=0.05, rng=rng1, data=data)
        noise_large = gaussian_noise_data_max(noise_level=0.1, rng=rng2, data=data)

        assert np.std(noise_large) > np.std(noise_small)

    def test_conforms_to_protocol(self):
        """Test that function conforms to NoiseModel protocol."""
        # Check that the function can be used as a NoiseModel with extra kwargs
        rng = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0])

        # This should work without issues even with arbitrary extra kwargs
        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data, foo="bar")

        assert isinstance(noise, np.ndarray)

    def test_with_negative_data(self):
        """Test that noise model works with negative data values."""
        rng = np.random.default_rng(42)
        data = np.array([-4.0, -2.0, 1.0, 3.0])

        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data)

        # Scale should be based on max absolute value
        expected_scale = np.abs(data).max() * 0.1  # 0.4
        assert np.isclose(np.std(noise), expected_scale, rtol=0.3)

    def test_with_zero_data(self):
        """Test that noise model works when data is all zeros."""
        rng = np.random.default_rng(42)
        data = np.zeros(100)

        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data)

        # Should not raise error and should produce noise with scale equal to noise_level
        assert np.isclose(np.std(noise), 0.1, rtol=0.3)


class TestNoiseModelsRegistry:
    """Tests for the noise_models registry."""

    def test_gaussian_data_max_registered(self):
        """Test that gaussian_data_max is registered."""
        assert "gaussian_data_max" in noise_models
        assert noise_models["gaussian_data_max"] is gaussian_noise_data_max

    def test_registry_is_callable(self):
        """Test that registered models are callable."""
        rng = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0])

        noise_fn = noise_models["gaussian_data_max"]
        noise = noise_fn(noise_level=0.1, rng=rng, data=data)

        assert isinstance(noise, np.ndarray)


class TestCreateSyntheticData:
    """Tests for create_synthetic_data function."""

    # Dummy input matching DEFAULT_TRUTH shape (7 parameters for bulk IC model)
    DUMMY_TRUTH = np.zeros(7)

    @staticmethod
    def _dummy_calculator(truth: np.ndarray) -> np.ndarray:
        """Dummy calculator function for testing.

        Returns fixed synthetic data regardless of input, mimicking the behavior
        of a calculator function that produces travel time data.
        """
        return np.array([1.0, 2.0, 3.0, 4.0])

    def test_with_default_noise_model(self):
        """Test that create_synthetic_data works with default noise model."""
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.1,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        # Result should differ from clean data due to noise
        clean_data = self._dummy_calculator(self.DUMMY_TRUTH)
        assert not np.allclose(result, clean_data)

    def test_with_non_default_noise_model(self):
        """Test selecting a non-default noise model."""

        # First, register a simple test noise model
        def simple_noise(
            noise_level: float,
            rng: np.random.Generator,
            data: np.ndarray | None = None,
            **kwargs: object,
        ) -> np.ndarray:
            """Simple noise model that returns constant noise."""
            if data is None:
                raise ValueError("Data required")
            return np.ones_like(data) * noise_level

        # Temporarily add to registry
        noise_models["test_simple"] = simple_noise

        try:
            result = create_synthetic_data(
                calculator_fn=self._dummy_calculator,
                noise_level=0.5,
                noise_model="test_simple",
            )

            # Expected: clean data + constant 0.5 noise
            expected = self._dummy_calculator(self.DUMMY_TRUTH) + 0.5
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            # Clean up registry
            del noise_models["test_simple"]

    def test_forwarding_noise_kwargs(self):
        """Test that noise_kwargs are forwarded to the noise model."""

        # Register a noise model that uses kwargs
        def kwargs_noise(
            noise_level: float,
            rng: np.random.Generator,
            data: np.ndarray | None = None,
            multiplier: float = 1.0,
            **kwargs: object,
        ) -> np.ndarray:
            """Noise model that accepts additional kwargs."""
            if data is None:
                raise ValueError("Data required")
            return np.ones_like(data) * noise_level * multiplier

        noise_models["test_kwargs"] = kwargs_noise

        try:
            result = create_synthetic_data(
                calculator_fn=self._dummy_calculator,
                noise_level=0.5,
                noise_model="test_kwargs",
                noise_kwargs={"multiplier": 2.0},
            )

            # Expected: clean data + (0.5 * 2.0) noise
            expected = self._dummy_calculator(self.DUMMY_TRUTH) + 1.0
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            del noise_models["test_kwargs"]

    def test_noise_level_zero_returns_clean_data(self):
        """Test that noise_level=0.0 returns data without noise."""
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.0,
            noise_model="gaussian_data_max",
        )

        # Should return exactly the clean data, no noise added
        expected = self._dummy_calculator(self.DUMMY_TRUTH)
        np.testing.assert_array_equal(result, expected)

    def test_noise_level_zero_ignores_noise_model(self):
        """Test that noise_level=0.0 works even with invalid noise model."""
        # Should not raise error because noise_level is zero
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.0,
            noise_model="this_does_not_exist",
        )

        expected = self._dummy_calculator(self.DUMMY_TRUTH)
        np.testing.assert_array_equal(result, expected)

    def test_unknown_noise_model_raises_error(self):
        """Test that unknown noise model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise model 'nonexistent'"):
            create_synthetic_data(
                calculator_fn=self._dummy_calculator,
                noise_level=0.1,
                noise_model="nonexistent",
            )

    def test_unknown_noise_model_error_lists_available_models(self):
        """Test that error message lists available noise models."""
        try:
            create_synthetic_data(
                calculator_fn=self._dummy_calculator,
                noise_level=0.1,
                noise_model="invalid_model",
            )
        except ValueError as e:
            error_msg = str(e)
            assert "gaussian_data_max" in error_msg
            assert "Available noise models are:" in error_msg
            assert "Use noise_level=0.0 or noise_model='none'/'identity'" in error_msg

    def test_noise_model_none_returns_clean_data(self):
        """Test that noise_model='none' returns data without noise."""
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.1,  # Non-zero noise level
            noise_model="none",
        )

        # Should return clean data despite non-zero noise_level
        expected = self._dummy_calculator(self.DUMMY_TRUTH)
        np.testing.assert_array_equal(result, expected)

    def test_noise_model_identity_returns_clean_data(self):
        """Test that noise_model='identity' returns data without noise."""
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.1,  # Non-zero noise level
            noise_model="identity",
        )

        # Should return clean data despite non-zero noise_level
        expected = self._dummy_calculator(self.DUMMY_TRUTH)
        np.testing.assert_array_equal(result, expected)

    def test_noise_kwargs_none_works_correctly(self):
        """Test that noise_kwargs=None is handled correctly."""
        # Should work without error and use empty dict internally
        result = create_synthetic_data(
            calculator_fn=self._dummy_calculator,
            noise_level=0.1,
            noise_model="gaussian_data_max",
            noise_kwargs=None,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_with_zero_calculator_adds_noise(self):
        """Integration test: create_synthetic_data with calculator returning zeros.

        This test ensures that when a calculator returns all zeros, the noise model
        is still applied correctly through the full call path. This prevents regressions
        if the registry behavior or call path changes.
        """

        def zero_calculator(_truth: np.ndarray) -> np.ndarray:
            """Dummy calculator that returns all zeros."""
            return np.zeros(100)

        result = create_synthetic_data(
            calculator_fn=zero_calculator,
            noise_level=0.1,
            noise_model="gaussian_data_max",
        )

        # Result should not be all zeros - noise should be added
        assert not np.allclose(result, 0.0)
        # Result should have non-zero standard deviation
        assert np.std(result) > 0.0
        # The standard deviation should be approximately equal to noise_level
        # Using rtol=0.3 consistent with other std checks - accounts for randomness
        assert np.isclose(np.std(result), 0.1, rtol=0.3)


class TestSyntheticConfig:
    """Tests for the SynthConfig dataclass."""

    def test_loadable_dumpable(self, tmp_path: Path) -> None:
        """Test that SynthConfig can be dumped and loaded correctly."""
        config = SynthConfig(
            sampling=SamplingConfig(
                nwalkers=10,
                nsteps=100,
                burn_in=10,
                thin=1,
                progress=False,
                vectorise=False,
                parallel=False,
            ),
            priors=PriorsConfig(components=[{}]),
            training=TrainConfig(
                epochs=1,
                batch_size=1,
                verbose=False,
            ),
            realnvp=RealNVPConfig(
                n_scaled_layers=1,
                n_unscaled_layers=1,
            ),
            hypotheses=[
                HypothesisConfig(
                    name="test_hypothesis",
                    indices=[0],
                    nu=[0.0],
                )
            ],
            truth=TrueBulkICConfig(),
            data=DataConfig(),
        )

        config.dump(tmp_path / "synth_config.yaml")
        loaded_config = SynthConfig.load(tmp_path / "synth_config.yaml")
        assert isinstance(loaded_config, SynthConfig)
        assert loaded_config == config

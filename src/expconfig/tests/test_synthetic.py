"""Tests for synthetic data creation and noise models."""

import numpy as np
import pytest

from expconfig.synthetic import (
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
        # Check that the function can be used as a NoiseModel
        rng = np.random.default_rng(42)
        data = np.array([1.0, 2.0, 3.0])

        # This should work without issues
        noise = gaussian_noise_data_max(
            noise_level=0.1, rng=rng, data=data, extra_kwargs=None
        )

        assert isinstance(noise, np.ndarray)

    def test_with_negative_data(self):
        """Test that noise model works with negative data values."""
        rng = np.random.default_rng(42)
        data = np.array([-4.0, -2.0, 1.0, 3.0])

        noise = gaussian_noise_data_max(noise_level=0.1, rng=rng, data=data)

        # Scale should be based on max absolute value
        expected_scale = np.abs(data).max() * 0.1  # 0.4
        assert np.isclose(np.std(noise), expected_scale, rtol=0.3)


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

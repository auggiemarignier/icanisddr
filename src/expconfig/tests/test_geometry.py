"""Tests for geometry configuration."""

import numpy as np

from expconfig.geometry import (
    BallConfig,
    GeometryConfig,
    HemisphereConfig,
    SphericalShellConfig,
)


def test_ball_config():
    """Test BallConfig creation and conversion."""
    config = BallConfig(radius=1221.5, label="IC")

    assert config.type == "ball"
    assert config.radius == 1221.5
    assert config.label == "IC"

    # Convert to region
    region = config.to_region()
    assert region.radius == 1221.5


def test_spherical_shell_config():
    """Test SphericalShellConfig creation and conversion."""
    config = SphericalShellConfig(
        radius_inner=1221.5,
        radius_outer=3480.0,
        label="OC",
    )

    assert config.type == "shell"
    assert config.radius_inner == 1221.5
    assert config.radius_outer == 3480.0
    assert config.label == "OC"

    # Convert to region
    region = config.to_region()
    assert region.radius_inner == 1221.5
    assert region.radius_outer == 3480.0


def test_hemisphere_config():
    """Test HemisphereConfig creation and conversion."""
    config = HemisphereConfig(
        radius=1221.5,
        normal=[0.0, 0.0, 1.0],
        centre=[0.0, 0.0, 0.0],
        label="IC_north",
    )

    assert config.type == "hemisphere"
    assert config.radius == 1221.5
    assert config.normal == [0.0, 0.0, 1.0]
    assert config.centre == [0.0, 0.0, 0.0]

    # Convert to region
    region = config.to_region()
    assert region.radius == 1221.5
    np.testing.assert_array_equal(region.normal, [0.0, 0.0, 1.0])


def test_geometry_config():
    """Test GeometryConfig with multiple regions."""
    config = GeometryConfig(
        regions=[
            BallConfig(radius=1221.5, label="IC"),
            SphericalShellConfig(
                radius_inner=1221.5,
                radius_outer=3480.0,
                label="OC",
            ),
        ]
    )

    assert len(config.regions) == 2
    assert config.regions[0].label == "IC"
    assert config.regions[1].label == "OC"

    # Convert to composite geometry
    geometry = config.to_composite_geometry()
    assert len(geometry.regions) == 2
    assert geometry.labels == ["IC", "OC"]


def test_earth_inner_outer_core():
    """Test convenience method for standard Earth configuration."""
    config = GeometryConfig.earth_inner_outer_core()

    assert len(config.regions) == 2
    assert config.regions[0].type == "ball"
    assert config.regions[0].radius == 1221.5
    assert config.regions[1].type == "shell"
    assert config.regions[1].radius_outer == 3480.0


def test_hemispheric_ic():
    """Test convenience method for hemispheric IC."""
    config = GeometryConfig.hemispheric_ic()

    assert len(config.regions) == 2
    assert config.regions[0].type == "hemisphere"
    assert config.regions[1].type == "hemisphere"
    assert config.regions[0].label == "IC_north"
    assert config.regions[1].label == "IC_south"


def test_geometry_config_yaml_serialization():
    """Test that GeometryConfig can be serialized to YAML."""
    config = GeometryConfig.earth_inner_outer_core()

    # Convert to dict (Pydantic model_dump)
    config_dict = config.model_dump()

    assert "regions" in config_dict
    assert len(config_dict["regions"]) == 2

    # Reconstruct from dict
    config_reconstructed = GeometryConfig(**config_dict)
    assert len(config_reconstructed.regions) == 2

"""Ray tracing through 3D spherical geometries."""

from raytracer.intersection import calculate_ray_region_distances
from raytracer.ray import Ray
from raytracer.regions import (
    Ball,
    CompositeRegion,
    Hemisphere,
    Region,
    SphericalShell,
)

__all__ = [
    "Ray",
    "Region",
    "SphericalShell",
    "Hemisphere",
    "Ball",
    "CompositeRegion",
    "calculate_ray_region_distances",
]

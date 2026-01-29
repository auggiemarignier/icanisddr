"""Ray-geometry intersection and distance calculations."""

import numpy as np

from raytracer.ray import Ray
from raytracer.regions import CompositeRegion


def calculate_ray_region_distances(
    geometry: CompositeRegion,
    ray: Ray,
) -> np.ndarray:
    """Calculate the distance travelled by ray(s) through each region.

    Parameters
    ----------
    geometry : CompositeGeometry
        The composite geometry.
    ray : Ray
        The ray to trace.

    Returns
    -------
    distances : ndarray, shape (..., n_regions)
        Distance travelled through each region.
    """
    return geometry.ray_distances(ray.origin, ray.direction)

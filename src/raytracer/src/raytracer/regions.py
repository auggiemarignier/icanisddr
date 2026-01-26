"""Geometric regions that compose a sphere."""

from abc import ABC, abstractmethod

import numpy as np


class Region(ABC):
    """Base class for geometric regions within a sphere.

    Each region defines a bounded volume and can compute the distance
    travelled by a ray through it.
    """

    @abstractmethod
    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if point(s) are inside the region.

        Parameters
        ----------
        point : ndarray, shape (..., 3)
            Point(s) in Cartesian coordinates.

        Returns
        -------
        ndarray, shape (...)
            Boolean array indicating membership.
        """
        pass

    @abstractmethod
    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance(s) travelled by ray(s) through the region.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distance : ndarray, shape (...,)
            Distance travelled through region. Zero if no intersection.
        """
        pass


class SphericalShell(Region):
    """A spherical shell (region between two concentric spheres).

    Parameters
    ----------
    radius_inner : float
        Inner radius of the shell.
    radius_outer : float
        Outer radius of the shell.
    """

    def __init__(self, radius_inner: float, radius_outer: float):
        if radius_inner >= radius_outer:
            raise ValueError("radius_inner must be less than radius_outer")
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the shell."""
        r = np.linalg.norm(point, axis=-1)
        return (r >= self.radius_inner) & (r <= self.radius_outer)

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the shell.

        Solves for ray-sphere intersections and computes the distance
        travelled between outer and inner spheres.
        """
        origin = np.asarray(origin)
        direction = np.asarray(direction)

        # Intersections with outer sphere
        t_outer = _ray_sphere_intersection(origin, direction, self.radius_outer)

        # Intersections with inner sphere
        t_inner = _ray_sphere_intersection(origin, direction, self.radius_inner)

        # Determine which intersections are valid
        # We need entry point on outer sphere and exit point on inner sphere

        # For outer sphere: take the first valid (smallest positive) t
        t_outer_entry = np.where(
            (t_outer[:, 0] >= 0) & np.isfinite(t_outer[:, 0]),
            t_outer[:, 0],
            t_outer[:, 1],
        )

        # For inner sphere: take the first valid t >= t_outer_entry
        t_inner_exit = np.full_like(t_inner[:, 0], np.inf)
        for i in range(2):
            valid = (t_inner[:, i] >= t_outer_entry) & np.isfinite(t_inner[:, i])
            t_inner_exit = np.where(
                valid & (t_inner[:, i] < t_inner_exit),
                t_inner[:, i],
                t_inner_exit,
            )

        # Distance is t_inner_exit - t_outer_entry
        distances = np.where(
            np.isfinite(t_inner_exit) & np.isfinite(t_outer_entry),
            t_inner_exit - t_outer_entry,
            0.0,
        )

        return distances


class Ball(Region):
    """A solid sphere (ball).

    Parameters
    ----------
    radius : float
        Radius of the ball.
    """

    def __init__(self, radius: float):
        self.radius = radius

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the ball."""
        r = np.linalg.norm(point, axis=-1)
        return r <= self.radius

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the ball.

        Returns the distance from the first intersection to the second.
        """
        origin = np.asarray(origin)
        direction = np.asarray(direction)

        t_intersections = _ray_sphere_intersection(origin, direction, self.radius)

        # Take the two intersection points
        t1 = t_intersections[..., 0]
        t2 = t_intersections[..., 1]

        # Distance is the difference
        distances = np.where(
            np.isfinite(t1) & np.isfinite(t2),
            np.abs(t2 - t1),
            0.0,
        )

        return distances


class Hemisphere(Region):
    """A hemispherical region (half of a sphere along a plane).

    Parameters
    ----------
    radius : float
        Radius of the hemisphere.
    normal : ndarray, shape (3,)
        Normal vector of the dividing plane (points towards the positive hemisphere).
    centre : ndarray, shape (3,), optional
        Centre of the hemisphere. Default is origin.
    """

    def __init__(
        self,
        radius: float,
        normal: np.ndarray,
        centre: np.ndarray | None = None,
    ):
        self.radius = radius
        self.normal = np.asarray(normal) / np.linalg.norm(normal)
        self.centre = np.asarray(centre) if centre is not None else np.zeros(3)

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the hemisphere."""
        point = np.asarray(point)

        # Check if within the sphere
        r = np.linalg.norm(point - self.centre, axis=-1)
        in_sphere = r <= self.radius

        # Check if on the correct side of the plane
        relative_pos = point - self.centre
        side = np.sum(relative_pos * self.normal, axis=-1) >= 0

        return in_sphere & side

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the hemisphere.

        This is more complex as it involves:
        1. Ray-sphere intersections
        2. Ray-plane intersection
        3. Determining valid segments
        """
        origin = np.asarray(origin)
        direction = np.asarray(direction)

        # Translate to hemisphere centre
        origin_rel = origin - self.centre

        # Find ray-sphere intersections
        t_sphere = _ray_sphere_intersection(origin_rel, direction, self.radius)

        # Find ray-plane intersection (dividing plane at centre)
        denom = np.sum(direction * self.normal, axis=-1)

        # Handle ray parallel to plane
        t_plane = np.full_like(denom, np.inf)
        valid_plane = np.abs(denom) > 1e-10
        t_plane = np.where(
            valid_plane,
            -np.sum(origin_rel * self.normal, axis=-1) / denom,
            np.inf,
        )

        # Determine entry and exit points
        # Entry is the smaller positive t from sphere
        t_entry = np.minimum(t_sphere[..., 0], t_sphere[..., 1])
        t_entry = np.where(t_entry >= 0, t_entry, np.inf)

        # Exit is the larger positive t from sphere or plane intersection
        t_exit = np.maximum(t_sphere[..., 0], t_sphere[..., 1])

        # If ray passes through dividing plane within hemisphere, use that as exit
        in_hemisphere_at_plane = (t_plane >= t_entry) & (t_plane < t_exit)
        t_exit = np.where(in_hemisphere_at_plane, t_plane, t_exit)

        # Calculate distance
        distances = np.where(
            (np.isfinite(t_entry)) & (np.isfinite(t_exit)) & (t_exit > t_entry),
            t_exit - t_entry,
            0.0,
        )

        return distances


class CompositeGeometry:
    """A composition of multiple regions forming a complete geometry.

    Parameters
    ----------
    regions : list of Region
        List of regions, in order.
    labels : list of str, optional
        Labels for each region.
    """

    def __init__(self, regions: list[Region], labels: list[str] | None = None):
        self.regions = regions
        self.labels = (
            labels
            if labels is not None
            else [f"region_{i}" for i in range(len(regions))]
        )

        if len(self.labels) != len(self.regions):
            raise ValueError("Number of labels must match number of regions")

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distances through all regions.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distances : ndarray, shape (..., n_regions)
            Distance through each region.
        """
        origin = np.asarray(origin)
        direction = np.asarray(direction)

        # Ensure consistent broadcasting shape
        if origin.ndim == 1:
            n_rays = 1
            origin = origin[np.newaxis, :]
            direction = direction[np.newaxis, :]
        else:
            n_rays = origin.shape[0]

        distances = np.zeros((n_rays, len(self.regions)))

        for i, region in enumerate(self.regions):
            distances[:, i] = region.ray_distances(origin, direction)

        return distances


def _ray_sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Find intersections between a ray and a sphere at origin.

    Solves the equation ||origin + t * direction|| = radius for t.

    Parameters
    ----------
    origin : ndarray, shape (..., 3)
        Ray origin(s).
    direction : ndarray, shape (..., 3)
        Ray direction(s) (assumed normalised).
    radius : float
        Sphere radius.

    Returns
    -------
    t : ndarray, shape (..., 2)
        Two intersection parameter values. Invalid intersections are NaN.
    """
    origin = np.asarray(origin)
    direction = np.asarray(direction)

    # Ray: P(t) = origin + t * direction
    # Sphere: ||P||^2 = radius^2
    # Substituting: ||origin + t*direction||^2 = radius^2
    # Expanding: ||origin||^2 + 2*t*(origin·direction) + t^2*||direction||^2 = radius^2

    # Coefficients of quadratic equation: a*t^2 + b*t + c = 0
    a = np.sum(direction * direction, axis=-1)
    b = 2.0 * np.sum(origin * direction, axis=-1)
    c = np.sum(origin * origin, axis=-1) - radius**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    # Intersection parameters
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Mark invalid intersections
    t1 = np.where(discriminant >= 0, t1, np.nan)
    t2 = np.where(discriminant >= 0, t2, np.nan)

    return np.stack([t1, t2], axis=-1)

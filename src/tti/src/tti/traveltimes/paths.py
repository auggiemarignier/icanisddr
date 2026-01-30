"""Determining the paths of rays in TTI media.

In the IC we assume that rays travel along straight paths from ICB to ICB.

Given pierce points at the inner core boundary (ICB) and a mesh of the inner core, this module determines the distance travelled in each cell of the mesh by each ray.
"""

import numpy as np


def calculate_path_direction_vector(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
) -> np.ndarray:
    """
    Calculate the path direction unit vector from ic_in to ic_out.

    Parameters
    ----------
    ic_in : ndarray, shape (..., 3)
        Where the path enters the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (..., 3)
        Where the path exits the inner core (longitude (deg), latitude (deg), radius (km)).

    Returns
    -------
    n : ndarray, shape (..., 3)
        Path direction unit vector.
    """
    ic_in_xyz = _spherical_to_cartesian(ic_in[..., 0], ic_in[..., 1], ic_in[..., 2])
    ic_out_xyz = _spherical_to_cartesian(ic_out[..., 0], ic_out[..., 1], ic_out[..., 2])
    path_vector = ic_out_xyz - ic_in_xyz
    return path_vector / np.linalg.norm(path_vector, axis=-1, keepdims=True)


def _spherical_to_cartesian(
    lon: float | np.ndarray, lat: float | np.ndarray, r: float | np.ndarray
) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : float | np.ndarray
        Longitude in degrees.
    lat : float | np.ndarray
        Latitude in degrees.
    r : float | np.ndarray
        Radius.

    Returns
    -------
    ndarray, shape (..., 3)
        Cartesian coordinates (x, y, z).
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    r = np.asarray(r)

    colat = np.radians(90 - lat)
    lon = np.radians(lon)
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)

    return np.stack((x, y, z), axis=-1)

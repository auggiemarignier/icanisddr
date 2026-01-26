"""Configuration classes for geometric regions."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class RegionConfig(BaseModel):
    """Base configuration for a geometric region."""

    type: str
    label: str | None = None


class BallConfig(RegionConfig):
    """Configuration for a solid sphere (ball).

    Parameters
    ----------
    radius : float
        Radius of the ball (km).
    label : str, optional
        Label for this region.
    """

    type: Literal["ball"] = "ball"
    radius: float = Field(gt=0, description="Radius in km")

    def to_region(self):
        """Create a Ball region from this configuration.

        Returns
        -------
        Ball
            The configured Ball region.
        """
        from raytracer import Ball

        return Ball(radius=self.radius)


class SphericalShellConfig(RegionConfig):
    """Configuration for a spherical shell.

    Parameters
    ----------
    radius_inner : float
        Inner radius of the shell (km).
    radius_outer : float
        Outer radius of the shell (km).
    label : str, optional
        Label for this region.
    """

    type: Literal["shell"] = "shell"
    radius_inner: float = Field(gt=0, description="Inner radius in km")
    radius_outer: float = Field(gt=0, description="Outer radius in km")

    def to_region(self):
        """Create a SphericalShell region from this configuration.

        Returns
        -------
        SphericalShell
            The configured SphericalShell region.
        """
        from raytracer import SphericalShell

        return SphericalShell(
            radius_inner=self.radius_inner,
            radius_outer=self.radius_outer,
        )


class HemisphereConfig(RegionConfig):
    """Configuration for a hemisphere.

    Parameters
    ----------
    radius : float
        Radius of the hemisphere (km).
    normal : list of float
        Normal vector of the dividing plane [x, y, z]. Will be normalised.
    centre : list of float, optional
        Centre of the hemisphere [x, y, z] (km). Default is [0, 0, 0].
    label : str, optional
        Label for this region.
    """

    type: Literal["hemisphere"] = "hemisphere"
    radius: float = Field(gt=0, description="Radius in km")
    normal: list[float] = Field(
        min_length=3, max_length=3, description="Normal vector [x, y, z]"
    )
    centre: list[float] | None = Field(
        None, min_length=3, max_length=3, description="Centre [x, y, z] in km"
    )

    def to_region(self):
        """Create a Hemisphere region from this configuration.

        Returns
        -------
        Hemisphere
            The configured Hemisphere region.
        """
        from raytracer import Hemisphere

        centre = np.array(self.centre) if self.centre is not None else None
        return Hemisphere(
            radius=self.radius,
            normal=np.array(self.normal),
            centre=centre,
        )


class GeometryConfig(BaseModel):
    """Configuration for a composite geometry.

    This defines a complete geometric structure composed of multiple regions
    (e.g., inner core + outer core, or hemispherical shells).

    Parameters
    ----------
    regions : list of RegionConfig
        List of region configurations that make up the geometry.
    """

    regions: list[BallConfig | SphericalShellConfig | HemisphereConfig]

    def to_composite_geometry(self):
        """Create a CompositeGeometry from this configuration.

        Returns
        -------
        CompositeGeometry
            The configured composite geometry with all regions.
        """
        from raytracer import CompositeGeometry

        # Convert each region config to a region object
        region_objects = [region_config.to_region() for region_config in self.regions]

        # Extract labels
        labels = [
            region_config.label if region_config.label is not None else f"region_{i}"
            for i, region_config in enumerate(self.regions)
        ]

        return CompositeGeometry(regions=region_objects, labels=labels)

    @classmethod
    def earth_inner_outer_core(
        cls, ic_radius: float = 1221.5, oc_radius: float = 3480.0
    ):
        """Create a standard Earth configuration with inner and outer core.

        Parameters
        ----------
        ic_radius : float, default=1221.5
            Inner core radius (km).
        oc_radius : float, default=3480.0
            Outer core outer radius (km).

        Returns
        -------
        GeometryConfig
            Configuration with IC and OC regions.
        """
        return cls(
            regions=[
                BallConfig(radius=ic_radius, label="IC"),
                SphericalShellConfig(
                    radius_inner=ic_radius,
                    radius_outer=oc_radius,
                    label="OC",
                ),
            ]
        )

    @classmethod
    def hemispheric_ic(
        cls,
        ic_radius: float = 1221.5,
        normal: list[float] = [0.0, 0.0, 1.0],
    ):
        """Create a hemispherically divided inner core.

        Parameters
        ----------
        ic_radius : float, default=1221.5
            Inner core radius (km).
        normal : list of float, default=[0.0, 0.0, 1.0]
            Normal vector defining the hemisphere division.

        Returns
        -------
        GeometryConfig
            Configuration with two hemispheres.
        """
        return cls(
            regions=[
                HemisphereConfig(
                    radius=ic_radius,
                    normal=normal,
                    label="IC_north",
                ),
                HemisphereConfig(
                    radius=ic_radius,
                    normal=[-n for n in normal],
                    label="IC_south",
                ),
            ]
        )

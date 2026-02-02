"""
Create a PREM-like inner core model.

Strategy is to steal the inner core part of premlike.PREM.
"""

from typing import Literal

from premlike import OneDModel
from premlike.PREM import (
    bps,
    density_params,
    q_kappa_params,
    q_mu_params,
    r_earth,
    vp_params,
    vs_params,
)


class PREMInnerCore(OneDModel):
    """PREM-like inner core model."""

    def __init__(self) -> None:
        """Initialise PREM-like inner core model."""
        # Inner core is the first set of polynomial coefficients in PREM
        ic_bps = bps[:2]  # 0 to ic_radius

        # Using [None, :] to ensure each set of polynomial coefficients is mapped properly in OneDModel
        ic_vp_params = vp_params[0][None, :]
        ic_vs_params = vs_params[0][None, :]
        ic_density_params = density_params[0][None, :]
        # The following params are 0D arrays, so need to index differently
        ic_q_mu_params = q_mu_params[:1][None, :]
        ic_q_kappa_params = q_kappa_params[:1][None, :]

        super().__init__(
            breakpoints=ic_bps,
            density_params=ic_density_params,
            vp_params=ic_vp_params,
            vs_params=ic_vs_params,
            q_mu_params=ic_q_mu_params,
            q_kappa_params=ic_q_kappa_params,
            r_earth=r_earth,
        )


def average(
    property_name: Literal["density", "vp", "vs", "qm", "qk"],
    radius: float,
    model: OneDModel = PREMInnerCore(),
) -> float:
    """Compute the average value of a property over the inner core.

    Args:
        property_name: Name of the property to average. One of 'density', 'vp', 'vs', 'qm', 'qk'.
        radius: Radius up to which to average (in km).
        model: The OneDModel to use for the calculation. Defaults to PREMInnerCore.

    Returns:
        The average value of the property over the specified radius.
    """
    poly = getattr(model, f"{property_name}_poly")
    if poly is None:
        raise ValueError(f"Model does not have property '{property_name}_poly'")
    integrating_poly = poly.integrating_poly()
    average_value = integrating_poly(radius) / radius
    return average_value


def lame_lambda(mu: float, kappa: float) -> float:
    """Compute Lamé's first parameter (lambda) from shear modulus and bulk modulus.

    Args:
        mu: Shear modulus.
        kappa: Bulk modulus.
    Returns:
        Lamé's first parameter (lambda).
    """
    return kappa - (2.0 / 3.0) * mu

"""Average Love parameters over the inner core.

`premlike` gives polynomial coefficients and radial integration for density and seismic velocities, but not for elastic moduli.
So we compute average elastic moduli over the inner core here, and use these to compute average Love parameters.
The average elastic moduli are computed using a coarse radial sampling and a weighted average.
"""

from dataclasses import dataclass

import numpy as np

from .model import PREMInnerCore


def _compute_average_elastic_moduli() -> tuple[float, float]:
    """Compute average elastic moduli over the inner core."""
    premic = PREMInnerCore()
    R = premic.breakpoints[1]  # Inner core radius in km
    dR = 100.0  # km
    r = np.arange(0, R + dR, dR)
    if r[-1] > R:
        r[-1] = R
    elif r[-1] < R:
        r = np.append(r, R)

    average_weights = r[1:] - r[:-1]
    average_weights_total = np.sum(average_weights)
    bulk_modulus = (
        average_weights * premic.bulk_modulus(r)[1:]
    ).sum() / average_weights_total
    shear_modulus = (
        average_weights * premic.shear_modulus(r)[1:]
    ).sum() / average_weights_total
    lambda_ = bulk_modulus - (2.0 / 3.0) * shear_modulus
    mu = shear_modulus

    return float(lambda_), float(mu)


lambda_, mu = _compute_average_elastic_moduli()


@dataclass(frozen=True)
class PREMLoveParameters:
    """Dataclass to hold average PREM-like inner core Love parameters."""

    A: float = lambda_ + 2 * mu
    C: float = lambda_ + 2 * mu
    F: float = lambda_
    L: float = mu
    N: float = mu

    def as_array(self) -> np.ndarray:
        """Return the Love parameters as a numpy array."""
        return np.array([self.A, self.C, self.F, self.L, self.N])


prem = PREMLoveParameters()

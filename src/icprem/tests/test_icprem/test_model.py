"""Test grabbing PREM-like model from icprem package."""

import numpy as np
import pytest
from premlike import PREM

from icprem.model import PREMInnerCore, average


def test_compare_icprem_prem_like():
    """Test that PREMInnerCore from icprem matches premlike.PREM in the inner core."""
    icprem_model = PREMInnerCore()

    np.testing.assert_equal(icprem_model.breakpoints, PREM.breakpoints[:2])
    np.testing.assert_equal(icprem_model.vp_params, PREM.vp_params[0][None, :])
    np.testing.assert_equal(icprem_model.vs_params, PREM.vs_params[0][None, :])
    np.testing.assert_equal(
        icprem_model.density_params, PREM.density_params[0][None, :]
    )
    np.testing.assert_equal(icprem_model.q_mu_params, PREM.q_mu_params[:1][None, :])
    np.testing.assert_equal(
        icprem_model.q_kappa_params, PREM.q_kappa_params[:1][None, :]
    )

    # Sample radii in the inner core
    # breakpoints[1] is the inner core radius
    # Exclude the endpoint because OneDModel treats breakpoints differently if you're at the last breakpoint, which is the case at the ICB in PREMInnerCore
    radii = np.linspace(0, PREM.breakpoints[1], 10, endpoint=False)
    # Get velocities and density from premlike
    vp_prem = PREM.vp(radii)
    vs_prem = PREM.vs(radii)
    rho_prem = PREM.density(radii)

    # Get velocities and density from icprem
    vp_icprem = icprem_model.vp(radii)
    vs_icprem = icprem_model.vs(radii)
    rho_icprem = icprem_model.density(radii)

    # Assert that they are close
    np.testing.assert_allclose(vp_icprem, vp_prem, rtol=1e-6)
    np.testing.assert_allclose(vs_icprem, vs_prem, rtol=1e-6)
    np.testing.assert_allclose(rho_icprem, rho_prem, rtol=1e-6)


def test_average() -> None:
    """Test averaging over the whole inner core gives roughly the same as manually computing from the table in the paper."""
    # Replicate the table from PREM paper (r, rho, vp, vs, q_mu, q_kappa)
    # Density converted to kg/m3, velocities in km/s
    PREMInnerCoreTable = np.array(
        [
            [0, 13088.48, 11.26220, 3.66780, 85, 1328],
            [100.0, 13086.30, 11.26064, 3.66670, 85, 1328],
            [200.0, 13079.77, 11.25593, 3.66342, 85, 1328],
            [300.0, 13068.88, 11.24809, 3.65794, 85, 1328],
            [400.0, 13053.64, 11.23712, 3.65027, 85, 1328],
            [500.0, 13034.04, 11.22301, 3.64041, 85, 1328],
            [600.0, 13010.09, 11.20576, 3.62835, 85, 1328],
            [700.0, 12981.78, 11.18538, 3.61411, 85, 1328],
            [800.0, 12949.12, 11.16186, 3.59767, 85, 1328],
            [900.0, 12912.11, 11.13521, 3.57950, 85, 1328],
            [1000.0, 12870.73, 11.10542, 3.55823, 85, 1328],
            [1100.0, 12825.01, 11.07249, 3.53522, 85, 1328],
            [1200.0, 12774.93, 11.03643, 3.51002, 85, 1328],
            [1221.5, 12763.60, 11.02827, 3.50432, 85, 1328],
        ]
    )
    average_weights = PREMInnerCoreTable[1:, 0] - PREMInnerCoreTable[:-1, 0]
    weighted_table = PREMInnerCoreTable[:-1, 1:] * average_weights[:, None]
    expected_averages = np.sum(weighted_table, axis=0) / np.sum(average_weights)

    ic_prem = PREMInnerCore()
    for property_name, expected_average in zip(
        ["density", "vp", "vs", "qm", "qk"], expected_averages
    ):
        # Tolerance of 1% should be sufficient
        assert average(property_name, 1221.5, model=ic_prem) == pytest.approx(
            expected=expected_average, rel=1e-2
        ), f"Average {property_name} does not match expected value."

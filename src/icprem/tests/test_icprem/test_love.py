"""Test the icprem.love module."""

from icprem.love import prem


def test_prem_love_parameters():
    """Test that the PREM Love parameters have expected isotropic relationships."""
    assert prem.A == prem.C
    assert prem.L == prem.N
    assert prem.F == prem.A - 2 * prem.N

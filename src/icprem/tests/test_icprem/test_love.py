"""Test the icprem.love module."""

import numpy as np

from icprem.love import prem


def test_prem_love_parameters():
    """Test that the PREM Love parameters have expected isotropic relationships."""
    assert prem.A == prem.C
    assert prem.L == prem.N
    assert prem.F == prem.A - 2 * prem.N


def test_prem_as_array():
    """Test that as_array() returns correct shape and order."""
    array = prem.as_array()
    
    # Check shape
    assert array.shape == (5,), f"Expected shape (5,), got {array.shape}"
    
    # Check elements are in correct order [A, C, F, L, N]
    assert array[0] == prem.A, "First element should be A"
    assert array[1] == prem.C, "Second element should be C"
    assert array[2] == prem.F, "Third element should be F"
    assert array[3] == prem.L, "Fourth element should be L"
    assert array[4] == prem.N, "Fifth element should be N"
    
    # Alternative verification using numpy array equality
    expected = np.array([prem.A, prem.C, prem.F, prem.L, prem.N])
    np.testing.assert_array_equal(array, expected)

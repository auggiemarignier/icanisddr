"""Likelihood functions of MCMC sampling."""

from collections.abc import Callable

import numpy as np


def gaussian_likelihood_factory(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    observed_data: np.ndarray,
    covar: np.ndarray,
    example_model: None | np.ndarray = None,
    covar_is_inverse: bool = False,
) -> Callable[[np.ndarray], float]:
    """
    Create a Gaussian likelihood function.

    Parameters
    ----------
    forward_fn : Callable[[np.ndarray], np.ndarray]
        Forward model function that takes model parameters and returns predicted data.
    observed_data : ndarray, shape (n,)
        Observed data.
    covar : ndarray, shape (n, n)
        Covariance matrix of the observed data. Must be symmetric and positive semidefinite.
        Taken to be the inverse covariance matrix if `covar_is_inverse` is True.
    example_model : None | ndarray, optional
        Example model parameters to validate the forward function. If None (default), no validation is performed.
    covar_is_inverse : bool, optional
        If True, the provided covariance matrix is treated as the inverse covariance matrix (default is False).

    Returns
    -------
    likelihood_fn : Callable[[np.ndarray], float]
        Likelihood function that takes model parameters and returns the log-likelihood.

    Raises
    ------
    ValueError
        If the covariance matrix is not symmetric or not positive semidefinite.
    """
    _validate_data_vector(observed_data)
    _validate_covariance_matrix(covar, observed_data.size)
    if example_model is not None:
        _validate_forward_function(forward_fn, example_model, observed_data.size)

    inv_covar = covar if covar_is_inverse else np.linalg.inv(covar)

    def likelihood_fn(model_params: np.ndarray) -> float:
        predicted_data = forward_fn(model_params)
        residual = observed_data - predicted_data
        return -0.5 * residual.T @ inv_covar @ residual

    return likelihood_fn


def _validate_data_vector(data: np.ndarray) -> None:
    """
    Validate that the data vector is one-dimensional.

    Parameters
    ----------
    data : ndarray
        Data vector to validate.

    Raises
    ------
    ValueError
        If the data vector is not one-dimensional.
    """
    if data.ndim != 1:
        raise ValueError("Data vector must be one-dimensional.")


def _validate_covariance_matrix(covar: np.ndarray, N: int) -> None:
    """
    Validate that the covariance matrix is symmetric and positive semidefinite.

    Parameters
    ----------
    covar : ndarray, shape (n, n)
        Covariance matrix to validate.
    N : int
        Expected size of the covariance matrix.

    Raises
    ------
    ValueError
        If the covariance matrix
            - has incorrect shape;
            - is not symmetric; or
            - is not positive semidefinite.
    """
    if covar.shape != (N, N):
        raise ValueError(f"Covariance matrix must be of shape ({N}, {N}).")

    if not np.allclose(covar, covar.T):
        raise ValueError("Covariance matrix must be symmetric.")

    try:
        np.linalg.cholesky(covar)
        # If Cholesky decomposition succeeds, the matrix is positive definite
        # It is very unlikely for a realistic covariance matrix to have zero eigenvalues (positive semidefinite) so this check is sufficient
    except np.linalg.LinAlgError as e:
        raise ValueError("Covariance matrix must be positive semidefinite.") from e


def _validate_forward_function(
    forward_fn: Callable[[np.ndarray], np.ndarray], example_model: np.ndarray, N: int
) -> None:
    """
    Validate that the forward function returns data of the correct shape.

    Parameters
    ----------
    forward_fn : Callable[[np.ndarray], np.ndarray]
        Forward model function to validate.
    example_model : ndarray
        Example model parameters to test the forward function.
    N : int
        Expected size of the output data vector.

    Raises
    ------
    ValueError
        If the forward function does not return data of the correct shape.
    """
    predicted_data = forward_fn(example_model)
    if predicted_data.shape != (N,):
        raise ValueError(f"Forward function must return prediction of shape ({N},).")

"""Functions for calculating the Savage-Dickey density ratio."""

from dataclasses import asdict, dataclass

import harmonic as hm
import numpy as np
from harmonic.model import RealNVPModel

from .posterior import marginalise_samples
from .prior import CompoundPrior


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training the flow model.

    Parameters to be passed to hm.model.FlowModel.fit().
    """

    batch_size: int = 64
    epochs: int = 10
    verbose: bool = True


@dataclass(frozen=True)
class RealNVPConfig:
    """Configuration of the RealNVP model.

    Just the list of parameters taken by hm.model.RealNVPModel.
    """

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4
    learning_rate: float = 1e-3
    momentum: float = 0.9
    standardize: bool = False
    temperature: float = 0.8


def fit_marginalised_posterior(
    samples: np.ndarray,
    marginal_indices: list[int],
    model_config: RealNVPConfig | None = None,
    train_config: TrainConfig | None = None,
) -> hm.model.FlowModel:
    """Fit a flow model to the marginalised posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters.
    marginal_indices : list of int
        Indices of the parameters to keep after marginalisation.
    model_config : RealNVPConfig, optional
        Configuration for the RealNVP model. If None, default configuration is used.
    train_config : TrainConfig, optional
        Configuration for training the flow model. If None, default configuration is used.

    Returns
    -------
    model : FlowModel
        Fitted flow model to the marginalised posterior.
    """
    if model_config is None:
        model_config = RealNVPConfig()
    if train_config is None:
        train_config = TrainConfig()

    marginalised_samples = marginalise_samples(samples, marginal_indices)
    model = RealNVPModel(ndim_in=len(marginal_indices), **asdict(model_config))
    model.fit(X=marginalised_samples, **asdict(train_config))
    return model


def sddr(
    marginalised_posterior: hm.model.FlowModel,
    marginalised_prior: CompoundPrior,
    nu: np.ndarray,
) -> float:
    """Calculate the Savage-Dickey density ratio (SDDR) for given marginalised posterior and prior.

    Parameters
    ----------
    marginalised_posterior : FlowModel
        Fitted flow model to the marginalised posterior.
    marginalised_prior : CompoundPrior
        Marginalised prior distribution.
    nu : ndarray, shape (k,)
        Point at which to evaluate the SDDR, where k is the number of marginalised parameters.

    Returns
    -------
    sddr : float
        Log SDDR value at the given point.
    """
    prior_log_prob = marginalised_prior(nu)
    posterior_log_prob = marginalised_posterior.predict(nu)
    return float(posterior_log_prob - prior_log_prob)

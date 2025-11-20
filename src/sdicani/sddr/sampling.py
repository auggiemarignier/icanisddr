"""Sampling using emcee."""

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from emcee import EnsembleSampler

from sdicani.util import DummyPool


@dataclass(frozen=True)
class MCMCConfig:
    """Configuration for MCMC sampling.

    Parameters
    ----------
    nwalkers : int
        Number of MCMC walkers.
    nsteps : int
        Number of MCMC steps.
    burn_in : int
        Number of burn-in steps to discard.
    initial_from_prior : bool
        Whether to initialize walkers from the prior distribution.
    parallel : bool
        Whether to use parallel processing.
    progress : bool
        Whether to display a progress bar.
    thin : int
        Thinning factor for the MCMC samples.
    """

    nwalkers: int = 50
    nsteps: int = 1000
    burn_in: int = 200
    initial_from_prior: bool = False  # Haven't implemented this yet
    parallel: bool = False
    progress: bool = True
    thin: int = 1


def mcmc(
    ndim: int,
    posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    config: MCMCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MCMC sampling for the synthetic bulk IC experiment.

    Returns
    -------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters, after burn-in and thinning.
    lnprob : ndarray, shape (num_samples,)
        Log-probabilities of the MCMC samples, after burn-in and thinning.
    """
    if config is None:
        config = MCMCConfig()

    if config.initial_from_prior:
        raise NotImplementedError("Initialisation from prior not yet implemented.")
    else:
        initial_pos = rng.normal(0, 1, size=(config.nwalkers, ndim))

    p = Pool if config.parallel else DummyPool

    with p() as pool:
        sampler = EnsembleSampler(config.nwalkers, ndim, posterior, pool=pool)
        sampler.run_mcmc(initial_pos, config.nsteps, progress=config.progress)
        samples = np.ascontiguousarray(
            sampler.get_chain(flat=True)[config.burn_in :: config.thin, :]
        )
        lnprob = np.ascontiguousarray(
            sampler.get_log_prob(flat=True)[config.burn_in :: config.thin]
        )

    return samples, lnprob

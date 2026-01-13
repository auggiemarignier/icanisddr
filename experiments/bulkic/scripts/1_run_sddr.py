"""Synthetic bulk IC experiment entry point."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from bulkic.config import Config, load_config
from harmonic.model import RealNVPModel
from harmonic.sddr import sddr as hmsddr
from sampling.priors import CompoundPrior, PriorFunction
from sddr.marginalisation import marginalise_prior, marginalise_samples
from sddr.sddr import (
    RealNVPConfig,
    TrainConfig,
    fit_marginalised_posterior,
    sddr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def configure_posterior_fit(
    cfg: dict[str, Any],
) -> tuple[TrainConfig, RealNVPConfig]:
    """Configure the posterior fitting parameters.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration dictionary for the prior.

    Returns
    -------
    train_cfg : TrainConfig
        Training configuration for the flow model.
    realnvp_cfg : RealNVPConfig
        RealNVP configuration for the flow model.
    """

    return TrainConfig(**cfg.get("training", {})), RealNVPConfig(
        **cfg.get("realnvp", {})
    )


def run_sddr_experiment(
    samples: np.ndarray,
    indices: list[int],
    prior: PriorFunction,
    nu: np.ndarray,
    cfg: Config,
) -> None:
    """Run the synthetic bulk IC SDDR experiment."""

    logger.info("Configuring posterior fitting parameters")
    train_cfg, realnvp_cfg = configure_posterior_fit(cfg.model_dump())
    logger.info("Training flow model on marginalised samples")
    marg_posterior = fit_marginalised_posterior(
        samples, indices, realnvp_cfg, train_cfg
    )

    logger.info("Marginalising the prior")
    marg_prior = marginalise_prior(prior, indices)

    logger.info("Calculating Savage-Dickey density ratio at point nu=0")
    sddr_h1 = sddr(marg_posterior, marg_prior, nu)
    logger.info(f"logSDDR for hypothesis 1 (vertical symmetry axis): {sddr_h1:.4f}")

    if sddr_h1 > 0:
        logger.info("Evidence in favour of nested model")
    else:
        logger.info("Evidence against nested model")

    logger.info("Experiment complete")

    logger.info("Calculating harmonic SDDR for comparison")
    flow_model = RealNVPModel(ndim_in=marg_prior.n, standardize=True, temperature=1.0)
    harm_sddr = hmsddr(flow_model, marginalise_samples(samples, indices))
    flow_log_bf, flow_log_bf_std = harm_sddr.log_bayes_factor(
        log_prior=marg_prior(nu),
        value=nu,
        nbootstraps=10,
        bootstrap_proportion=0.5,
        bootstrap=True,
        epochs=10,
    )
    logger.info(
        f"Harmonic logSDDR for hypothesis 1 (vertical symmetry axis): {flow_log_bf:.4f} ± {flow_log_bf_std:.4f}"
    )


OUTPUTS_PATH = Path(__file__).parent.parent / "outputs"


def get_results_dir(n: int = 0) -> Path:
    """Get the most recent results directory.

    Parameters
    ----------
    n : int, optional
        Index of the results directory to retrieve.
        0 for the most recent, 1 for the second most recent, etc.
        Default is 0.

    Returns
    -------
    Path
        Path to the results directory.
    """
    output_dirs = sorted(
        OUTPUTS_PATH.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return output_dirs[n]


def main() -> None:
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    results_dir = get_results_dir(n)
    SAMPLES_PATH = results_dir / "samples.pkl"
    cfg = load_config(results_dir / "config.yaml")

    logger.info("Loading posterior samples from disk")
    samples = pickle.load(open(SAMPLES_PATH, "rb"))

    logger.info("Setting up prior")
    prior = CompoundPrior.from_dict(cfg.priors.model_dump())

    for hypothesis in cfg.hypotheses:
        logger.info("========================================")
        logger.info(f"Running SDDR for hypothesis: {hypothesis.name}")
        nu = np.array(hypothesis.nu)
        run_sddr_experiment(
            samples=samples,
            indices=hypothesis.indices,
            nu=nu,
            prior=prior,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()

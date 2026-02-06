"""Synthetic bulk IC experiment entry point."""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np

from expconfig import dump_config, load_config
from expconfig.config import PriorsConfig
from expconfig.synthetic import (
    create_paths,
    create_synthetic_data,
)
from icprem import PREM
from sampling.likelihood import GaussianLikelihood
from sampling.priors import CompoundPrior
from sampling.sampling import MCMCConfig, mcmc
from tti.traveltimes import TravelTimeCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CFG_FILE = Path(__file__).parent.parent / "config.yaml"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


def _setup_synthetic_data(
    truth: np.ndarray, noise_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Creating synthetic data...")
    ic_in, ic_out = create_paths(source_spacing=20.0)
    synthetic_data = create_synthetic_data(
        TravelTimeCalculator(
            ic_in, ic_out, reference_love=PREM.as_array(), nested=False, shear=True
        ),
        truth,
        noise_level,
    )[0]
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")
    return ic_in, ic_out, synthetic_data


def _setup_likelihood(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
    synthetic_data: np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    ttc = TravelTimeCalculator(
        ic_in, ic_out, reference_love=PREM.as_array(), nested=True, shear=True
    )
    inv_covar = np.array([1 / synthetic_data.std() ** 2])
    likelihood = GaussianLikelihood(ttc, synthetic_data, inv_covar)
    return likelihood


def _setup_prior(prior_cfg: PriorsConfig) -> CompoundPrior:
    logger.info("Setting up prior distributions...")
    prior = CompoundPrior.from_dict(prior_cfg.model_dump())
    return prior


def dump_results(samples: np.ndarray, lnprob: np.ndarray, output_dir: Path) -> None:
    """Dump the results to disk.

    Dumped files are
    - samples_full.pkl: the full (after burn and thin) MCMC samples
    - lnprob_full.pkl: the log-probabilities of the full (after burn and thin) MCMC samples

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples.
    lnprob : np.ndarray
        Log-probabilities of the samples.
    output_dir : Path
        Directory to save the results.
    """
    with open(output_dir / "samples_full.pkl", "wb") as f:
        pickle.dump(samples, f)
    with open(output_dir / "lnprob_full.pkl", "wb") as f:
        pickle.dump(lnprob, f)

    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")
    cfg = load_config(CFG_FILE)

    rng = np.random.default_rng(42)
    ic_in, ic_out, synthetic_data = _setup_synthetic_data(
        cfg.truth.as_array(), cfg.data.noise_level
    )
    likelihood = _setup_likelihood(ic_in, ic_out, synthetic_data)
    prior = _setup_prior(cfg.priors)

    logger.info("Running MCMC sampling")
    samples, lnprob = mcmc(
        prior.n, likelihood, prior, rng, MCMCConfig(**cfg.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")

    logger.info("Saving samples to disk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dump_results(samples, lnprob, OUTPUT_DIR)
    dump_config(cfg, OUTPUT_DIR / "config.yaml")


if __name__ == "__main__":
    main()

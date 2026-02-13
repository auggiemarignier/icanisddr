"""Real Data bulk IC experiment entry point."""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from expconfig import dump_config, load_config
from icprem import PREM_IC_RHO, PREM_IC_VP
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
DATA_FILE = Path(__file__).parent.parent / "data" / "brett2024_ic_traveltimes.parquet"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)

# Hierarchical noise levels obtained by Brett et al., 2022
noise_levels: dict[str, float] = {
    "ab": 0.95,
    "bc": 0.63,
    "cd": 0.29,
    "df": 0.95,
}


def _setup_data(
    data_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading real data...")
    df = pd.read_parquet(data_file)
    ic_in = np.stack(df.in_location.values)
    ic_out = np.stack(df.out_location.values)
    dt_over_t = (df.delta_t / df.inner_core_travel_time).values
    sigma = df["reference_phase"].map(noise_levels).values
    logger.info(f"Real data shape: {dt_over_t.shape}")
    return ic_in, ic_out, dt_over_t, sigma


def _setup_likelihood(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
    dt_over_t: np.ndarray,
    sigma: np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function")
    normalisation = -1 / (2 * PREM_IC_RHO * (PREM_IC_VP * 1e3) ** 2)
    ttc = TravelTimeCalculator(
        ic_in, ic_out, normalisation=normalisation, nested=True, shear=True, N=False
    )
    inv_covar = 1 / sigma**2
    likelihood = GaussianLikelihood(ttc, dt_over_t, inv_covar)
    return likelihood


def _setup_prior(prior_cfg: dict[str, Any]) -> CompoundPrior:
    logger.info("Setting up prior distributions")
    prior = CompoundPrior.from_dict(prior_cfg)
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
    """Main function for real data bulk IC experiment."""
    logger.info("Starting real data bulk IC experiment")
    cfg = load_config(CFG_FILE)

    rng = np.random.default_rng(42)
    ic_in, ic_out, dt_over_t, sigma = _setup_data(DATA_FILE)
    prior = _setup_prior(cfg.priors.model_dump())
    likelihood = _setup_likelihood(ic_in, ic_out, dt_over_t, sigma)

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

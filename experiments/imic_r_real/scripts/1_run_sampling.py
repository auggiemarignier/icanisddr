"""Synthetic IMIC experiment entry point."""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from expconfig.config import ExpConfig, PriorsConfig
from expconfig.geometry import IC_RADIUS
from raytracer import BallInShell
from sampling.likelihood import GaussianLikelihood
from sampling.likelihood._base import ForwardBase
from sampling.priors import CompoundPrior
from sampling.sampling import MCMCConfig, ptmcmc
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes._types import seven_arrays
from tti.traveltimes.parametrisations import BaseParametriser
from tti.traveltimes.paths import calculate_path_direction_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CFG_FILE = Path(__file__).parent.parent / "config.yaml"
DATA_FILE = Path(__file__).parent.parent / "data" / "brett2024_ic_traveltimes.parquet"

# Hierarchical noise levels obtained by Brett et al., 2022
noise_levels: dict[str, float] = {
    "ab": 0.95,
    "bc": 0.63,
    "cd": 0.29,
    "df": 0.95,
}

TO_RADIANS = np.pi / 180.0


class Parametriser(BaseParametriser):
    """Nested parametrisation comparing corresponding Love parameters in 2 layers.

    A1, A2-A1, C1, C2-C1 etc.
    """

    n_model_params_per_segment = 2

    def __init__(self) -> None:
        self.transformation = np.eye(14, 10)

        self.transformation[1, 0] = 1.0  # A2
        self.transformation[3, 2] = 1.0  # C2
        self.transformation[5, 4] = 1.0  # F2
        self.transformation[6:10, :] = 0  # L1, L2, N1, N2
        self.transformation[10, 6] = TO_RADIANS  # eta11
        self.transformation[11, 6:8] = TO_RADIANS  # eta12
        self.transformation[12, 8] = TO_RADIANS  # eta21
        self.transformation[13, 8:10] = TO_RADIANS  # eta22

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        """Transform m to love parameters.

        m will be of shape (batch, n_segmentss*n_model_params_per_segment)
        """
        lv = np.matvec(self.transformation, m)
        lv = np.atleast_2d(lv)
        A, C, F, L, N, eta1, eta2 = lv.reshape(7, -1, self.n_model_params_per_segment)
        return A, C, F, L, N, eta1, eta2

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        """Apply the jacobian of this transformation."""
        raise NotImplementedError("Not using gradients at the moment")


def _setup_data(
    data_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading real data...")
    df = pd.read_parquet(data_file)
    ic_in = np.stack(df.in_location.values)
    ic_out = np.stack(df.out_location.values)
    dt_over_t = (df.delta_t / df.inner_core_travel_time).values
    #  The noise levels for each reference phase are given in seconds, so we need to convert them to fractional traveltime perturbations by dividing by the inner core travel time.
    # In principle this gives a different sigma for each observation.
    sigma = (
        df["reference_phase"].map(noise_levels) / df["inner_core_travel_time"]
    ).values
    logger.info(f"Real data shape: {dt_over_t.shape}")
    return ic_in, ic_out, dt_over_t, sigma


def lonlatrad_to_xyz(lonlatrad: np.ndarray) -> np.ndarray:
    """Convert (lon, lat, radius) to Cartesian (x, y, z) coordinates.

    Parameters
    ----------
    lonlatrad : np.ndarray, shape (..., 3)
        Array of longitude (degrees), latitude (degrees), and radius (km).

    Returns
    -------
    xyz : np.ndarray, shape (..., 3)
        Array of Cartesian coordinates in km.
    """
    lon = np.radians(lonlatrad[..., 0])
    lat = np.radians(lonlatrad[..., 1])
    r = lonlatrad[..., 2]

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def determine_weights(
    region, ic_in: np.ndarray, path_directions: np.ndarray
) -> np.ndarray:
    """Determine weights for each path based on the distance travelled in each region.

    Parameters
    ----------
    region : CompositeRegion
        The composite region defining the geometry and properties of the inner core.
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    path_directions : ndarray, shape (num_paths, 3)
        Direction vectors for each path.

    Returns
    -------
    weights : ndarray, shape (1, num_segments, num_paths)
        Fractional distance of each path in each segment.  Additional axis for broadcasting with travel time calculator.
    """
    segment_distances = region.ray_distances_per_region(
        lonlatrad_to_xyz(ic_in), path_directions
    )
    total_distances = segment_distances.sum(axis=1)
    weights = segment_distances / total_distances[:, None]
    return weights.T[None, ...]


def forward(ttc: TravelTimeCalculator, params: np.ndarray) -> np.ndarray:
    """Forward model for synthetic IMIC experiment.

    Parameters
    ----------
    ttc : TravelTimeCalculator
        The travel time calculator.
    params : np.ndarray
        Model parameters. Shape (n_samples, n_parameters).
        Parameters are sorted as [A_1, A_2, C_1, C_2, F_1, F_2, eta1_1, eta2_1, eta1_2, eta2_2, r],
        where A, C, F, eta1, eta2 are the usual TTI parameters for each region (IMIC first, then OIC), and r is the radius of IMIC.

    Returns
    -------
    np.ndarray
        Predicted travel times. Shape (n_samples, n_rays).
    """
    tti_params = params[:, :-1]
    imic_radius = params[:, -1]

    if imic_radius < 0 or imic_radius > IC_RADIUS:
        # Return large travel times for unphysical radii to effectively zero out their likelihood
        # This case would normally be caught by the prior, but this allows us to skip the actual travel time calculation for these unphysical cases.
        # The logprior will be -inf anyway.
        # This also avoids having to try/catch errors when determining weights for unphysical radii.
        return np.full((params.shape[0], ttc.ic_in.shape[0]), 1e6)

    weights = determine_weights(
        BallInShell(imic_radius, IC_RADIUS), ttc.ic_in, ttc.path_directions
    )  # shape (1, n_cells, npaths)
    ttc.update_weights(weights)
    return ttc(tti_params)


OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


class Forward(ForwardBase[TravelTimeCalculator]):
    """Forward wrapper implementing the new `ForwardBase` API.

    The internal state is a `TravelTimeCalculator` instance. The forward
    callable expects model parameter arrays where the final column is the
    IMIC radius.
    """

    state: TravelTimeCalculator

    def __init__(self, state: TravelTimeCalculator) -> None:
        self.state = state

    @classmethod
    def from_state(cls, state: TravelTimeCalculator) -> Self:
        """Initialise from state."""
        return cls(state)

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Call on a batch of model parameters.

        Determines weights, updates them in the calculator, then calls the calculator.
        """
        model_params = np.atleast_2d(model_params)
        tti_params = model_params[:, :-1]
        imic_radii = model_params[:, -1]
        n_cells = self.state.parametriser.n_model_params_per_segment

        n_samples = model_params.shape[0]
        npaths = self.state.ic_in.shape[0]

        invalid = (imic_radii < 0) | (imic_radii > IC_RADIUS)
        batched_weights = np.zeros((n_samples, n_cells, npaths))
        for i, (is_sample_invalid, imic_r) in enumerate(zip(invalid, imic_radii)):
            if not is_sample_invalid:
                batched_weights[i] = determine_weights(
                    BallInShell(imic_r, IC_RADIUS),
                    self.state.ic_in,
                    self.state.path_directions,
                )

        self.state.update_weights(batched_weights)
        out = self.state(tti_params)
        out[invalid, :] = 1e6
        return out


def _setup_likelihood(
    ttc: TravelTimeCalculator,
    dt_over_t: np.ndarray,
    sigma: float | np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    inv_covar = 1 / sigma**2
    forward_inst = Forward.from_state(ttc)
    likelihood = GaussianLikelihood(forward_inst, dt_over_t, inv_covar)
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
    """Main function for real data imic_r experiment."""
    logger.info("Starting real data IMIC_r experiment")
    cfg = ExpConfig.load(CFG_FILE)

    ic_in, ic_out, dt_over_t, sigma = _setup_data(DATA_FILE)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    region = cfg.geometry.to_composite_region()
    initial_weights = determine_weights(region, ic_in, path_directions)
    ttc = TravelTimeCalculator(
        ic_in=ic_in,
        ic_out=ic_out,
        normalisation=-0.5,
        weights=initial_weights,
        parametriser=Parametriser(),
    )
    likelihood = _setup_likelihood(ttc, dt_over_t, sigma)
    prior = _setup_prior(cfg.priors)

    logger.info("Running MCMC sampling")
    rng = np.random.default_rng(42)
    samples, lnprob = ptmcmc(
        prior.n, likelihood, prior, rng, MCMCConfig(**cfg.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")

    logger.info("Saving samples to disk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dump_results(samples, lnprob, OUTPUT_DIR)
    cfg.dump(OUTPUT_DIR / "config.yaml")


if __name__ == "__main__":
    main()

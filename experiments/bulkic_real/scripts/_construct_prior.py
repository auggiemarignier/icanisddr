"""Helper script to construct the prior on the Love parameters."""

from dataclasses import replace

import numpy as np

from icprem import PREM

PREM = replace(
    PREM, A=PREM.A * 1e9, C=PREM.C * 1e9, F=PREM.F * 1e9, L=PREM.L * 1e9, N=PREM.N * 1e9
)  # Convert from GPa to Pa

SCALE = 0.2  # percentage scale from PREM values to set the width of the Gaussian prior
LF_CORR = -0.75  # correlation coefficient between L and F in the Gaussian prior


def construct_prior_covariance() -> np.ndarray:
    """Construct the covariance matrix for the Gaussian prior on the Love parameters."""
    variances = (SCALE * PREM.as_array()[:-1]) ** 2  # [:-1] to exclude N
    cov = np.diag(variances)
    cov[2, 3] = LF_CORR * np.sqrt(variances[2] * variances[3])  # F-L correlation
    cov[3, 2] = cov[2, 3]  # symmetric covariance matrix
    return cov


if __name__ == "__main__":
    inv_cov = np.linalg.inv(construct_prior_covariance())
    print("Inverse covariance matrix for the Gaussian prior on Love parameters:")
    # Print each row as a YAML-friendly list (copy-paste into config.yaml)
    for row in inv_cov:
        formatted = [f"{x:.8e}" for x in row]
        print(f"        - [{', '.join(formatted)}]")

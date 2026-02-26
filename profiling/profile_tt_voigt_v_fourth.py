"""Profile traveltime calculation using Voigt vs 4th-order tensor methods."""

import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from tti.elastic.voigt_mapping import elastic_tensor_to_voigt
from tti.traveltimes.traveltimes import (
    calculate_relative_traveltime_4th,
    calculate_relative_traveltime_voigt,
)


def profile_operation(func, *args, n_runs=100, warmup=10):
    """Profile a function with multiple runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timing runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times), result


def make_arbitrary_symmetric_D() -> np.ndarray:
    """Construct a deterministic, arbitrary 4th-order tensor with minor and major symmetries.

    The helper sets a handful of independent components and mirrors them so that
    D[i,j,k,l] = D[j,i,k,l] = D[i,j,l,k] = D[k,l,i,j]. This keeps tests
    deterministic while exercising off-diagonal components.
    """
    D = np.zeros((3, 3, 3, 3))

    def set_sym(i: int, j: int, k: int, l: int, value: float) -> None:  # noqa: E741
        pairs_ij = [(i, j), (j, i)]
        pairs_kl = [(k, l), (l, k)]
        for a, b in pairs_ij:
            for c, d in pairs_kl:
                D[a, b, c, d] = value
                D[c, d, a, b] = value

    # Primary diagonal-like entries
    set_sym(0, 0, 0, 0, 1.0)
    set_sym(1, 1, 1, 1, 2.0)
    set_sym(2, 2, 2, 2, 3.0)

    # Some off-diagonal couplings (mirrored by set_sym)
    set_sym(0, 0, 1, 1, 0.5)
    set_sym(0, 0, 2, 2, 0.25)
    set_sym(1, 1, 2, 2, 0.75)

    # Shear-like components
    set_sym(0, 1, 0, 1, 0.1)
    set_sym(0, 2, 0, 2, 0.2)
    set_sym(1, 2, 1, 2, 0.3)

    return D


BATCH_SIZE = 1
MAX_CELLS = 100

RNG = np.random.default_rng(42)
D_BASE = make_arbitrary_symmetric_D()
PATHS = RNG.random((5000, 3))


def profile(batch_size: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Profile both Voigt and 4th-order methods for a given batch size and number of cells."""
    D = np.broadcast_to(D_BASE, (batch_size, cells) + D_BASE.shape)
    D_voigt = elastic_tensor_to_voigt(D)

    voigt_times, _ = profile_operation(
        calculate_relative_traveltime_voigt, PATHS, D_voigt
    )
    fourth_times, _ = profile_operation(calculate_relative_traveltime_4th, PATHS, D)

    return voigt_times.mean(), fourth_times.mean()


def main():
    """Main profiling function."""
    cell_counts = range(1, MAX_CELLS + 1, MAX_CELLS // 10)
    results = Parallel(-1)(
        delayed(profile)(BATCH_SIZE, cells)
        for cells in cell_counts)
    print("-----------------------------------")

    voigt_times = np.array([r[0] for r in results])
    fourth_times = np.array([r[1] for r in results])

    rel_times = voigt_times / fourth_times

    plt.plot(cell_counts, rel_times, marker="o")
    plt.title("Relative traveltime calculation speed (Voigt vs 4th-order)")
    plt.savefig("profiling/tt_voigt_v_fourth.png")


if __name__ == "__main__":
    main()

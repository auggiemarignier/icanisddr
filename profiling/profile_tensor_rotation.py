"""Profile direct 4th-order tensor rotation vs Voigt notation approach."""

import time

import matplotlib.pyplot as plt
import numpy as np

from tti.elastic import (
    elastic_tensor_to_voigt,
    transformation_to_voigt,
    transverse_isotropic_tensor,
    voigt_to_elastic_tensor,
)
from tti.rotation import rotation_matrix_zy, transformation_4th_order


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


def rotate_direct_einsum(C_tti, R):
    """Rotate 4th-order tensor directly using einsum."""
    return np.einsum("...pi,...qj,...rk,...sl,...ijkl->...pqrs", R, R, R, R, C_tti)


def rotate_via_voigt(C_tti, R):
    """Rotate 4th-order tensor via Voigt notation conversion."""
    C_voigt = elastic_tensor_to_voigt(C_tti)
    R_voigt = transformation_to_voigt(transformation_4th_order(R))
    C_rotated_voigt = R_voigt @ C_voigt @ R_voigt.swapaxes(-2, -1)
    return voigt_to_elastic_tensor(C_rotated_voigt)


def main():
    """Run profiling comparison."""
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1_000, 10_000, 100_000]
    rng = np.random.default_rng(42)

    print("Profiling 4th-order tensor rotation methods")
    print("Number of runs: 100 (after 10 warmup runs)\n")
    print(
        f"{'Batch Size':<12} {'Direct einsum (ms)':<22} {'Via Voigt (ms)':<22} {'Speedup':<10}"
    )
    print("-" * 75)

    # Store results for plotting
    means_direct = []
    stds_direct = []
    means_voigt = []
    stds_voigt = []

    for n in batch_sizes:
        # Create random TTI tensor parameters
        A = rng.uniform(200, 300, n)
        C = rng.uniform(200, 300, n)
        F = rng.uniform(50, 100, n)
        L = rng.uniform(50, 100, n)
        N = rng.uniform(50, 100, n)

        # Create random rotation angles
        eta1 = rng.uniform(0, 2 * np.pi, n)
        eta2 = rng.uniform(0, np.pi, n)

        # Construct TTI tensor and rotation matrix
        C_tti = transverse_isotropic_tensor(A, C, F, L, N)
        R = rotation_matrix_zy(eta1, eta2)

        # Profile direct einsum approach
        times_direct, result_direct = profile_operation(rotate_direct_einsum, C_tti, R)
        mean_direct = np.mean(times_direct) * 1000  # Convert to ms
        std_direct = np.std(times_direct) * 1000

        # Profile Voigt notation approach
        times_voigt, result_voigt = profile_operation(rotate_via_voigt, C_tti, R)
        mean_voigt = np.mean(times_voigt) * 1000
        std_voigt = np.std(times_voigt) * 1000

        # Verify results are the same (within numerical precision)
        assert np.allclose(result_direct, result_voigt, rtol=1e-10), (
            f"Results differ for batch size {n}"
        )

        # Store results
        means_direct.append(mean_direct)
        stds_direct.append(std_direct)
        means_voigt.append(mean_voigt)
        stds_voigt.append(std_voigt)

        # Calculate speedup
        speedup = mean_direct / mean_voigt

        print(
            f"{n:<12} {mean_direct:>9.4f} ± {std_direct:<7.4f}   "
            f"{mean_voigt:>9.4f} ± {std_voigt:<7.4f}   {speedup:>6.2f}x"
        )

    print("\nConclusion:")
    print("  Speedup > 1: Voigt approach is faster")
    print("  Speedup < 1: Direct einsum is faster")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute times
    ax1.errorbar(
        batch_sizes,
        means_direct,
        yerr=stds_direct,
        marker="o",
        label="Direct einsum",
        capsize=5,
    )
    ax1.errorbar(
        batch_sizes,
        means_voigt,
        yerr=stds_voigt,
        marker="s",
        label="Via Voigt",
        capsize=5,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Computation Time (ms)")
    ax1.set_title("4th-Order Tensor Rotation: Computation Time vs Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup
    speedups = np.array(means_direct) / np.array(means_voigt)
    ax2.plot(batch_sizes, speedups, marker="o", color="C2")
    ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Equal performance")
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speedup (direct time / Voigt time)")
    ax2.set_title("Voigt approach speedup over direct einsum")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("profiling/tensor_rotation_comparison.png", dpi=150)
    print("\nPlot saved to profiling/tensor_rotation_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()

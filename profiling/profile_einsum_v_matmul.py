"""Profile einsum vs @ operator for batched matrix multiplications."""

import time

import matplotlib.pyplot as plt
import numpy as np


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


def matmul_at_operator(A, B, C):
    """Compute A @ B @ C.T using @ operator."""
    return A @ B @ C.swapaxes(-2, -1)


def matmul_einsum(A, B, C):
    """Compute A @ B @ C.T using einsum."""
    return np.einsum("...ij,...jk,...lk->...il", A, B, C)


def main():
    """Run profiling comparison."""
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1_000, 10_000, 100_000]
    matrix_size = 6  # Voigt notation size
    rng = np.random.default_rng(42)

    print("Profiling batched matrix multiplication: A @ B @ C.T")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print("Number of runs: 100 (after 10 warmup runs)\n")
    print(
        f"{'Batch Size':<12} {'@ operator (ms)':<20} {'einsum (ms)':<20} {'Speedup':<10}"
    )
    print("-" * 70)

    # Store results for plotting
    means_at = []
    stds_at = []
    means_einsum = []
    stds_einsum = []

    for n in batch_sizes:
        # Create random batched matrices
        A = rng.standard_normal((n, matrix_size, matrix_size))
        B = rng.standard_normal((n, matrix_size, matrix_size))
        C = rng.standard_normal((n, matrix_size, matrix_size))

        # Profile @ operator
        times_at, result_at = profile_operation(matmul_at_operator, A, B, C)
        mean_at = np.mean(times_at) * 1000  # Convert to ms
        std_at = np.std(times_at) * 1000

        # Profile einsum
        times_einsum, result_einsum = profile_operation(matmul_einsum, A, B, C)
        mean_einsum = np.mean(times_einsum) * 1000
        std_einsum = np.std(times_einsum) * 1000

        # Verify results are the same
        assert np.allclose(result_at, result_einsum), (
            f"Results differ for batch size {n}"
        )

        # Store results
        means_at.append(mean_at)
        stds_at.append(std_at)
        means_einsum.append(mean_einsum)
        stds_einsum.append(std_einsum)

        # Calculate speedup
        speedup = mean_einsum / mean_at

        print(
            f"{n:<12} {mean_at:>8.4f} ± {std_at:<6.4f}   {mean_einsum:>8.4f} ± {std_einsum:<6.4f}   {speedup:>6.2f}x"
        )

    print("\nConclusion:")
    print("  Speedup > 1: @ operator is faster")
    print("  Speedup < 1: einsum is faster")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute times
    ax1.errorbar(
        batch_sizes, means_at, yerr=stds_at, marker="o", label="@ operator", capsize=5
    )
    ax1.errorbar(
        batch_sizes,
        means_einsum,
        yerr=stds_einsum,
        marker="s",
        label="einsum",
        capsize=5,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Computation Time (ms)")
    ax1.set_title("Computation Time vs Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup
    speedups = np.array(means_einsum) / np.array(means_at)
    ax2.plot(batch_sizes, speedups, marker="o", color="C2")
    ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Equal performance")
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speedup (einsum time / @ time)")
    ax2.set_title("@ operator speedup over einsum")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("profiling/einsum_vs_matmul.png", dpi=150)
    print("\nPlot saved to profiling/einsum_vs_matmul.png")
    plt.show()


if __name__ == "__main__":
    main()

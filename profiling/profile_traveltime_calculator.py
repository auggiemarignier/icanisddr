"""Profile TravelTimeCalculator performance as the number of paths increases."""

import time

import matplotlib.pyplot as plt
import numpy as np

from tti.forward import TravelTimeCalculator


def generate_random_paths(num_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random entry and exit paths for the inner core."""
    rng = np.random.default_rng(42)

    lon_in = rng.uniform(-180, 180, num_paths)
    lat_in = rng.uniform(-90, 90, num_paths)
    r_in = rng.uniform(1000, 4000, num_paths)

    lon_out = rng.uniform(-180, 180, num_paths)
    lat_out = rng.uniform(-90, 90, num_paths)
    r_out = rng.uniform(1000, 4000, num_paths)

    ic_in = np.column_stack((lon_in, lat_in, r_in))
    ic_out = np.column_stack((lon_out, lat_out, r_out))

    return ic_in, ic_out


def profile_travel_time_calculator(max_paths: int, step: int) -> None:
    """Profile the TravelTimeCalculator performance."""
    num_paths_list = list(range(step, max_paths + 1, step))
    times = []
    rng = np.random.default_rng(42)

    for num_paths in num_paths_list:
        ic_in, ic_out = generate_random_paths(num_paths)
        calculator = TravelTimeCalculator(ic_in, ic_out)

        # Random model parameters [A, C, F, L, N, eta1, eta2]
        model_params = rng.uniform(
            [1, 1, 0.5, 0.5, 0.5, 0, 0], [10, 10, 5, 5, 5, np.pi, np.pi]
        )

        start_time = time.perf_counter()
        calculator(model_params)
        elapsed_time = time.perf_counter() - start_time

        times.append(elapsed_time)
        print(f"Paths: {num_paths:4d}, Time: {elapsed_time * 1e3:8.2f} ms")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_paths_list, np.array(times) * 1e3, marker="o")
    plt.title("TravelTimeCalculator Performance")
    plt.xlabel("Number of Paths")
    plt.ylabel("Time (ms)")
    plt.grid()
    plt.savefig("profiling/travel_time_performance.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    profile_travel_time_calculator(max_paths=10_000, step=100)

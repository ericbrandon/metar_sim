"""
Obstacle Removal Simulation

Simulates how removing derailing obstacles from an LLM workflow affects
"time on task". Each obstacle has a 1% chance of occurring at each time step,
and any occurrence halts the process.

The key insight: the probability of surviving a single time step is
0.99^(number_of_problems). As problems are removed, survival probability
per step increases, and total time on task grows hyperbolically.
"""

import random
import matplotlib.pyplot as plt


def run_simulation(num_problems: int, seed: int | None = None) -> int:
    """Run a single simulation and return the number of time steps completed.

    At each step, we draw a random number in [0, 1). If it is >= 0.99^num_problems
    (i.e. we hit at least one obstacle), the simulation stops.

    Args:
        num_problems: Number of active obstacles.
        seed: Optional random seed for reproducibility.

    Returns:
        Number of time steps completed before being stopped.
    """
    rng = random.Random(seed)
    threshold = 0.99 ** num_problems  # probability of surviving one step
    steps = 0
    while True:
        if rng.random() >= threshold:
            break
        steps += 1
    return steps


def run_batch(num_problems: int, num_simulations: int = 10) -> float:
    """Run multiple simulations and return the average steps completed."""
    total = sum(run_simulation(num_problems) for _ in range(num_simulations))
    return total / num_simulations


def main():
    max_problems = 50
    num_simulations = 10

    problems_list = list(range(max_problems, 0, -1))  # 50 down to 1
    avg_steps_list = []

    for num_problems in problems_list:
        avg_steps = run_batch(num_problems, num_simulations)
        avg_steps_list.append(avg_steps)
        print(f"Problems: {num_problems:3d}  |  Avg steps: {avg_steps:10.1f}")

    # Also include 0 problems (infinite run — cap at a large number for display)
    # With 0 problems, threshold = 1.0 and the simulation never stops.
    # We skip 0 to avoid an infinite loop, but note the theoretical value is infinity.

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: linear Y scale
    ax1.plot(problems_list, avg_steps_list, "o-", color="steelblue", markersize=4)
    ax1.set_xlabel("Number of Problems")
    ax1.set_ylabel("Average Time Steps Before Failure")
    ax1.set_title("Obstacle Removal vs Time on Task (Linear Scale)")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    # Right plot: log Y scale to show the hyperbolic / exponential growth
    ax2.plot(problems_list, avg_steps_list, "o-", color="darkorange", markersize=4)
    ax2.set_xlabel("Number of Problems")
    ax2.set_ylabel("Average Time Steps Before Failure (log scale)")
    ax2.set_title("Obstacle Removal vs Time on Task (Log Scale)")
    ax2.set_yscale("log")
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3, which="both")

    # Add theoretical curve: E[steps] = 1 / (1 - 0.99^n) for geometric distribution
    import numpy as np
    n_theory = np.arange(1, max_problems + 1)
    expected_steps = 1.0 / (1.0 - 0.99 ** n_theory)
    ax1.plot(n_theory, expected_steps, "--", color="red", alpha=0.6, label="Theoretical E[steps]")
    ax2.plot(n_theory, expected_steps, "--", color="red", alpha=0.6, label="Theoretical E[steps]")
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig("obstacle_simulation_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to obstacle_simulation_results.png")
    plt.show()


if __name__ == "__main__":
    main()

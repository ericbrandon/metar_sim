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
    num_simulations = 20

    problems_list = list(range(max_problems, 0, -1))  # 50 down to 1
    avg_steps_list = []

    for num_problems in problems_list:
        avg_steps = run_batch(num_problems, num_simulations)
        avg_steps_list.append(avg_steps)
        print(f"Problems: {num_problems:3d}  |  Avg steps: {avg_steps:10.1f}")

    # Plot results — single linear plot
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(problems_list, avg_steps_list, "o-", color="steelblue", markersize=4, label="Simulated (20 runs)")
    ax.set_xlabel("Number of Problems")
    ax.set_ylabel("Average Time Steps Before Failure")
    ax.set_title("Obstacle Removal vs Time on Task")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Theoretical curve: E[steps] = 1 / (1 - 0.99^n) for geometric distribution
    n_theory = np.arange(1, max_problems + 1)
    expected_steps = 1.0 / (1.0 - 0.99 ** n_theory)
    ax.plot(n_theory, expected_steps, "--", color="red", alpha=0.6, label="Theoretical E[steps]")
    ax.legend()

    plt.tight_layout()
    plt.savefig("obstacle_simulation_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to obstacle_simulation_results.png")
    plt.show()


if __name__ == "__main__":
    main()

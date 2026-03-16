from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.analysis import BenchmarkResults


def plot_results(
    benchmark: BenchmarkResults,
    output_path: str = "results.png",
) -> None:
    task_sizes = benchmark.task_sizes
    policies = benchmark.policy_names

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Monte Carlo completion-related failure trends vs task count.
    ax_mc = axes[0]
    y_lows: list[float] = []
    y_highs: list[float] = []
    for policy_name in policies:
        completion_mean = [
            benchmark.aggregates[n][policy_name].mc_mean["task_completion"] for n in task_sizes
        ]
        completion_std = [
            benchmark.aggregates[n][policy_name].mc_std["task_completion"] for n in task_sizes
        ]
        low = [max(0.0, m - s) for m, s in zip(completion_mean, completion_std)]
        high = [min(1.0, m + s) for m, s in zip(completion_mean, completion_std)]
        y_lows.extend(low)
        y_highs.extend(high)

        ax_mc.plot(task_sizes, completion_mean, marker="o", label=f"{policy_name} completion")
        ax_mc.fill_between(
            task_sizes,
            low,
            high,
            alpha=0.15,
        )

    ax_mc.set_title("Monte Carlo Completion Failure vs Task Count")
    ax_mc.set_ylabel("Violation Probability")
    if y_lows and y_highs:
        ymin = min(y_lows)
        ymax = max(y_highs)
        yrange = max(0.08, ymax - ymin)
        margin = 0.08 * yrange + 0.02
        ax_mc.set_ylim(max(-0.02, ymin - margin), min(1.04, ymax + margin))
    else:
        ax_mc.set_ylim(0.0, 1.0)
    ax_mc.set_xlabel("Number of Tasks")
    ax_mc.set_xticks(task_sizes)
    ax_mc.legend(fontsize=8)
    ax_mc.grid(axis="y", alpha=0.3)

    # Subplot 2: SAT categorical status matrix for task completion.
    ax_sat = axes[1]
    status_grid = np.zeros((len(policies), len(task_sizes)))
    # 0=safe, 1=timeout-dominant, 2=unsafe-dominant
    for i, policy_name in enumerate(policies):
        for j, n_tasks in enumerate(task_sizes):
            agg = benchmark.aggregates[n_tasks][policy_name]
            unsafe_rate = agg.sat_unsafe_rate["task_completion"]
            timeout_rate = agg.sat_timeout_rate["task_completion"]
            safe_rate = max(0.0, 1.0 - unsafe_rate - timeout_rate)
            if unsafe_rate >= timeout_rate and unsafe_rate >= safe_rate:
                status_grid[i, j] = 2
            elif timeout_rate >= unsafe_rate and timeout_rate >= safe_rate:
                status_grid[i, j] = 1
            else:
                status_grid[i, j] = 0

    cmap = ListedColormap(["#4daf4a", "#ffb347", "#e41a1c"])
    image = ax_sat.imshow(status_grid, vmin=0, vmax=2, aspect="auto", cmap=cmap)

    ax_sat.set_title("SAT Status (task_completion)")
    ax_sat.set_xlabel("Number of Tasks")
    ax_sat.set_ylabel("Policy")
    ax_sat.set_xticks(range(len(task_sizes)))
    ax_sat.set_xticklabels(task_sizes)
    ax_sat.set_yticks(range(len(policies)))
    ax_sat.set_yticklabels(policies)

    for i, policy_name in enumerate(policies):
        for j, n_tasks in enumerate(task_sizes):
            timeout_rate = benchmark.aggregates[n_tasks][policy_name].sat_timeout_rate["task_completion"]
            unsafe_rate = benchmark.aggregates[n_tasks][policy_name].sat_unsafe_rate["task_completion"]
            safe_rate = max(0.0, 1.0 - unsafe_rate - timeout_rate)
            ax_sat.text(
                j,
                i,
                f"U{unsafe_rate:.2f}\nS{safe_rate:.2f}\nT{timeout_rate:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    cbar = fig.colorbar(image, ax=ax_sat, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["safe", "timeout", "unsafe"])
    cbar.set_label("Dominant SAT status")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

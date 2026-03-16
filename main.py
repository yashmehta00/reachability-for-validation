from __future__ import annotations

import argparse

from src.analysis import BenchmarkResults, plot_results, run_benchmark
from src.model import (
    ConservativePolicy,
    Disturbance,
    Environment,
    GreedyPolicy,
    Policy,
    PriorityPolicy,
    generate_environment,
)
from src.validation import MONITORS, simulate


DEFAULT_TASK_SIZES = [4, 6, 8, 10, 12]


def parse_task_sizes(raw: str) -> list[int]:
    sizes = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not sizes:
        raise ValueError("task_sizes must contain at least one value")
    if any(size <= 0 for size in sizes):
        raise ValueError("task_sizes values must be positive integers")
    return sorted(set(sizes))


def print_environment_preview(env: Environment, disturbance: Disturbance) -> None:
    task_info = ", ".join(
        f"T{task.id}(dur={task.base_duration},r={task.resource_req})" for task in env.tasks
    )
    resource_info = ", ".join(f"R{resource.id}(cap={resource.capacity})" for resource in env.resources)

    print("Environment Preview")
    print("-" * 50)
    print(f"Tasks      : {task_info}")
    print(f"Resources  : {resource_info}")
    print(f"Precedence : {env.precedence}")
    print(f"Horizon    : {env.horizon}")
    print(
        "Disturbance: "
        f"duration_noise_prob={disturbance.duration_noise_prob}, "
        f"delay_event_prob={disturbance.delay_event_prob}"
    )
    print()


def print_results(benchmark: BenchmarkResults) -> None:
    print("Benchmark Summary")
    print("=" * 70)
    print(f"Task sizes: {benchmark.task_sizes}")
    print(f"Policies  : {benchmark.policy_names}")
    print()

    for n_tasks in benchmark.task_sizes:
        print(f"n_tasks={n_tasks}")
        print("-" * 70)
        header = (
            f"{'Policy':<36s}"
            f"{'MC completion':>14s}"
            f"{'MC no_abandon':>14s}"
            f"{'SAT unsafe':>12s}"
            f"{'SAT timeout':>12s}"
            f"{'SAT sec':>10s}"
        )
        print(header)
        print("-" * len(header))
        for policy_name in benchmark.policy_names:
            aggregate = benchmark.aggregates[n_tasks][policy_name]
            print(
                f"{policy_name:<36s}"
                f"{aggregate.mc_mean['task_completion']:>14.3f}"
                f"{aggregate.mc_mean['no_abandonment']:>14.3f}"
                f"{aggregate.sat_unsafe_rate['task_completion']:>12.3f}"
                f"{aggregate.sat_timeout_rate['task_completion']:>12.3f}"
                f"{aggregate.avg_sat_runtime_sec:>10.3f}"
            )
        print()


def print_example_counterexample(
    policies: list[Policy],
    disturbance: Disturbance,
    task_sizes: list[int],
) -> None:
    # Print one representative trajectory to keep output concise.
    largest = max(task_sizes)
    env = generate_environment(n_tasks=largest, seed=4242)
    policy = policies[0]
    trajectory = simulate(env, policy, disturbance, seed=777)
    print(f"Example trajectory (policy={policy.name}, n_tasks={largest})")
    print("-" * 70)
    for state in trajectory[: min(8, len(trajectory))]:
        z_string = " ".join(f"T{tid}={int(status)}" for tid, status in sorted(state.z.items()))
        print(f"t={state.t:2d} [{z_string}] loads={state.l} remaining={state.remaining}")
    if len(trajectory) > 8:
        print("...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAT + Monte Carlo scheduling benchmark")
    parser.add_argument(
        "--task-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TASK_SIZES),
        help="Comma-separated task counts, e.g. 4,6,8,10,12",
    )
    parser.add_argument("--env-seeds", type=int, default=5, help="Number of random environments per task size")
    parser.add_argument("--mc-trials", type=int, default=600, help="Monte Carlo trials per environment")
    parser.add_argument("--sat-timeout-sec", type=float, default=30.0, help="SAT timeout per policy/environment")
    args = parser.parse_args()

    task_sizes = parse_task_sizes(args.task_sizes)
    disturbance = Disturbance(duration_noise_prob=0.2, delay_event_prob=0.1)
    policies: list[Policy] = [
        GreedyPolicy(),
        ConservativePolicy(base_slack=1, scale_with_tasks=0.06),
        PriorityPolicy(w1=2.8, w2=1.1, w3=0.1),
    ]

    preview_env = generate_environment(n_tasks=task_sizes[0], seed=7)
    print_environment_preview(preview_env, disturbance)
    print(f"Properties: {list(MONITORS.keys())}\n")
    if args.env_seeds < 5:
        print(
            "Note: env-seeds < 5 gives coarse rates (e.g. 0.00/0.50/1.00); "
            "use >= 8 for smoother comparisons.\n"
        )

    benchmark = run_benchmark(
        task_sizes=task_sizes,
        policies=policies,
        num_env_seeds=args.env_seeds,
        monte_carlo_trials=args.mc_trials,
        disturbance=disturbance,
        sat_timeout_sec=args.sat_timeout_sec,
        base_seed=1000,
    )

    print_results(benchmark)
    print_example_counterexample(policies=policies, disturbance=disturbance, task_sizes=task_sizes)

    plot_results(benchmark, output_path="results.png")
    print("Saved summary figure: results.png")


if __name__ == "__main__":
    main()

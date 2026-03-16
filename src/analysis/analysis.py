from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.model.env import Environment, generate_environment
from src.model.policy import Policy
from src.validation.sat_reachability import SATReachabilityResult, sat_bounded_reachability
from src.model.scheduler import Disturbance
from src.validation.simulation import MONITORS, MonteCarloResult, monte_carlo_validate


def run_monte_carlo(
    env: Environment,
    policies: list[Policy],
    disturbance: Disturbance,
    num_trials: int,
    base_seed: int = 42,
) -> dict[str, MonteCarloResult]:
    results: dict[str, MonteCarloResult] = {}
    for idx, policy in enumerate(policies):
        results[policy.name] = monte_carlo_validate(
            env=env,
            policy=policy,
            disturbance=disturbance,
            num_trials=num_trials,
            base_seed=base_seed + idx * num_trials,
        )
    return results


def run_sat_reachability(
    env: Environment,
    policies: list[Policy],
    bound: int | None = None,
    timeout_sec: float | None = None,
) -> dict[str, SATReachabilityResult]:
    results: dict[str, SATReachabilityResult] = {}
    for policy in policies:
        results[policy.name] = sat_bounded_reachability(
            env=env,
            policy=policy,
            bound=bound,
            timeout_sec=timeout_sec,
        )
    return results


@dataclass
class PolicyTaskAggregate:
    mc_mean: dict[str, float]
    mc_std: dict[str, float]
    sat_unsafe_rate: dict[str, float]
    sat_timeout_rate: dict[str, float]
    avg_sat_runtime_sec: float
    avg_sat_states: float


@dataclass
class BenchmarkResults:
    task_sizes: list[int]
    policy_names: list[str]
    aggregates: dict[int, dict[str, PolicyTaskAggregate]]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def run_benchmark(
    task_sizes: list[int],
    policies: list[Policy],
    num_env_seeds: int,
    monte_carlo_trials: int,
    disturbance: Disturbance,
    sat_timeout_sec: float,
    base_seed: int = 1000,
) -> BenchmarkResults:
    policy_names = [policy.name for policy in policies]
    aggregates: dict[int, dict[str, PolicyTaskAggregate]] = {}

    for n_tasks in task_sizes:
        by_policy_mc: dict[str, dict[str, list[float]]] = {
            policy_name: {property_name: [] for property_name in MONITORS}
            for policy_name in policy_names
        }
        by_policy_sat_unsafe: dict[str, dict[str, int]] = {
            policy_name: {property_name: 0 for property_name in MONITORS}
            for policy_name in policy_names
        }
        by_policy_sat_timeout: dict[str, dict[str, int]] = {
            policy_name: {property_name: 0 for property_name in MONITORS}
            for policy_name in policy_names
        }
        by_policy_sat_runtime: dict[str, list[float]] = {policy_name: [] for policy_name in policy_names}
        by_policy_sat_states: dict[str, list[float]] = {policy_name: [] for policy_name in policy_names}

        for env_seed_idx in range(num_env_seeds):
            env_seed = base_seed + n_tasks * 100 + env_seed_idx
            env = generate_environment(n_tasks=n_tasks, seed=env_seed)

            for policy_idx, policy in enumerate(policies):
                policy_name = policy.name
                mc_result = monte_carlo_validate(
                    env=env,
                    policy=policy,
                    disturbance=disturbance,
                    num_trials=monte_carlo_trials,
                    base_seed=env_seed * 1000 + policy_idx * monte_carlo_trials,
                )
                sat_result = sat_bounded_reachability(
                    env=env,
                    policy=policy,
                    bound=env.horizon,
                    timeout_sec=sat_timeout_sec,
                )

                for property_name in MONITORS:
                    by_policy_mc[policy_name][property_name].append(
                        mc_result.violation_rates[property_name]
                    )
                    property_status = sat_result.property_results[property_name].status
                    if property_status == "unsafe":
                        by_policy_sat_unsafe[policy_name][property_name] += 1
                    elif property_status == "timeout":
                        by_policy_sat_timeout[policy_name][property_name] += 1

                by_policy_sat_runtime[policy_name].append(sat_result.runtime_sec)
                by_policy_sat_states[policy_name].append(float(sat_result.num_states_explored))

        aggregates[n_tasks] = {}
        for policy_name in policy_names:
            aggregates[n_tasks][policy_name] = PolicyTaskAggregate(
                mc_mean={
                    property_name: _mean(by_policy_mc[policy_name][property_name])
                    for property_name in MONITORS
                },
                mc_std={
                    property_name: _std(by_policy_mc[policy_name][property_name])
                    for property_name in MONITORS
                },
                sat_unsafe_rate={
                    property_name: by_policy_sat_unsafe[policy_name][property_name] / num_env_seeds
                    for property_name in MONITORS
                },
                sat_timeout_rate={
                    property_name: by_policy_sat_timeout[policy_name][property_name] / num_env_seeds
                    for property_name in MONITORS
                },
                avg_sat_runtime_sec=_mean(by_policy_sat_runtime[policy_name]),
                avg_sat_states=_mean(by_policy_sat_states[policy_name]),
            )

    return BenchmarkResults(
        task_sizes=task_sizes,
        policy_names=policy_names,
        aggregates=aggregates,
    )

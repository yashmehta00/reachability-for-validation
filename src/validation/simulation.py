from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from src.model.env import Environment, TaskStatus
from src.model.policy import Policy
from src.model.scheduler import Disturbance, transition
from src.model.state import State, make_initial_state


def check_resource_overload(trajectory: list[State], env: Environment) -> bool:
    """G(no resource overload)."""
    return all(
        state.l[resource.id] <= resource.capacity
        for state in trajectory
        for resource in env.resources
    )


def check_precedence_violation(trajectory: list[State], env: Environment) -> bool:
    """G(task j active => predecessor i complete)."""
    return all(
        not (state.z[j] == TaskStatus.ACTIVE and state.z[i] != TaskStatus.COMPLETE)
        for state in trajectory
        for i, j in env.precedence
    )


def check_task_completion(trajectory: list[State], env: Environment) -> bool:
    """F(all tasks complete)."""
    final_state = trajectory[-1]
    return all(final_state.z[task.id] == TaskStatus.COMPLETE for task in env.tasks)


def check_no_abandonment(trajectory: list[State], env: Environment) -> bool:
    """G(active(i) => F(complete(i)))."""
    ever_active = {
        task.id
        for state in trajectory
        for task in env.tasks
        if state.z[task.id] == TaskStatus.ACTIVE
    }
    final_state = trajectory[-1]
    return all(final_state.z[tid] == TaskStatus.COMPLETE for tid in ever_active)


MONITORS: dict[str, Callable[[list[State], Environment], bool]] = {
    "resource_overload": check_resource_overload,
    "precedence_violation": check_precedence_violation,
    "task_completion": check_task_completion,
    "no_abandonment": check_no_abandonment,
}


@dataclass
class MonteCarloResult:
    num_trials: int
    violation_counts: dict[str, int]

    @property
    def violation_rates(self) -> dict[str, float]:
        return {
            property_name: count / self.num_trials
            for property_name, count in self.violation_counts.items()
        }


def simulate(
    env: Environment,
    policy: Policy,
    disturbance: Disturbance,
    seed: int | None = None,
) -> list[State]:
    rng = random.Random(seed)
    state = make_initial_state(env)
    trajectory = [state]

    for _ in range(env.horizon):
        state = transition(state, env, policy, disturbance, rng)
        trajectory.append(state)
        if state.all_complete(env):
            break

    return trajectory


def monte_carlo_validate(
    env: Environment,
    policy: Policy,
    disturbance: Disturbance,
    num_trials: int = 1000,
    base_seed: int = 42,
) -> MonteCarloResult:
    violation_counts: dict[str, int] = {name: 0 for name in MONITORS}

    for trial_idx in range(num_trials):
        trajectory = simulate(env, policy, disturbance, seed=base_seed + trial_idx)
        for property_name, monitor in MONITORS.items():
            if not monitor(trajectory, env):
                violation_counts[property_name] += 1

    return MonteCarloResult(num_trials=num_trials, violation_counts=violation_counts)

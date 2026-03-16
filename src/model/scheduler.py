from __future__ import annotations

import random
from dataclasses import dataclass

from src.model.env import Environment, TaskStatus
from src.model.policy import Policy
from src.model.state import State


@dataclass
class Disturbance:
    duration_noise_prob: float = 0.2
    delay_event_prob: float = 0.1

    def sample_duration(self, base: int, rng: random.Random) -> int:
        return base + (1 if rng.random() < self.duration_noise_prob else 0)

    def sample_delay(self, rng: random.Random) -> bool:
        return rng.random() < self.delay_event_prob


def transition(
    state: State,
    env: Environment,
    policy: Policy,
    disturbance: Disturbance,
    rng: random.Random,
) -> State:
    s = state.copy()

    for tid, rid in policy(s, env).items():
        task = env.task_by_id(tid)
        s.z[tid] = TaskStatus.ACTIVE
        s.a[tid] = rid
        s.remaining[tid] = disturbance.sample_duration(task.base_duration, rng)
        s.l[rid] += 1

    completed: list[int] = []
    for tid in list(s.remaining):
        if disturbance.sample_delay(rng):
            continue
        s.remaining[tid] -= 1
        if s.remaining[tid] <= 0:
            completed.append(tid)

    for tid in completed:
        s.z[tid] = TaskStatus.COMPLETE
        rid = s.a[tid]
        s.a[tid] = None
        del s.remaining[tid]
        if rid is not None:
            s.l[rid] -= 1

    s.t += 1
    return s


def deterministic_transition(
    state: State,
    env: Environment,
    policy: Policy,
    dur_extra: dict[int, int],
    delay_set: frozenset[int],
) -> State:
    """Transition under a fixed disturbance realization."""
    s = state.copy()

    for tid, rid in policy(s, env).items():
        task = env.task_by_id(tid)
        s.z[tid] = TaskStatus.ACTIVE
        s.a[tid] = rid
        s.remaining[tid] = task.base_duration + dur_extra.get(tid, 0)
        s.l[rid] += 1

    completed: list[int] = []
    for tid in list(s.remaining):
        if tid in delay_set:
            continue
        s.remaining[tid] -= 1
        if s.remaining[tid] <= 0:
            completed.append(tid)

    for tid in completed:
        s.z[tid] = TaskStatus.COMPLETE
        rid = s.a[tid]
        s.a[tid] = None
        del s.remaining[tid]
        if rid is not None:
            s.l[rid] -= 1

    s.t += 1
    return s

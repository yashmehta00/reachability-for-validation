from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State


class TaskStatus(IntEnum):
    NOT_STARTED = 0
    ACTIVE = 1
    COMPLETE = 2


@dataclass(frozen=True)
class Task:
    id: int
    base_duration: int
    resource_req: int


@dataclass(frozen=True)
class Resource:
    id: int
    capacity: int


@dataclass
class Environment:
    tasks: list[Task]
    resources: list[Resource]
    precedence: list[tuple[int, int]]
    horizon: int

    def predecessors(self, task_id: int) -> list[int]:
        return [i for i, j in self.precedence if j == task_id]

    def successors(self, task_id: int) -> list[int]:
        return [j for i, j in self.precedence if i == task_id]

    def task_by_id(self, tid: int) -> Task:
        for task in self.tasks:
            if task.id == tid:
                return task
        raise KeyError(tid)

    def feasible_tasks(self, state: State) -> list[Task]:
        return [
            task
            for task in self.tasks
            if state.z[task.id] == TaskStatus.NOT_STARTED
            and all(state.z[p] == TaskStatus.COMPLETE for p in self.predecessors(task.id))
        ]


@dataclass(frozen=True)
class EnvironmentGeneratorConfig:
    max_duration: int = 4
    min_precedence_density: float = 0.10
    max_precedence_density: float = 0.28
    base_horizon_slack: float = 1.45
    large_n_horizon_bonus: float = 0.006


def generate_environment(
    n_tasks: int,
    n_resources: int | None = None,
    max_duration: int | None = None,
    precedence_density: float | None = None,
    horizon_multiplier: float | None = None,
    seed: int = 7,
    config: EnvironmentGeneratorConfig | None = None,
) -> Environment:
    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1")

    cfg = config or EnvironmentGeneratorConfig()
    rng = random.Random(seed)
    # Scale resources gently with problem size to avoid trivially impossible large instances.
    resolved_resources = n_resources if n_resources is not None else max(2, (n_tasks + 2) // 3)
    resolved_duration_cap = max_duration if max_duration is not None else cfg.max_duration

    if precedence_density is None:
        # Keep precedence structure meaningful without saturating large DAGs.
        scaled_density = 0.12 + 0.008 * n_tasks
        resolved_density = max(
            cfg.min_precedence_density,
            min(cfg.max_precedence_density, scaled_density),
        )
    else:
        resolved_density = precedence_density

    tasks = [
        Task(
            id=i,
            base_duration=rng.randint(1, resolved_duration_cap),
            resource_req=i % resolved_resources,
        )
        for i in range(n_tasks)
    ]

    precedence: list[tuple[int, int]] = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if rng.random() < resolved_density:
                precedence.append((i, j))

    total_work = sum(task.base_duration for task in tasks)
    if horizon_multiplier is None:
        # Allow modestly more slack for larger n to avoid universal completion failure.
        slack_factor = cfg.base_horizon_slack + cfg.large_n_horizon_bonus * n_tasks
        resolved_horizon_multiplier = min(1.85, max(1.2, slack_factor))
    else:
        resolved_horizon_multiplier = horizon_multiplier

    horizon = max(
        n_tasks + 2,
        int(total_work * resolved_horizon_multiplier / max(1, resolved_resources)) + n_tasks // 3,
    )

    resources = [Resource(id=r, capacity=1) for r in range(resolved_resources)]

    return Environment(tasks=tasks, resources=resources, precedence=precedence, horizon=horizon)

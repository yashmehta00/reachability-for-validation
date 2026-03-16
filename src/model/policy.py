from __future__ import annotations

from abc import ABC, abstractmethod

from src.model.env import Environment, Task
from src.model.state import State


class Policy(ABC):
    @abstractmethod
    def __call__(self, state: State, env: Environment) -> dict[int, int]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class GreedyPolicy(Policy):
    """Assign available resources to feasible tasks in id order."""

    def __call__(self, state: State, env: Environment) -> dict[int, int]:
        assignments: dict[int, int] = {}
        spare = {resource.id: resource.capacity - state.l[resource.id] for resource in env.resources}
        feasible = sorted(env.feasible_tasks(state), key=lambda task: task.id)

        for task in feasible:
            if spare.get(task.resource_req, 0) > 0:
                assignments[task.id] = task.resource_req
                spare[task.resource_req] -= 1
        return assignments


class ConservativePolicy(Policy):
    """Start tasks only when enough horizon remains."""

    def __init__(self, base_slack: int = 1, scale_with_tasks: float = 0.06):
        self.base_slack = base_slack
        self.scale_with_tasks = scale_with_tasks

    @property
    def name(self) -> str:
        return f"Conservative(s={self.base_slack},k={self.scale_with_tasks:.2f})"

    def _effective_slack(self, env: Environment) -> int:
        return self.base_slack + int(len(env.tasks) * self.scale_with_tasks)

    def __call__(self, state: State, env: Environment) -> dict[int, int]:
        assignments: dict[int, int] = {}
        spare = {resource.id: resource.capacity - state.l[resource.id] for resource in env.resources}
        feasible = sorted(env.feasible_tasks(state), key=lambda task: task.id)
        time_left = env.horizon - state.t
        slack = self._effective_slack(env)

        for task in feasible:
            if task.base_duration + slack > time_left:
                continue
            if spare.get(task.resource_req, 0) > 0:
                assignments[task.id] = task.resource_req
                spare[task.resource_req] -= 1
        return assignments


class PriorityPolicy(Policy):
    """
    Rank feasible tasks by:
      w1 * num_successors - w2 * duration - w3 * slack
    """

    def __init__(self, w1: float = 2.8, w2: float = 1.1, w3: float = 0.1):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    @property
    def name(self) -> str:
        return f"Priority(w1={self.w1:.1f},w2={self.w2:.1f},w3={self.w3:.1f})"

    def _score(self, task: Task, state: State, env: Environment) -> float:
        num_successors = len(env.successors(task.id))
        slack = (env.horizon - state.t) - task.base_duration
        return self.w1 * num_successors - self.w2 * task.base_duration - self.w3 * slack

    def __call__(self, state: State, env: Environment) -> dict[int, int]:
        assignments: dict[int, int] = {}
        spare = {resource.id: resource.capacity - state.l[resource.id] for resource in env.resources}
        feasible = env.feasible_tasks(state)

        for task in sorted(
            feasible,
            key=lambda t: (self._score(t, state, env), -t.base_duration, -t.id),
            reverse=True,
        ):
            if spare.get(task.resource_req, 0) > 0:
                assignments[task.id] = task.resource_req
                spare[task.resource_req] -= 1

        return assignments

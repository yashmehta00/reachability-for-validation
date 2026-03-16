from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.model.env import Environment, TaskStatus


@dataclass
class State:
    z: dict[int, TaskStatus]
    a: dict[int, Optional[int]]
    remaining: dict[int, int]
    l: dict[int, int]
    t: int

    def copy(self) -> "State":
        return State(
            z=dict(self.z),
            a=dict(self.a),
            remaining=dict(self.remaining),
            l=dict(self.l),
            t=self.t,
        )

    def to_hashable(self) -> tuple:
        z_t = tuple(sorted(self.z.items()))
        a_t = tuple(sorted(self.a.items()))
        rem_t = tuple(sorted(self.remaining.items()))
        l_t = tuple(sorted(self.l.items()))
        return (z_t, a_t, rem_t, l_t, self.t)

    def all_complete(self, env: Environment) -> bool:
        return all(self.z[task.id] == TaskStatus.COMPLETE for task in env.tasks)


def make_initial_state(env: Environment) -> State:
    return State(
        z={task.id: TaskStatus.NOT_STARTED for task in env.tasks},
        a={task.id: None for task in env.tasks},
        remaining={},
        l={resource.id: 0 for resource in env.resources},
        t=0,
    )

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from src.model.env import Environment, TaskStatus
from src.model.policy import Policy
from src.model.scheduler import deterministic_transition
from src.model.state import State, make_initial_state


PROPERTY_ORDER = [
    "resource_overload",
    "precedence_violation",
    "task_completion",
    "no_abandonment",
]


@dataclass
class PropertySATResult:
    status: str  # "unsafe", "safe", "timeout"
    counterexample_trajectory: list[State] | None


@dataclass
class SATReachabilityResult:
    violation_found: bool
    violating_property: str
    counterexample_trajectory: list[State] | None
    property_results: dict[str, PropertySATResult]
    num_states_explored: int
    bound: int
    runtime_sec: float
    timed_out: bool


@dataclass
class _BoundedGraph:
    bound: int
    layers: list[list[tuple]]
    transitions: list[dict[tuple, set[tuple]]]
    predecessors: list[dict[tuple, set[tuple]]]
    states: dict[tuple, State]
    initial_hash: tuple


class _BuildTimeout(RuntimeError):
    def __init__(self, num_states: int):
        super().__init__("SAT graph build timed out")
        self.num_states = num_states


def _resource_overload(state: State, env: Environment) -> bool:
    return any(state.l[resource.id] > resource.capacity for resource in env.resources)


def _precedence_violation(state: State, env: Environment) -> bool:
    return any(
        state.z[j] == TaskStatus.ACTIVE and state.z[i] != TaskStatus.COMPLETE
        for i, j in env.precedence
    )


def _task_complete(state: State, env: Environment) -> bool:
    return all(state.z[task.id] == TaskStatus.COMPLETE for task in env.tasks)


def _active(state: State, task_id: int) -> bool:
    return state.z[task_id] == TaskStatus.ACTIVE


def _complete(state: State, task_id: int) -> bool:
    return state.z[task_id] == TaskStatus.COMPLETE


def _expand_disturbances(state: State, env: Environment, policy: Policy) -> list[State]:
    assignments = list(policy(state, env).keys())
    active_ids = list(state.remaining.keys())
    all_running = active_ids + assignments

    next_states: list[State] = []
    for dur_bits in itertools.product([0, 1], repeat=len(assignments)):
        dur_extra = dict(zip(assignments, dur_bits))
        for delay_bits in itertools.product([False, True], repeat=len(all_running)):
            delay_set = frozenset(tid for tid, delayed in zip(all_running, delay_bits) if delayed)
            next_states.append(
                deterministic_transition(
                    state=state,
                    env=env,
                    policy=policy,
                    dur_extra=dur_extra,
                    delay_set=delay_set,
                )
            )
    return next_states


def _build_bounded_graph(env: Environment, policy: Policy, bound: int) -> _BoundedGraph:
    initial = make_initial_state(env)
    initial_hash = initial.to_hashable()

    layers: list[list[tuple]] = [[initial_hash]]
    transitions: list[dict[tuple, set[tuple]]] = []
    predecessors: list[dict[tuple, set[tuple]]] = []
    states: dict[tuple, State] = {initial_hash: initial}

    for step in range(bound):
        del step
        current_layer = layers[-1]
        next_layer_set: set[tuple] = set()
        step_transitions: dict[tuple, set[tuple]] = defaultdict(set)
        step_predecessors: dict[tuple, set[tuple]] = defaultdict(set)

        for state_hash in current_layer:
            state = states[state_hash]
            if state.all_complete(env):
                step_transitions[state_hash].add(state_hash)
                step_predecessors[state_hash].add(state_hash)
                next_layer_set.add(state_hash)
                continue

            for next_state in _expand_disturbances(state, env, policy):
                next_hash = next_state.to_hashable()
                if next_hash not in states:
                    states[next_hash] = next_state
                step_transitions[state_hash].add(next_hash)
                step_predecessors[next_hash].add(state_hash)
                next_layer_set.add(next_hash)

        layers.append(sorted(next_layer_set, key=str))
        transitions.append(step_transitions)
        predecessors.append(step_predecessors)

    return _BoundedGraph(
        bound=bound,
        layers=layers,
        transitions=transitions,
        predecessors=predecessors,
        states=states,
        initial_hash=initial_hash,
    )


def _build_bounded_graph_with_deadline(
    env: Environment,
    policy: Policy,
    bound: int,
    deadline: float | None,
) -> _BoundedGraph:
    initial = make_initial_state(env)
    initial_hash = initial.to_hashable()

    layers: list[list[tuple]] = [[initial_hash]]
    transitions: list[dict[tuple, set[tuple]]] = []
    predecessors: list[dict[tuple, set[tuple]]] = []
    states: dict[tuple, State] = {initial_hash: initial}

    for _step in range(bound):
        if deadline is not None and time.perf_counter() > deadline:
            raise _BuildTimeout(num_states=len(states))

        current_layer = layers[-1]
        next_layer_set: set[tuple] = set()
        step_transitions: dict[tuple, set[tuple]] = defaultdict(set)
        step_predecessors: dict[tuple, set[tuple]] = defaultdict(set)

        for state_hash in current_layer:
            if deadline is not None and time.perf_counter() > deadline:
                raise _BuildTimeout(num_states=len(states))

            state = states[state_hash]
            if state.all_complete(env):
                step_transitions[state_hash].add(state_hash)
                step_predecessors[state_hash].add(state_hash)
                next_layer_set.add(state_hash)
                continue

            for next_state in _expand_disturbances(state, env, policy):
                next_hash = next_state.to_hashable()
                if next_hash not in states:
                    states[next_hash] = next_state
                step_transitions[state_hash].add(next_hash)
                step_predecessors[next_hash].add(state_hash)
                next_layer_set.add(next_hash)

        layers.append(sorted(next_layer_set, key=str))
        transitions.append(step_transitions)
        predecessors.append(step_predecessors)

    return _BoundedGraph(
        bound=bound,
        layers=layers,
        transitions=transitions,
        predecessors=predecessors,
        states=states,
        initial_hash=initial_hash,
    )


def _state_var(vpool: IDPool, t: int, state_hash: tuple) -> int:
    return vpool.id(("s", t, state_hash))


def _trajectory_from_model(graph: _BoundedGraph, vpool: IDPool, model: list[int]) -> list[State]:
    positives = {lit for lit in model if lit > 0}
    trajectory: list[State] = []

    for t in range(graph.bound + 1):
        chosen_hash = None
        for state_hash in graph.layers[t]:
            if _state_var(vpool, t, state_hash) in positives:
                chosen_hash = state_hash
                break
        if chosen_hash is None:
            raise RuntimeError(f"No chosen state found at timestep {t}")
        trajectory.append(graph.states[chosen_hash])

    return trajectory


def _add_exactly_one_constraints(cnf: CNF, vpool: IDPool, graph: _BoundedGraph) -> None:
    for t, layer in enumerate(graph.layers):
        lits = [_state_var(vpool, t, state_hash) for state_hash in layer]
        cnf.append(lits)  # at least one
        for clause in CardEnc.atmost(lits=lits, bound=1, vpool=vpool).clauses:
            cnf.append(clause)


def _add_transition_constraints(cnf: CNF, vpool: IDPool, graph: _BoundedGraph) -> None:
    for t in range(graph.bound):
        step_transitions = graph.transitions[t]
        step_predecessors = graph.predecessors[t]

        for src_hash in graph.layers[t]:
            src_var = _state_var(vpool, t, src_hash)
            dst_hashes = step_transitions.get(src_hash, set())
            if not dst_hashes:
                cnf.append([-src_var])
                continue
            dst_vars = [_state_var(vpool, t + 1, dst_hash) for dst_hash in dst_hashes]
            cnf.append([-src_var] + dst_vars)

        for dst_hash in graph.layers[t + 1]:
            pred_hashes = step_predecessors.get(dst_hash, set())
            dst_var = _state_var(vpool, t + 1, dst_hash)
            if not pred_hashes:
                cnf.append([-dst_var])
                continue
            pred_vars = [_state_var(vpool, t, src_hash) for src_hash in pred_hashes]
            cnf.append([-dst_var] + pred_vars)


def _resource_overload_clause(graph: _BoundedGraph, env: Environment, vpool: IDPool) -> list[int]:
    lits: list[int] = []
    for t, layer in enumerate(graph.layers):
        for state_hash in layer:
            if _resource_overload(graph.states[state_hash], env):
                lits.append(_state_var(vpool, t, state_hash))
    return lits


def _precedence_violation_clause(graph: _BoundedGraph, env: Environment, vpool: IDPool) -> list[int]:
    lits: list[int] = []
    for t, layer in enumerate(graph.layers):
        for state_hash in layer:
            if _precedence_violation(graph.states[state_hash], env):
                lits.append(_state_var(vpool, t, state_hash))
    return lits


def _task_completion_clause(graph: _BoundedGraph, env: Environment, vpool: IDPool) -> list[int]:
    t = graph.bound
    lits: list[int] = []
    for state_hash in graph.layers[t]:
        state = graph.states[state_hash]
        if not _task_complete(state, env):
            lits.append(_state_var(vpool, t, state_hash))
    return lits


def _encode_no_abandonment_violation(cnf: CNF, vpool: IDPool, graph: _BoundedGraph, env: Environment) -> list[int]:
    bound = graph.bound
    task_ids = [task.id for task in env.tasks]

    for t, layer in enumerate(graph.layers):
        for tid in task_ids:
            active_var = vpool.id(("active", t, tid))
            for state_hash in layer:
                s_var = _state_var(vpool, t, state_hash)
                state = graph.states[state_hash]
                if _active(state, tid):
                    cnf.append([-s_var, active_var])
                else:
                    cnf.append([-s_var, -active_var])

            seen_var = vpool.id(("seen", t, tid))
            if t == 0:
                cnf.append([-seen_var, active_var])
                cnf.append([-active_var, seen_var])
            else:
                prev_seen_var = vpool.id(("seen", t - 1, tid))
                cnf.append([-seen_var, prev_seen_var, active_var])
                cnf.append([-prev_seen_var, seen_var])
                cnf.append([-active_var, seen_var])

    violation_vars: list[int] = []
    for tid in task_ids:
        complete_var = vpool.id(("complete", bound, tid))
        for state_hash in graph.layers[bound]:
            s_var = _state_var(vpool, bound, state_hash)
            state = graph.states[state_hash]
            if _complete(state, tid):
                cnf.append([-s_var, complete_var])
            else:
                cnf.append([-s_var, -complete_var])

        seen_var = vpool.id(("seen", bound, tid))
        violation_var = vpool.id(("abandon", tid))
        violation_vars.append(violation_var)

        # violation_var <-> seen_var and (not complete_var)
        cnf.append([-violation_var, seen_var])
        cnf.append([-violation_var, -complete_var])
        cnf.append([-seen_var, complete_var, violation_var])

    return violation_vars


def _sat_check_property(
    graph: _BoundedGraph,
    env: Environment,
    property_name: str,
    deadline: float | None,
) -> tuple[bool, list[State] | None]:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("SAT property check timed out")

    vpool = IDPool()
    cnf = CNF()

    _add_exactly_one_constraints(cnf, vpool, graph)
    _add_transition_constraints(cnf, vpool, graph)

    cnf.append([_state_var(vpool, 0, graph.initial_hash)])

    if property_name == "resource_overload":
        violation_lits = _resource_overload_clause(graph, env, vpool)
    elif property_name == "precedence_violation":
        violation_lits = _precedence_violation_clause(graph, env, vpool)
    elif property_name == "task_completion":
        violation_lits = _task_completion_clause(graph, env, vpool)
    elif property_name == "no_abandonment":
        violation_lits = _encode_no_abandonment_violation(cnf, vpool, graph, env)
    else:
        raise ValueError(f"Unknown property: {property_name}")

    if not violation_lits:
        return False, None

    cnf.append(violation_lits)

    with Solver(name="g3", bootstrap_with=cnf.clauses) as solver:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("SAT solve timed out")
        sat = solver.solve()
        if not sat:
            return False, None
        model = solver.get_model()
        if model is None:
            return False, None
        return True, _trajectory_from_model(graph, vpool, model)


def sat_bounded_reachability(
    env: Environment,
    policy: Policy,
    bound: int | None = None,
    timeout_sec: float | None = None,
) -> SATReachabilityResult:
    start = time.perf_counter()
    deadline = start + timeout_sec if timeout_sec is not None else None
    bmc_bound = env.horizon if bound is None else bound
    property_results: dict[str, PropertySATResult] = {}
    first_violation_property: str | None = None
    first_counterexample: list[State] | None = None
    timed_out = False

    try:
        graph = _build_bounded_graph_with_deadline(
            env=env,
            policy=policy,
            bound=bmc_bound,
            deadline=deadline,
        )
    except _BuildTimeout as err:
        for property_name in PROPERTY_ORDER:
            property_results[property_name] = PropertySATResult(
                status="timeout",
                counterexample_trajectory=None,
            )
        return SATReachabilityResult(
            violation_found=False,
            violating_property="timeout",
            counterexample_trajectory=None,
            property_results=property_results,
            num_states_explored=err.num_states,
            bound=bmc_bound,
            runtime_sec=time.perf_counter() - start,
            timed_out=True,
        )

    for property_name in PROPERTY_ORDER:
        try:
            violation_found, trajectory = _sat_check_property(
                graph=graph,
                env=env,
                property_name=property_name,
                deadline=deadline,
            )
        except TimeoutError:
            timed_out = True
            property_results[property_name] = PropertySATResult(
                status="timeout",
                counterexample_trajectory=None,
            )
            continue

        if violation_found:
            property_results[property_name] = PropertySATResult(
                status="unsafe",
                counterexample_trajectory=trajectory,
            )
            if first_violation_property is None:
                first_violation_property = property_name
                first_counterexample = trajectory
        else:
            property_results[property_name] = PropertySATResult(
                status="safe",
                counterexample_trajectory=None,
            )

    if timed_out:
        for property_name in PROPERTY_ORDER:
            if property_name not in property_results:
                property_results[property_name] = PropertySATResult(
                    status="timeout",
                    counterexample_trajectory=None,
                )

    violation_found = first_violation_property is not None
    violating_property = first_violation_property if first_violation_property else ("timeout" if timed_out else "none")
    return SATReachabilityResult(
        violation_found=violation_found,
        violating_property=violating_property,
        counterexample_trajectory=first_counterexample,
        property_results=property_results,
        num_states_explored=len(graph.states),
        bound=bmc_bound,
        runtime_sec=time.perf_counter() - start,
        timed_out=timed_out,
    )

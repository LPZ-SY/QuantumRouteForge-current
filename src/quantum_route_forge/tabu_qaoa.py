from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Optional

from .geometry import cycle_distance, euclidean
from .maxcut_qaoa import (
    QAOACircuit,
    SparseMaxCutProblem,
    bitstring_to_bits,
    build_qaoa_openqasm,
    build_sparse_maxcut,
    solve_maxcut_exact,
)
from .models import Customer, Point
from .routing import nearest_neighbor


@dataclass(frozen=True)
class QuantumSample:
    bitstring: str
    source: str
    task_id: Optional[str] = None
    backend: Optional[str] = None
    endpoint: Optional[str] = None
    message: str = ""
    counts: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class TabuIterationRecord:
    iteration: int
    vehicle_pair: tuple[int, int]
    selected_customer_ids: list[int]
    sub_k: int
    n_edges: int
    sub_cnot: int
    bitstring: str
    sampler_source: str
    accepted: bool
    cut_before: float
    cut_proposed: float
    cut_after: float
    classical_bitstring: str
    classical_cut: float
    proxy_before: float
    proxy_after: float
    candidate_distance: float
    classical_candidate_distance: float
    quantum_improvement: float
    classical_improvement: float
    paired_improvement_delta: float
    best_distance: float
    task_id: Optional[str] = None
    backend: Optional[str] = None
    message: str = ""
    endpoint: Optional[str] = None
    counts: Optional[dict[str, int]] = None
    qasm: str = ""
    edges: Optional[list[tuple[int, int, float]]] = None
    cnot_count: int = 0


@dataclass(frozen=True)
class TabuQAOAResult:
    assignments: dict[int, list[Customer]]
    initial_distance: float
    best_distance: float
    records: list[TabuIterationRecord]

    @property
    def quantum_task_ids(self) -> list[str]:
        return [
            record.task_id
            for record in self.records
            if record.task_id
        ]


QuantumSampler = Callable[
    [SparseMaxCutProblem, QAOACircuit, int],
    QuantumSample,
]


def _copy_assignments(
    assignments: dict[int, list[Customer]],
) -> dict[int, list[Customer]]:
    return {
        vehicle: list(customers)
        for vehicle, customers in assignments.items()
    }


def _route_proxy_distance(customers: list[Customer], depot: Point) -> float:
    route = nearest_neighbor(depot, customers)
    return cycle_distance([customer.point for customer in route], depot=depot)


def assignment_distance(
    assignments: dict[int, list[Customer]],
    depot: Point,
) -> float:
    return sum(
        _route_proxy_distance(customers, depot)
        for customers in assignments.values()
    )


def _center(customers: list[Customer], fallback: Point) -> Point:
    if not customers:
        return fallback
    return (
        sum(customer.x for customer in customers) / len(customers),
        sum(customer.y for customer in customers) / len(customers),
    )


def _vehicle_pairs(
    assignments: dict[int, list[Customer]],
    depot: Point,
) -> list[tuple[int, int]]:
    vehicles = sorted(assignments)
    centers = {
        vehicle: _center(assignments[vehicle], fallback=depot)
        for vehicle in vehicles
    }
    pairs = [
        (left, right)
        for index, left in enumerate(vehicles)
        for right in vehicles[index + 1 :]
    ]
    pairs.sort(
        key=lambda pair: (
            euclidean(centers[pair[0]], centers[pair[1]]),
            pair,
        )
    )
    return pairs


def _select_boundary_customers(
    assignments: dict[int, list[Customer]],
    vehicle_pair: tuple[int, int],
    depot: Point,
    subproblem_size: int,
    rng: random.Random,
) -> list[Customer]:
    left, right = vehicle_pair
    left_customers = assignments[left]
    right_customers = assignments[right]
    if not left_customers or not right_customers:
        combined = left_customers + right_customers
        return combined[:subproblem_size]

    left_center = _center(left_customers, fallback=depot)
    right_center = _center(right_customers, fallback=depot)
    per_side = max(1, subproblem_size // 2)

    left_ranked = sorted(
        left_customers,
        key=lambda customer: (
            euclidean(customer.point, right_center)
            - euclidean(customer.point, left_center),
            customer.customer_id,
        ),
    )
    right_ranked = sorted(
        right_customers,
        key=lambda customer: (
            euclidean(customer.point, left_center)
            - euclidean(customer.point, right_center),
            customer.customer_id,
        ),
    )
    selected = left_ranked[:per_side] + right_ranked[:per_side]

    remaining = [
        customer
        for customer in left_customers + right_customers
        if customer not in selected
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, subproblem_size - len(selected))])
    return selected[:subproblem_size]


def _candidate_from_bits(
    assignments: dict[int, list[Customer]],
    selected: list[Customer],
    vehicle_pair: tuple[int, int],
    bits: list[int],
    reverse_partition: bool,
) -> dict[int, list[Customer]]:
    candidate = _copy_assignments(assignments)
    selected_ids = {customer.customer_id for customer in selected}
    left, right = vehicle_pair
    candidate[left] = [
        customer
        for customer in candidate[left]
        if customer.customer_id not in selected_ids
    ]
    candidate[right] = [
        customer
        for customer in candidate[right]
        if customer.customer_id not in selected_ids
    ]
    for customer, bit in zip(selected, bits):
        effective_bit = 1 - bit if reverse_partition else bit
        target = left if effective_bit == 0 else right
        candidate[target].append(customer)
    return candidate


def _is_capacity_feasible(
    assignments: dict[int, list[Customer]],
    capacity: int,
) -> bool:
    return all(
        sum(customer.demand for customer in customers) <= capacity
        for customers in assignments.values()
    )


def _move_signature(
    before: dict[int, list[Customer]],
    after: dict[int, list[Customer]],
) -> frozenset[tuple[int, int]]:
    before_vehicle = {
        customer.customer_id: vehicle
        for vehicle, customers in before.items()
        for customer in customers
    }
    return frozenset(
        (customer.customer_id, vehicle)
        for vehicle, customers in after.items()
        for customer in customers
        if before_vehicle.get(customer.customer_id) != vehicle
    )


def _weighted_cut_from_bits(
    problem: SparseMaxCutProblem,
    bits: list[int],
) -> float:
    return sum(
        edge.weight
        for edge in problem.edges
        if bits[edge.left] != bits[edge.right]
    )


def _local_exact_sampler(
    problem: SparseMaxCutProblem,
    _circuit: QAOACircuit,
    _iteration: int,
) -> QuantumSample:
    return QuantumSample(
        bitstring=solve_maxcut_exact(problem),
        source="local_exact_maxcut",
        message="Used deterministic exact Max-Cut fallback.",
    )


def run_tabu_qaoa(
    assignments: dict[int, list[Customer]],
    depot: Point,
    vehicle_capacity: int,
    iterations: int = 20,
    subproblem_size: int = 6,
    max_edges: int = 10,
    gamma: float = 1.1,
    beta: float = 0.8,
    tabu_tenure: int = 5,
    seed: int = 2026,
    sampler: QuantumSampler | None = None,
) -> TabuQAOAResult:
    """Refine assignments with hardware-safe Max-Cut proposals and Tabu control."""
    if subproblem_size < 2 or subproblem_size > 6:
        raise ValueError("subproblem_size must be between 2 and 6.")
    if max_edges < 1 or max_edges > 10:
        raise ValueError("max_edges must be between 1 and 10.")

    current = _copy_assignments(assignments)
    best = _copy_assignments(assignments)
    current_distance = assignment_distance(current, depot)
    initial_distance = current_distance
    best_distance = current_distance
    sampler = sampler or _local_exact_sampler
    rng = random.Random(seed)
    tabu_until: dict[tuple[int, int], int] = {}
    records: list[TabuIterationRecord] = []

    for iteration in range(max(0, int(iterations))):
        pairs = [
            pair
            for pair in _vehicle_pairs(current, depot)
            if (
                len(current[pair[0]])
                + len(current[pair[1]])
                >= 2
            )
        ]
        if not pairs:
            break
        vehicle_pair = pairs[iteration % len(pairs)]
        selected = _select_boundary_customers(
            assignments=current,
            vehicle_pair=vehicle_pair,
            depot=depot,
            subproblem_size=subproblem_size,
            rng=rng,
        )
        if len(selected) < 2:
            raise RuntimeError(
                "Selected vehicle pair cannot form a two-customer Max-Cut "
                "subproblem."
            )

        problem = build_sparse_maxcut(selected, max_edges=max_edges)
        circuit = build_qaoa_openqasm(problem, gamma=gamma, beta=beta)
        left_vehicle, _right_vehicle = vehicle_pair
        before_bits = [
            0
            if any(
                current_customer.customer_id == customer.customer_id
                for current_customer in current[left_vehicle]
            )
            else 1
            for customer in selected
        ]
        cut_before = _weighted_cut_from_bits(problem, before_bits)
        proxy_before = current_distance
        try:
            sample = sampler(problem, circuit, iteration)
            bits = bitstring_to_bits(sample.bitstring, len(selected))
        except Exception as exc:
            sample = QuantumSample(
                bitstring=solve_maxcut_exact(problem),
                source="local_exact_after_sampler_error",
                message=f"Sampler failed: {type(exc).__name__}: {exc}",
            )
            bits = bitstring_to_bits(sample.bitstring, len(selected))
        cut_proposed = _weighted_cut_from_bits(problem, bits)
        classical_bitstring = solve_maxcut_exact(problem)
        classical_bits = bitstring_to_bits(
            classical_bitstring,
            len(selected),
        )
        classical_cut = _weighted_cut_from_bits(
            problem,
            classical_bits,
        )

        def feasible_candidates_for(
            partition_bits: list[int],
        ) -> list[
            tuple[
                float,
                dict[int, list[Customer]],
                frozenset[tuple[int, int]],
            ]
        ]:
            candidates = []
            for reverse_partition in (False, True):
                candidate = _candidate_from_bits(
                    assignments=current,
                    selected=selected,
                    vehicle_pair=vehicle_pair,
                    bits=partition_bits,
                    reverse_partition=reverse_partition,
                )
                if not _is_capacity_feasible(
                    candidate,
                    vehicle_capacity,
                ):
                    continue
                signature = _move_signature(current, candidate)
                if not signature:
                    continue
                candidate_distance = assignment_distance(
                    candidate,
                    depot,
                )
                is_tabu = any(
                    tabu_until.get(move, -1) > iteration
                    for move in signature
                )
                aspiration = (
                    candidate_distance + 1e-9 < best_distance
                )
                if not is_tabu or aspiration:
                    candidates.append(
                        (candidate_distance, candidate, signature)
                    )
            return candidates

        feasible_candidates = feasible_candidates_for(bits)
        classical_candidates = feasible_candidates_for(classical_bits)
        classical_candidate_distance = (
            min(candidate[0] for candidate in classical_candidates)
            if classical_candidates
            else proxy_before
        )

        accepted = False
        candidate_distance = current_distance
        if feasible_candidates:
            candidate_distance, candidate, signature = min(
                feasible_candidates,
                key=lambda item: item[0],
            )
            current = candidate
            current_distance = candidate_distance
            accepted = True
            for move in signature:
                tabu_until[move] = iteration + max(1, int(tabu_tenure))
            if current_distance + 1e-9 < best_distance:
                best = _copy_assignments(current)
                best_distance = current_distance
        cut_after = cut_proposed if accepted else cut_before
        quantum_improvement = proxy_before - candidate_distance
        classical_improvement = (
            proxy_before - classical_candidate_distance
        )

        records.append(
            TabuIterationRecord(
                iteration=iteration + 1,
                vehicle_pair=vehicle_pair,
                selected_customer_ids=[
                    customer.customer_id for customer in selected
                ],
                sub_k=problem.num_qubits,
                n_edges=len(problem.edges),
                sub_cnot=circuit.cnot_count,
                bitstring=sample.bitstring,
                sampler_source=sample.source,
                accepted=accepted,
                cut_before=cut_before,
                cut_proposed=cut_proposed,
                cut_after=cut_after,
                classical_bitstring=classical_bitstring,
                classical_cut=classical_cut,
                proxy_before=proxy_before,
                proxy_after=current_distance,
                candidate_distance=candidate_distance,
                classical_candidate_distance=(
                    classical_candidate_distance
                ),
                quantum_improvement=quantum_improvement,
                classical_improvement=classical_improvement,
                paired_improvement_delta=(
                    quantum_improvement - classical_improvement
                ),
                best_distance=best_distance,
                task_id=sample.task_id,
                backend=sample.backend,
                message=sample.message,
                endpoint=sample.endpoint,
                counts=sample.counts,
                qasm=circuit.qasm,
                edges=[
                    (edge.left, edge.right, edge.weight)
                    for edge in problem.edges
                ],
                cnot_count=circuit.cnot_count,
            )
        )

    return TabuQAOAResult(
        assignments=best,
        initial_distance=initial_distance,
        best_distance=best_distance,
        records=records,
    )

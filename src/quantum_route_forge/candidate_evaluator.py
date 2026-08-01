from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from .models import Customer, Point
from .routing import build_route_plans
from .sparse_proxy_qubo import (
    SparseProxyQUBO,
    assignment_for_bits,
    bitstring_to_bits,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    bitstring: str
    count: int
    probability: float
    proxy_energy: float
    feasible_before_repair: bool
    feasible_after_repair: bool
    repaired: bool
    repair_summary: str
    true_distance: float
    improvement: float
    assignment_customer_ids: dict[str, list[int]]

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateBatch:
    arm: str
    shots: int
    unique_candidates: int
    feasible_candidates: int
    evaluated_candidates: int
    top_frequency: tuple[CandidateEvaluation, ...]
    best_of_shots: CandidateEvaluation | None
    accepted: CandidateEvaluation | None
    raw_ranked: tuple[tuple[str, int], ...]
    distribution_entropy: float
    top_k_probability: float
    all_zero_probability: float
    all_one_probability: float

    def payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "shots": self.shots,
            "unique_candidates": self.unique_candidates,
            "feasible_candidates": self.feasible_candidates,
            "evaluated_candidates": self.evaluated_candidates,
            "top_frequency": [row.payload() for row in self.top_frequency],
            "best_of_shots": self.best_of_shots.payload() if self.best_of_shots else None,
            "accepted": self.accepted.payload() if self.accepted else None,
            "raw_ranked": [[bitstring, count] for bitstring, count in self.raw_ranked],
            "distribution_entropy": self.distribution_entropy,
            "top_k_probability": self.top_k_probability,
            "all_zero_probability": self.all_zero_probability,
            "all_one_probability": self.all_one_probability,
        }


def normalize_counts(counts: Mapping[object, object], width: int) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_bitstring, raw_count in (counts or {}).items():
        cleaned = str(raw_bitstring or "").replace(" ", "").strip()
        if not cleaned or any(bit not in "01" for bit in cleaned):
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        canonical = cleaned[-width:].zfill(width)
        normalized[canonical] = normalized.get(canonical, 0) + count
    return normalized


def _loads(assignments: Mapping[int, Sequence[Customer]]) -> dict[int, int]:
    return {
        vehicle: sum(customer.demand for customer in customers)
        for vehicle, customers in assignments.items()
    }


def _capacity_feasible(assignments: Mapping[int, Sequence[Customer]], capacity: int) -> bool:
    return all(load <= capacity for load in _loads(assignments).values())


def repair_capacity(
    assignments: Mapping[int, Sequence[Customer]],
    capacity: int,
    max_iterations: int = 2_000,
) -> tuple[dict[int, list[Customer]], bool, str]:
    """Apply the repository's deterministic least-loaded-route repair policy."""
    repaired = {vehicle: list(customers) for vehicle, customers in assignments.items()}
    changed = False
    for _ in range(max(1, int(max_iterations))):
        loads = _loads(repaired)
        overloaded = [vehicle for vehicle, load in loads.items() if load > capacity]
        if not overloaded:
            return repaired, changed, "capacity_repaired" if changed else "no_repair_needed"
        moved = False
        for source in sorted(overloaded, key=lambda vehicle: (-loads[vehicle], vehicle)):
            for customer in sorted(repaired[source], key=lambda row: (-row.demand, row.customer_id)):
                targets = [
                    vehicle
                    for vehicle in repaired
                    if vehicle != source and loads[vehicle] + customer.demand <= capacity
                ]
                if not targets:
                    continue
                target = min(targets, key=lambda vehicle: (loads[vehicle], vehicle))
                repaired[source].remove(customer)
                repaired[target].append(customer)
                changed = moved = True
                break
            if moved:
                break
        if not moved:
            return repaired, changed, "repair_attempted_but_stuck"
    return repaired, changed, "repair_iteration_limit_reached"


def true_route_distance(
    assignments: Mapping[int, Sequence[Customer]],
    depot: Point,
    routing_method: str = "heuristic",
    two_opt_rounds: int = 2,
    ortools_time_limit_sec: int = 1,
) -> float:
    routes = build_route_plans(
        assignments={vehicle: list(customers) for vehicle, customers in assignments.items()},
        depot=depot,
        two_opt_rounds=two_opt_rounds,
        routing_method=routing_method,
        ortools_time_limit_sec=ortools_time_limit_sec,
        ortools_fallback=True,
    )
    return float(sum(route.distance for route in routes))


def _serialize(assignments: Mapping[int, Sequence[Customer]]) -> dict[str, list[int]]:
    return {
        str(vehicle): [customer.customer_id for customer in customers]
        for vehicle, customers in sorted(assignments.items())
    }


def evaluate_bitstring(
    *,
    bitstring: str,
    count: int,
    shots: int,
    proxy: SparseProxyQUBO,
    assignments: Mapping[int, Sequence[Customer]],
    block_customers: Sequence[Customer],
    depot: Point,
    vehicle_capacity: int,
    baseline_distance: float,
    routing_method: str = "heuristic",
) -> tuple[CandidateEvaluation, dict[int, list[Customer]]]:
    bits = bitstring_to_bits(bitstring, proxy.width)
    candidate = assignment_for_bits(assignments, block_customers, proxy.vehicle_pair, bits)
    feasible_before = _capacity_feasible(candidate, vehicle_capacity)
    repaired, changed, repair_summary = repair_capacity(candidate, vehicle_capacity)
    feasible_after = _capacity_feasible(repaired, vehicle_capacity)
    distance = (
        true_route_distance(repaired, depot, routing_method=routing_method)
        if feasible_after
        else math.inf
    )
    row = CandidateEvaluation(
        bitstring=bitstring,
        count=int(count),
        probability=float(count) / max(1, int(shots)),
        proxy_energy=proxy.energy(bits),
        feasible_before_repair=feasible_before,
        feasible_after_repair=feasible_after,
        repaired=changed,
        repair_summary=repair_summary,
        true_distance=float(distance),
        improvement=float(baseline_distance - distance),
        assignment_customer_ids=_serialize(repaired),
    )
    return row, repaired


def evaluate_counts(
    *,
    arm: str,
    counts: Mapping[object, object],
    proxy: SparseProxyQUBO,
    assignments: Mapping[int, Sequence[Customer]],
    block_customers: Sequence[Customer],
    depot: Point,
    vehicle_capacity: int,
    candidate_k: int = 8,
    filter_extremes: bool = False,
    routing_method: str = "heuristic",
) -> tuple[CandidateBatch, dict[int, list[Customer]] | None]:
    """Apply one fixed budget and one full-objective policy to every arm."""
    normalized = normalize_counts(counts, proxy.width)
    if not normalized:
        raise ValueError("counts are empty after normalization")
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    shots = sum(normalized.values())
    baseline = true_route_distance(assignments, depot, routing_method=routing_method)
    extremes = {"0" * proxy.width, "1" * proxy.width}
    eligible = [row for row in ranked if not filter_extremes or row[0] not in extremes]
    cache: dict[str, tuple[CandidateEvaluation, dict[int, list[Customer]]]] = {}

    def evaluated(bitstring: str, count: int):
        if bitstring not in cache:
            cache[bitstring] = evaluate_bitstring(
                bitstring=bitstring,
                count=count,
                shots=shots,
                proxy=proxy,
                assignments=assignments,
                block_customers=block_customers,
                depot=depot,
                vehicle_capacity=vehicle_capacity,
                baseline_distance=baseline,
                routing_method=routing_method,
            )
        return cache[bitstring]

    top_rows: list[CandidateEvaluation] = []
    top_assignments: dict[str, dict[int, list[Customer]]] = {}
    for bitstring, count in eligible[: max(0, int(candidate_k))]:
        row, candidate_assignment = evaluated(bitstring, count)
        top_rows.append(row)
        top_assignments[bitstring] = candidate_assignment

    # Diagnostic only: evaluate all strings sampled by this arm.  It does not
    # increase the fixed downstream acceptance budget above candidate_k.
    all_rows = [evaluated(bitstring, count)[0] for bitstring, count in eligible]
    feasible_rows = [row for row in all_rows if row.feasible_after_repair]
    best_of_shots = min(feasible_rows, key=lambda row: (row.true_distance, row.bitstring), default=None)
    accepted = min(
        (row for row in top_rows if row.feasible_after_repair and row.improvement > 1e-9),
        key=lambda row: (row.true_distance, row.bitstring),
        default=None,
    )
    accepted_assignment = top_assignments.get(accepted.bitstring) if accepted else None
    probabilities = [count / shots for count in normalized.values()]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability > 0.0)

    batch = CandidateBatch(
        arm=arm,
        shots=shots,
        unique_candidates=len(normalized),
        feasible_candidates=len(feasible_rows),
        evaluated_candidates=len(top_rows),
        top_frequency=tuple(top_rows),
        best_of_shots=best_of_shots,
        accepted=accepted,
        raw_ranked=tuple(ranked),
        distribution_entropy=float(entropy),
        top_k_probability=sum(row.count for row in top_rows) / shots,
        all_zero_probability=normalized.get("0" * proxy.width, 0) / shots,
        all_one_probability=normalized.get("1" * proxy.width, 0) / shots,
    )
    return batch, accepted_assignment

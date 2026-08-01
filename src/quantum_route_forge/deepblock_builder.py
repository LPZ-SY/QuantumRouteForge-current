from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence

from .geometry import euclidean
from .models import Customer, Point


@dataclass(frozen=True)
class BoundaryCandidate:
    customer_id: int
    vehicle_id: int
    score: float
    reason: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeepBlock:
    block_id: str
    customer_ids: tuple[int, ...]
    vehicle_pair: tuple[int, int]
    overlap_with_previous: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        return len(self.customer_ids)

    def payload(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "customer_ids": list(self.customer_ids),
            "vehicle_pair": list(self.vehicle_pair),
            "overlap_with_previous": list(self.overlap_with_previous),
            "width": self.width,
        }


def _center(customers: Sequence[Customer], fallback: Point) -> Point:
    if not customers:
        return fallback
    return (
        sum(customer.x for customer in customers) / len(customers),
        sum(customer.y for customer in customers) / len(customers),
    )


def rank_vehicle_pairs(
    assignments: Mapping[int, Sequence[Customer]],
    depot: Point,
) -> list[tuple[int, int]]:
    """Rank pairs whose route regions are closest and can form a local move."""
    centers = {
        vehicle: _center(list(customers), fallback=depot)
        for vehicle, customers in assignments.items()
    }
    pairs = [
        (left, right)
        for left, right in combinations(sorted(assignments), 2)
        if assignments[left] or assignments[right]
    ]
    return sorted(
        pairs,
        key=lambda pair: (
            euclidean(centers[pair[0]], centers[pair[1]]),
            pair,
        ),
    )


def select_boundary_pool(
    assignments: Mapping[int, Sequence[Customer]],
    vehicle_pair: tuple[int, int],
    depot: Point,
    pool_size: int = 16,
) -> list[BoundaryCandidate]:
    """Select real boundary customers for one two-vehicle reassignment pool.

    The score is deliberately interpretable: customers close to the other
    route centre, close to the geometric boundary, and carrying more demand
    receive higher priority.  Every returned item records that rationale.
    """
    left, right = vehicle_pair
    left_customers = list(assignments.get(left, ()))
    right_customers = list(assignments.get(right, ()))
    left_center = _center(left_customers, fallback=depot)
    right_center = _center(right_customers, fallback=depot)
    max_demand = max(
        (customer.demand for customer in left_customers + right_customers),
        default=1,
    )
    ranked: list[BoundaryCandidate] = []
    for vehicle, customers, own_center, other_center in (
        (left, left_customers, left_center, right_center),
        (right, right_customers, right_center, left_center),
    ):
        for customer in customers:
            own_distance = euclidean(customer.point, own_center)
            other_distance = euclidean(customer.point, other_center)
            boundary_gap = abs(other_distance - own_distance)
            boundary_score = 1.0 / (1.0 + boundary_gap)
            cross_route_score = 1.0 / (1.0 + other_distance)
            demand_score = customer.demand / max_demand
            score = 0.55 * boundary_score + 0.30 * cross_route_score + 0.15 * demand_score
            ranked.append(
                BoundaryCandidate(
                    customer_id=customer.customer_id,
                    vehicle_id=vehicle,
                    score=float(score),
                    reason=(
                        "boundary_gap="
                        f"{boundary_gap:.6g}; distance_to_other_center={other_distance:.6g}; "
                        f"demand={customer.demand}"
                    ),
                )
            )
    ranked.sort(key=lambda item: (-item.score, item.customer_id))
    return ranked[: max(0, int(pool_size))]


def build_interaction_graph(
    customers: Iterable[Customer],
    candidate_scores: Mapping[int, float] | None = None,
) -> dict[tuple[int, int], float]:
    """Create a complete, weighted customer interaction graph for blocking."""
    rows = list(customers)
    candidate_scores = candidate_scores or {}
    max_demand = max((customer.demand for customer in rows), default=1)
    interactions: dict[tuple[int, int], float] = {}
    for left, right in combinations(rows, 2):
        distance_affinity = 1.0 / (1.0 + euclidean(left.point, right.point))
        demand_affinity = 1.0 - abs(left.demand - right.demand) / max_demand
        priority = 0.5 * (
            float(candidate_scores.get(left.customer_id, 0.0))
            + float(candidate_scores.get(right.customer_id, 0.0))
        )
        weight = 0.60 * distance_affinity + 0.20 * demand_affinity + 0.20 * priority
        key = tuple(sorted((left.customer_id, right.customer_id)))
        interactions[key] = float(weight)
    return interactions


def _interaction(
    left: int,
    right: int,
    interactions: Mapping[tuple[int, int], float],
) -> float:
    if left == right:
        return 0.0
    return float(interactions.get(tuple(sorted((left, right))), 0.0))


def _grow_group(
    seeds: Sequence[int],
    available: Sequence[int],
    target_size: int,
    interactions: Mapping[tuple[int, int], float],
    rank: Mapping[int, int],
) -> list[int]:
    group = list(dict.fromkeys(seeds))
    remaining = [
        customer_id
        for customer_id in dict.fromkeys(available)
        if customer_id not in group
    ]
    while remaining and len(group) < target_size:
        chosen = max(
            remaining,
            key=lambda customer_id: (
                sum(_interaction(customer_id, member, interactions) for member in group),
                -rank.get(customer_id, math.inf),
                -customer_id,
            ),
        )
        group.append(chosen)
        remaining.remove(chosen)
    return group


def build_overlapping_blocks(
    candidate_ids: Sequence[int],
    vehicle_pair: tuple[int, int],
    interactions: Mapping[tuple[int, int], float],
    block_size: int = 8,
    overlap: int = 3,
    max_blocks: int = 3,
) -> list[DeepBlock]:
    """Cover a ranked pool with 1--3 interaction-aware overlapping blocks."""
    ordered = list(dict.fromkeys(int(customer_id) for customer_id in candidate_ids))
    if not ordered:
        return []
    block_size = max(1, int(block_size))
    overlap = max(0, min(int(overlap), block_size - 1))
    max_blocks = max(1, int(max_blocks))
    rank = {customer_id: index for index, customer_id in enumerate(ordered)}

    first = _grow_group(
        seeds=ordered[:1],
        available=ordered,
        target_size=min(block_size, len(ordered)),
        interactions=interactions,
        rank=rank,
    )
    blocks: list[list[int]] = [first]
    covered = set(first)

    while len(blocks) < max_blocks and len(covered) < len(ordered):
        previous = blocks[-1]
        uncovered = [customer_id for customer_id in ordered if customer_id not in covered]
        overlap_nodes = sorted(
            previous,
            key=lambda customer_id: (
                -sum(_interaction(customer_id, other, interactions) for other in uncovered),
                rank[customer_id],
            ),
        )[: min(overlap, len(previous))]
        new_nodes = _grow_group(
            seeds=uncovered[:1],
            available=uncovered,
            target_size=min(len(uncovered), block_size - len(overlap_nodes)),
            interactions=interactions,
            rank=rank,
        )
        seeds = overlap_nodes + new_nodes
        group = _grow_group(
            seeds=seeds,
            available=previous + ordered,
            target_size=min(block_size, len(ordered)),
            interactions=interactions,
            rank=rank,
        )
        blocks.append(group)
        covered.update(group)

    # If a strict max_blocks budget leaves customers uncovered, replace the
    # weakest repeated positions in the final block with the missing customers.
    missing = [customer_id for customer_id in ordered if customer_id not in covered]
    if missing:
        last = blocks[-1]
        previous = set(blocks[-2]) if len(blocks) > 1 else set()
        replaceable = [
            index
            for index, customer_id in enumerate(last)
            if customer_id in previous and customer_id not in missing
        ]
        for customer_id, index in zip(missing, reversed(replaceable)):
            last[index] = customer_id

    result: list[DeepBlock] = []
    previous: set[int] = set()
    for index, group in enumerate(blocks, start=1):
        result.append(
            DeepBlock(
                block_id=f"B{index}",
                customer_ids=tuple(group),
                vehicle_pair=vehicle_pair,
                overlap_with_previous=tuple(customer_id for customer_id in group if customer_id in previous),
            )
        )
        previous = set(group)
    return result


def scan_sequence(blocks: Sequence[DeepBlock]) -> list[DeepBlock]:
    """Return the required forward then reverse block sequence."""
    return list(blocks) + list(reversed(blocks))

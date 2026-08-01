from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Sequence

from .geometry import cycle_distance
from .models import Customer, Point
from .routing import nearest_neighbor


@dataclass(frozen=True)
class ProxyInteraction:
    left: int
    right: int
    coefficient: float
    kept: bool
    reason: str


@dataclass(frozen=True)
class SparseProxyQUBO:
    customer_ids: tuple[int, ...]
    vehicle_pair: tuple[int, int]
    linear: tuple[float, ...]
    quadratic: tuple[ProxyInteraction, ...]
    constant: float
    current_bitstring: str
    scale: float

    @property
    def width(self) -> int:
        return len(self.customer_ids)

    @property
    def kept_interactions(self) -> tuple[ProxyInteraction, ...]:
        return tuple(row for row in self.quadratic if row.kept)

    @property
    def pruned_interactions(self) -> tuple[ProxyInteraction, ...]:
        return tuple(row for row in self.quadratic if not row.kept)

    @property
    def logical_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((row.left, row.right) for row in self.kept_interactions)

    def energy(self, bits: Sequence[int]) -> float:
        if len(bits) != self.width:
            raise ValueError(f"Expected {self.width} bits, received {len(bits)}.")
        values = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in values):
            raise ValueError("QUBO bits must be 0 or 1.")
        energy = self.constant + sum(coefficient * values[index] for index, coefficient in enumerate(self.linear))
        energy += sum(
            row.coefficient * values[row.left] * values[row.right]
            for row in self.kept_interactions
        )
        return float(energy)

    def payload(self) -> dict[str, object]:
        return {
            "customer_ids": list(self.customer_ids),
            "vehicle_pair": list(self.vehicle_pair),
            "linear": list(self.linear),
            "quadratic": [asdict(row) for row in self.quadratic],
            "constant": self.constant,
            "current_bitstring": self.current_bitstring,
            "scale": self.scale,
            "logical_edges": [list(edge) for edge in self.logical_edges],
        }


def bitstring_to_bits(bitstring: str, width: int) -> tuple[int, ...]:
    cleaned = str(bitstring or "").replace(" ", "").strip()
    if not cleaned or any(bit not in "01" for bit in cleaned):
        raise ValueError("bitstring must contain only 0 and 1")
    canonical = cleaned[-width:].zfill(width)
    # OpenQASM count keys print the highest classical bit on the left.
    return tuple(int(bit) for bit in reversed(canonical))


def bits_to_bitstring(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in reversed(tuple(bits)))


def assignment_for_bits(
    assignments: Mapping[int, Sequence[Customer]],
    customers: Sequence[Customer],
    vehicle_pair: tuple[int, int],
    bits: Sequence[int],
) -> dict[int, list[Customer]]:
    if len(customers) != len(bits):
        raise ValueError("customers and bits must have the same length")
    candidate = {vehicle: list(rows) for vehicle, rows in assignments.items()}
    selected_ids = {customer.customer_id for customer in customers}
    left, right = vehicle_pair
    candidate.setdefault(left, [])
    candidate.setdefault(right, [])
    candidate[left] = [row for row in candidate[left] if row.customer_id not in selected_ids]
    candidate[right] = [row for row in candidate[right] if row.customer_id not in selected_ids]
    for customer, bit in zip(customers, bits):
        candidate[left if int(bit) == 0 else right].append(customer)
    return candidate


def _proxy_distance(assignments: Mapping[int, Sequence[Customer]], depot: Point) -> float:
    total = 0.0
    for customers in assignments.values():
        route = nearest_neighbor(depot, customers)
        total += cycle_distance([customer.point for customer in route], depot=depot)
    return float(total)


def _capacity_penalty(
    assignments: Mapping[int, Sequence[Customer]],
    capacity: int,
    weight: float,
) -> float:
    return float(
        weight
        * sum(
            max(0, sum(customer.demand for customer in customers) - capacity) ** 2
            for customers in assignments.values()
        )
    )


def _score(
    assignments: Mapping[int, Sequence[Customer]],
    depot: Point,
    capacity: int,
    capacity_penalty: float,
) -> float:
    return _proxy_distance(assignments, depot) + _capacity_penalty(assignments, capacity, capacity_penalty)


def build_sparse_proxy_qubo(
    assignments: Mapping[int, Sequence[Customer]],
    block_customers: Sequence[Customer],
    vehicle_pair: tuple[int, int],
    depot: Point,
    vehicle_capacity: int,
    allowed_logical_edges: Sequence[tuple[int, int]],
    capacity_penalty: float = 25.0,
    coefficient_threshold: float = 1e-9,
    normalize: bool = True,
) -> SparseProxyQUBO:
    """Build an interpretable pairwise route proxy and prune it to hardware edges."""
    customers = list(block_customers)
    width = len(customers)
    if width < 1 or width > 8:
        raise ValueError("A Baihua proxy QUBO must contain between 1 and 8 customers.")
    by_vehicle = {
        customer.customer_id: vehicle
        for vehicle, rows in assignments.items()
        for customer in rows
    }
    left, right = vehicle_pair
    invalid = [customer.customer_id for customer in customers if by_vehicle.get(customer.customer_id) not in {left, right}]
    if invalid:
        raise ValueError(f"Block customers are not assigned to vehicle pair {vehicle_pair}: {invalid}")
    current_bits = tuple(0 if by_vehicle[customer.customer_id] == left else 1 for customer in customers)

    def score_with(overrides: Mapping[int, int]) -> float:
        bits = list(current_bits)
        for index, bit in overrides.items():
            bits[int(index)] = int(bit)
        candidate = assignment_for_bits(assignments, customers, vehicle_pair, bits)
        return _score(candidate, depot, vehicle_capacity, capacity_penalty)

    linear: list[float] = []
    for index in range(width):
        cost_zero = score_with({index: 0})
        cost_one = score_with({index: 1})
        linear.append(cost_one - cost_zero)

    allowed = {tuple(sorted((int(left_index), int(right_index)))) for left_index, right_index in allowed_logical_edges}
    quadratic: list[ProxyInteraction] = []
    for left_index, right_index in combinations(range(width), 2):
        cost00 = score_with({left_index: 0, right_index: 0})
        cost01 = score_with({left_index: 0, right_index: 1})
        cost10 = score_with({left_index: 1, right_index: 0})
        cost11 = score_with({left_index: 1, right_index: 1})
        coefficient = cost11 - cost10 - cost01 + cost00
        edge = (left_index, right_index)
        topology_allowed = edge in allowed
        strong_enough = abs(coefficient) > float(coefficient_threshold)
        kept = topology_allowed and strong_enough
        if not topology_allowed:
            reason = "pruned_not_supported_by_physical_subgraph"
        elif not strong_enough:
            reason = "pruned_below_coefficient_threshold"
        else:
            reason = "kept_direct_physical_coupling"
        quadratic.append(
            ProxyInteraction(
                left=left_index,
                right=right_index,
                coefficient=float(coefficient),
                kept=kept,
                reason=reason,
            )
        )

    scale = max(
        [abs(value) for value in linear]
        + [abs(row.coefficient) for row in quadratic if row.kept]
        + [1.0]
    )
    if normalize and scale > 1.0:
        linear = [value / scale for value in linear]
        quadratic = [
            ProxyInteraction(
                left=row.left,
                right=row.right,
                coefficient=row.coefficient / scale,
                kept=row.kept,
                reason=row.reason,
            )
            for row in quadratic
        ]
    else:
        scale = 1.0

    return SparseProxyQUBO(
        customer_ids=tuple(customer.customer_id for customer in customers),
        vehicle_pair=vehicle_pair,
        linear=tuple(float(value) for value in linear),
        quadratic=tuple(quadratic),
        constant=0.0,
        current_bitstring=bits_to_bitstring(current_bits),
        scale=float(scale),
    )

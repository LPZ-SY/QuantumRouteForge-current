from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional

import dimod

from .geometry import euclidean
from .models import Customer, DispatchInstance


def assignment_var(customer_id: int, vehicle_id: int) -> str:
    return f"x_c{customer_id}_v{vehicle_id}"


def build_assignment_bqm(
    instance: DispatchInstance,
    compactness_weight: float = 1.0,
    assignment_penalty: float = 90.0,
    capacity_penalty: float = 45.0,
    balance_weight: float = 1.2,
    preferred_vehicle_by_customer: Optional[Dict[int, int]] = None,
    preference_strength: float = 2.5,
) -> dimod.BinaryQuadraticModel:
    """Build a BQM for client-to-vehicle assignment."""
    if not instance.feasible_capacity:
        raise ValueError(
            "Total demand exceeds fleet capacity. Increase vehicles or vehicle capacity."
        )

    vehicles = range(instance.num_vehicles)
    customers = instance.customers
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")

    # Ensure all assignment variables exist.
    for c in customers:
        for v in vehicles:
            bqm.add_variable(assignment_var(c.customer_id, v), 0.0)

    # Compactness affinity: nearby customers are rewarded for sharing a vehicle.
    for c1, c2 in combinations(customers, 2):
        dist = euclidean(c1.point, c2.point)
        affinity = compactness_weight * (1.0 / (1.0 + dist))
        for v in vehicles:
            bqm.add_interaction(
                assignment_var(c1.customer_id, v),
                assignment_var(c2.customer_id, v),
                -affinity,
            )

    # Each customer must be assigned to exactly one vehicle.
    for c in customers:
        terms = [(assignment_var(c.customer_id, v), 1.0) for v in vehicles]
        bqm.add_linear_equality_constraint(
            terms=terms,
            lagrange_multiplier=assignment_penalty,
            constant=-1.0,
        )

    # Capacity constraints per vehicle: 0 <= sum(d_i * x_i_v) <= capacity
    for v in vehicles:
        terms = [(assignment_var(c.customer_id, v), int(c.demand)) for c in customers]
        bqm.add_linear_inequality_constraint(
            terms=terms,
            lagrange_multiplier=capacity_penalty,
            label=f"cap_v{v}",
            constant=0,
            lb=0,
            ub=int(instance.vehicle_capacity),
            cross_zero=False,
            penalization_method="slack",
        )

    # Load balancing around fleet-average target.
    target_load = instance.total_demand / instance.num_vehicles
    for v in vehicles:
        terms = [(assignment_var(c.customer_id, v), float(c.demand)) for c in customers]
        bqm.add_linear_equality_constraint(
            terms=terms,
            lagrange_multiplier=balance_weight,
            constant=-target_load,
        )

    # Optional soft hints from quantum partition seeds.
    if preferred_vehicle_by_customer:
        for customer in customers:
            preferred = preferred_vehicle_by_customer.get(customer.customer_id)
            if preferred is None:
                continue
            preferred = int(preferred) % instance.num_vehicles
            for v in vehicles:
                var = assignment_var(customer.customer_id, v)
                if v == preferred:
                    bqm.add_linear(var, -abs(preference_strength))
                else:
                    bqm.add_linear(var, abs(preference_strength) / max(1, instance.num_vehicles - 1))

    return bqm


def decode_assignment(
    sample: Dict[str, int],
    customers: Iterable[Customer],
    num_vehicles: int,
) -> Dict[int, List[Customer]]:
    assignments: Dict[int, List[Customer]] = {v: [] for v in range(num_vehicles)}
    for c in customers:
        best_v = 0
        best_val = -1
        for v in range(num_vehicles):
            val = int(sample.get(assignment_var(c.customer_id, v), 0))
            if val > best_val:
                best_val = val
                best_v = v
        assignments[best_v].append(c)
    return assignments

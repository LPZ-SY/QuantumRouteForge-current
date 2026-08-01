from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from .geometry import euclidean
from .models import Customer, DispatchInstance, Point


@dataclass(frozen=True)
class CapacityClusteringResult:
    assignments: dict[int, list[Customer]]
    centers: list[Point]
    loads: dict[int, int]
    method: str


def _mean_center(customers: Iterable[Customer], fallback: Point) -> Point:
    customers = list(customers)
    if not customers:
        return fallback
    return (
        sum(customer.x for customer in customers) / len(customers),
        sum(customer.y for customer in customers) / len(customers),
    )


def _initial_centers(
    customers: list[Customer],
    num_vehicles: int,
    seed: int,
    demand_weight: float,
) -> tuple[list[Point], str]:
    """Use sklearn K-means when available and a deterministic fallback otherwise."""
    try:
        from sklearn.cluster import KMeans

        max_demand = max(customer.demand for customer in customers)
        coord_scale = max(
            1.0,
            max(math.hypot(customer.x, customer.y) for customer in customers),
        )
        features = [
            [
                customer.x / coord_scale,
                customer.y / coord_scale,
                demand_weight * customer.demand / max_demand,
            ]
            for customer in customers
        ]
        model = KMeans(n_clusters=num_vehicles, random_state=seed, n_init=10)
        model.fit(features)

        grouped: dict[int, list[Customer]] = {vehicle: [] for vehicle in range(num_vehicles)}
        for customer, label in zip(customers, model.labels_):
            grouped[int(label)].append(customer)
        centers = [
            _mean_center(grouped[vehicle], fallback=(0.0, 0.0))
            for vehicle in range(num_vehicles)
        ]
        return centers, "sklearn_kmeans_capacity_assignment"
    except ImportError:
        pass

    rng = random.Random(seed)
    first = customers[rng.randrange(len(customers))]
    centers = [first.point]
    while len(centers) < num_vehicles:
        remaining = [
            customer
            for customer in customers
            if customer.point not in centers
        ]
        if not remaining:
            centers.append(customers[len(centers) % len(customers)].point)
            continue
        nxt = max(
            remaining,
            key=lambda customer: min(
                euclidean(customer.point, center) for center in centers
            ),
        )
        centers.append(nxt.point)
    return centers, "deterministic_farthest_first_capacity_assignment"


def _try_capacity_assignment(
    customers: list[Customer],
    centers: list[Point],
    capacity: int,
    order: list[Customer],
) -> dict[int, list[Customer]] | None:
    assignments: dict[int, list[Customer]] = {
        vehicle: [] for vehicle in range(len(centers))
    }
    loads = {vehicle: 0 for vehicle in assignments}
    target_load = sum(customer.demand for customer in customers) / len(centers)

    for customer in order:
        feasible = [
            vehicle
            for vehicle in assignments
            if loads[vehicle] + customer.demand <= capacity
        ]
        if not feasible:
            return None

        def score(vehicle: int) -> tuple[float, float, int]:
            spatial = euclidean(customer.point, centers[vehicle])
            projected = loads[vehicle] + customer.demand
            balance = abs(projected - target_load) / max(1.0, target_load)
            return spatial * (1.0 + 0.18 * balance), projected, vehicle

        vehicle = min(feasible, key=score)
        assignments[vehicle].append(customer)
        loads[vehicle] += customer.demand

    return assignments


def capacity_constrained_kmeans(
    instance: DispatchInstance,
    seed: int = 2026,
    demand_weight: float = 0.35,
    max_rounds: int = 8,
    assignment_restarts: int = 40,
) -> CapacityClusteringResult:
    """Build spatially compact vehicle groups while enforcing hard capacities.

    K-means supplies geometric centers only. Customer-to-center assignment is
    performed separately because ordinary K-means does not enforce vehicle
    capacity.
    """
    if not instance.feasible_capacity:
        raise ValueError("Total demand exceeds fleet capacity.")
    if not instance.customers:
        assignments = {
            vehicle: [] for vehicle in range(instance.num_vehicles)
        }
        return CapacityClusteringResult(
            assignments=assignments,
            centers=[instance.depot] * instance.num_vehicles,
            loads={vehicle: 0 for vehicle in assignments},
            method="empty_instance",
        )
    oversized = [
        customer.customer_id
        for customer in instance.customers
        if customer.demand > instance.vehicle_capacity
    ]
    if oversized:
        raise ValueError(
            "Individual customer demand exceeds vehicle capacity: "
            + ", ".join(str(customer_id) for customer_id in oversized)
        )

    customers = list(instance.customers)
    centers, method = _initial_centers(
        customers=customers,
        num_vehicles=instance.num_vehicles,
        seed=seed,
        demand_weight=demand_weight,
    )
    rng = random.Random(seed)
    assignments: dict[int, list[Customer]] | None = None

    for round_id in range(max(1, max_rounds)):
        orders: list[list[Customer]] = [
            sorted(
                customers,
                key=lambda customer: (
                    -customer.demand,
                    min(euclidean(customer.point, center) for center in centers),
                    customer.customer_id,
                ),
            ),
            sorted(
                customers,
                key=lambda customer: (
                    -customer.demand,
                    -min(euclidean(customer.point, center) for center in centers),
                    customer.customer_id,
                ),
            ),
        ]
        for _ in range(max(0, assignment_restarts - len(orders))):
            order = customers[:]
            rng.shuffle(order)
            order.sort(key=lambda customer: -customer.demand)
            orders.append(order)

        for order in orders:
            assignments = _try_capacity_assignment(
                customers=customers,
                centers=centers,
                capacity=instance.vehicle_capacity,
                order=order,
            )
            if assignments is not None:
                break
        if assignments is None:
            continue

        new_centers = [
            _mean_center(assignments[vehicle], fallback=centers[vehicle])
            for vehicle in range(instance.num_vehicles)
        ]
        shift = sum(
            euclidean(old_center, new_center)
            for old_center, new_center in zip(centers, new_centers)
        )
        centers = new_centers
        if shift <= 1e-8:
            break
        assignments = None if round_id + 1 < max_rounds else assignments

    if assignments is None:
        raise ValueError(
            "Could not construct capacity-feasible clusters. "
            "The fleet has enough total capacity, but the demand items may not fit "
            "the available vehicle bins."
        )

    loads = {
        vehicle: sum(customer.demand for customer in grouped)
        for vehicle, grouped in assignments.items()
    }
    return CapacityClusteringResult(
        assignments=assignments,
        centers=centers,
        loads=loads,
        method=method,
    )

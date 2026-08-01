from __future__ import annotations

import math
import random

from .models import Customer, DispatchInstance


def _polar_to_xy(radius: float, angle: float) -> tuple[float, float]:
    return radius * math.cos(angle), radius * math.sin(angle)


def generate_dispatch_instance(
    seed: int,
    num_customers: int,
    num_vehicles: int,
    vehicle_capacity: int,
    service_radius: float = 18.0,
    cluster_count: int = 4,
    demand_low: int = 1,
    demand_high: int = 5,
) -> DispatchInstance:
    """Generate a synthetic last-mile instance."""
    if num_customers < 4:
        raise ValueError("num_customers must be >= 4")
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be >= 1")
    if vehicle_capacity < 1:
        raise ValueError("vehicle_capacity must be >= 1")

    rng = random.Random(seed)
    depot = (0.0, 0.0)

    # Build random demand hotspots.
    centers: list[tuple[float, float]] = []
    for _ in range(max(1, cluster_count)):
        radius = rng.uniform(service_radius * 0.2, service_radius * 0.75)
        angle = rng.uniform(0.0, 2 * math.pi)
        centers.append(_polar_to_xy(radius, angle))

    customers: list[Customer] = []
    for customer_id in range(1, num_customers + 1):
        cx, cy = centers[rng.randrange(len(centers))]
        jitter_r = rng.uniform(0.0, service_radius * 0.18)
        jitter_a = rng.uniform(0.0, 2 * math.pi)
        jx, jy = _polar_to_xy(jitter_r, jitter_a)

        x = cx + jx
        y = cy + jy
        demand = rng.randint(demand_low, demand_high)
        customers.append(Customer(customer_id=customer_id, x=x, y=y, demand=demand))

    return DispatchInstance(
        depot=depot,
        customers=customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )

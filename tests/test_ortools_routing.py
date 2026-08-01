from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.geometry import cycle_distance
from quantum_route_forge.routing import nearest_neighbor, solve_tsp_ortools


def test_ortools_route_contains_each_customer_once():
    instance = generate_dispatch_instance(
        seed=77,
        num_customers=10,
        num_vehicles=1,
        vehicle_capacity=40,
    )
    ordered = solve_tsp_ortools(
        depot=instance.depot,
        customers=instance.customers,
        time_limit_sec=1,
    )
    nearest = nearest_neighbor(instance.depot, instance.customers)

    assert sorted(customer.customer_id for customer in ordered) == list(
        range(1, 11)
    )
    assert len({customer.customer_id for customer in ordered}) == 10
    assert cycle_distance(
        [customer.point for customer in ordered],
        depot=instance.depot,
    ) <= cycle_distance(
        [customer.point for customer in nearest],
        depot=instance.depot,
    ) + 1e-9

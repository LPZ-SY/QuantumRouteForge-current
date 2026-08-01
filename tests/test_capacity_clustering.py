from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.clustering import capacity_constrained_kmeans


def test_capacity_clustering_assigns_every_customer_once():
    instance = generate_dispatch_instance(
        seed=2026,
        num_customers=24,
        num_vehicles=4,
        vehicle_capacity=22,
    )
    result = capacity_constrained_kmeans(instance, seed=2026)

    assigned_ids = [
        customer.customer_id
        for customers in result.assignments.values()
        for customer in customers
    ]
    assert sorted(assigned_ids) == [
        customer.customer_id for customer in instance.customers
    ]
    assert len(assigned_ids) == len(set(assigned_ids))
    assert all(load <= instance.vehicle_capacity for load in result.loads.values())

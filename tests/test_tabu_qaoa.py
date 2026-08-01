from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import Customer, generate_dispatch_instance
from quantum_route_forge.ablation import summarize_paired_ablation
from quantum_route_forge.clustering import capacity_constrained_kmeans
from quantum_route_forge.tabu_qaoa import run_tabu_qaoa


def test_adaptive_tabu_qaoa_keeps_capacity_and_best_solution():
    instance = generate_dispatch_instance(
        seed=1234,
        num_customers=20,
        num_vehicles=4,
        vehicle_capacity=19,
    )
    clustered = capacity_constrained_kmeans(instance, seed=1234)
    result = run_tabu_qaoa(
        assignments=clustered.assignments,
        depot=instance.depot,
        vehicle_capacity=instance.vehicle_capacity,
        iterations=8,
        seed=1234,
    )

    assert result.best_distance <= result.initial_distance + 1e-9
    assigned_ids = [
        customer.customer_id
        for customers in result.assignments.values()
        for customer in customers
    ]
    assert len(assigned_ids) == len(set(assigned_ids)) == len(instance.customers)
    assert all(
        sum(customer.demand for customer in customers)
        <= instance.vehicle_capacity
        for customers in result.assignments.values()
    )
    assert all(
        2 <= len(record.selected_customer_ids) <= 6
        for record in result.records
    )
    assert all(record.cnot_count <= 20 for record in result.records)
    assert all(
        record.cnot_count == 2 * len(record.edges or [])
        for record in result.records
    )


def _uneven_assignments():
    return {
        0: [Customer(1, 0.0, 0.0, 1)],
        1: [
            Customer(2, 1.0, 0.0, 1),
            Customer(3, 1.2, 0.0, 1),
        ],
        2: [
            Customer(customer_id, 40.0 + customer_id, 0.0, 1)
            for customer_id in range(4, 10)
        ],
        3: [
            Customer(customer_id, 90.0 + customer_id, 0.0, 1)
            for customer_id in range(10, 16)
        ],
    }


def test_adaptive_policy_keeps_small_vehicle_pair_in_neighborhood():
    result = run_tabu_qaoa(
        assignments=_uneven_assignments(),
        depot=(0.0, 0.0),
        vehicle_capacity=20,
        iterations=1,
        subproblem_size=6,
        seed=2026,
    )

    assert len(result.records) == 1
    assert result.records[0].vehicle_pair == (0, 1)
    assert len(result.records[0].selected_customer_ids) == 3
    assert result.records[0].sub_k == 3
    assert result.records[0].n_edges == 3
    assert result.records[0].sub_cnot == 6
    assert result.records[0].cnot_count == 6
    assert result.records[0].cut_after in {
        result.records[0].cut_before,
        result.records[0].cut_proposed,
    }
    assert (
        result.records[0].paired_improvement_delta
        == result.records[0].quantum_improvement
        - result.records[0].classical_improvement
    )


def test_paired_ablation_groups_by_actual_subproblem_size():
    assignments = _uneven_assignments()
    result = run_tabu_qaoa(
        assignments=assignments,
        depot=(0.0, 0.0),
        vehicle_capacity=20,
        iterations=4,
        subproblem_size=6,
        seed=2026,
    )
    summary = summarize_paired_ablation(result.records)

    assert summary["total_iterations"] == len(result.records)
    assert sum(
        layer["iterations"]
        for layer in summary["layers"].values()
    ) == len(result.records)
    for sub_k, layer in summary["layers"].items():
        assert int(sub_k) == layer["sub_k"]
        assert "mean_quantum_improvement" in layer
        assert "mean_classical_improvement" in layer
        assert "mean_delta_b_minus_a" in layer


def test_pair_with_fewer_than_two_customers_does_not_waste_iteration():
    assignments = {
        0: [],
        1: [Customer(1, 0.0, 0.0, 1)],
        2: [
            Customer(2, 20.0, 0.0, 1),
            Customer(3, 20.5, 0.0, 1),
        ],
    }
    result = run_tabu_qaoa(
        assignments=assignments,
        depot=(0.0, 0.0),
        vehicle_capacity=10,
        iterations=2,
        subproblem_size=6,
        seed=2026,
    )

    assert len(result.records) == 2
    assert all(record.sub_k >= 2 for record in result.records)

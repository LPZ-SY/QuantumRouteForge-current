from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization


def test_classical_smoke():
    instance = generate_dispatch_instance(
        seed=123,
        num_customers=16,
        num_vehicles=3,
        vehicle_capacity=20,
    )
    result = run_optimization(instance=instance, mode="classical", num_reads=100)
    assert len(result.routes) == 3
    assert result.total_distance > 0
    total_load = sum(route.load for route in result.routes)
    assert total_load == instance.total_demand
    for route in result.routes:
        assert route.load <= instance.vehicle_capacity


def test_hybrid_local_emits_adaptive_stratified_metadata():
    instance = generate_dispatch_instance(
        seed=2026,
        num_customers=18,
        num_vehicles=4,
        vehicle_capacity=28,
    )
    result = run_optimization(
        instance=instance,
        mode="hybrid_local",
        tabu_iterations=4,
        routing_method="heuristic",
    )

    assert result.metadata.qaoa_subproblem_policy == "adaptive"
    records = result.metadata.quantum_iteration_records or []
    assert len(records) == 4
    required = {
        "iteration",
        "sub_k",
        "n_edges",
        "sub_cnot",
        "vehicles",
        "cut_before",
        "cut_proposed",
        "cut_after",
        "quantum_improvement",
        "classical_improvement",
        "paired_improvement_delta",
    }
    assert all(required <= record.keys() for record in records)
    summary = result.metadata.stratified_ablation or {}
    assert summary["total_iterations"] == len(records)
    assert sum(
        layer["iterations"]
        for layer in summary["layers"].values()
    ) == len(records)

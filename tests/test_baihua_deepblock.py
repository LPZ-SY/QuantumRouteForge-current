from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.baihua_topology import select_baihua_subgraph
from quantum_route_forge.candidate_evaluator import evaluate_counts, true_route_distance
from quantum_route_forge.clustering import capacity_constrained_kmeans
from quantum_route_forge.deepblock_builder import build_overlapping_blocks, scan_sequence
from quantum_route_forge.deepblock_solver import DeepBlockConfig, run_deepblock_arm
from quantum_route_forge.models import Customer
from quantum_route_forge.qaoa_depth_runner import (
    build_qaoa_qasm,
    compilation_audit,
    pretrain_parameters,
    simulate_counts,
)
from quantum_route_forge.scenario import generate_dispatch_instance
from quantum_route_forge.sparse_proxy_qubo import build_sparse_proxy_qubo


def _interaction(ids):
    return {
        (left, right): 1.0 / (1 + abs(left - right))
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    }


def test_overlapping_blocks_cover_pool_and_scan_forward_reverse():
    ids = list(range(1, 17))
    blocks = build_overlapping_blocks(
        ids,
        vehicle_pair=(0, 1),
        interactions=_interaction(ids),
        block_size=8,
        overlap=3,
    )
    assert len(blocks) == 3
    assert set().union(*(set(block.customer_ids) for block in blocks)) == set(ids)
    assert all(block.width == 8 for block in blocks)
    assert all(len(block.overlap_with_previous) >= 3 for block in blocks[1:])
    assert [block.block_id for block in scan_sequence(blocks)] == ["B1", "B2", "B3", "B3", "B2", "B1"]


def test_scan_order_validation():
    assert DeepBlockConfig(scan_order="forward").scan_order == "forward"
    try:
        DeepBlockConfig(scan_order="invalid")
    except ValueError as exc:
        assert "scan_order" in str(exc)
    else:
        raise AssertionError("invalid scan order must be rejected")


def test_blocks_degrade_without_dummy_customers():
    small = list(range(1, 8))
    blocks = build_overlapping_blocks(small, (0, 1), _interaction(small), block_size=8, overlap=3)
    assert len(blocks) == 1
    assert blocks[0].width == 7
    medium = list(range(1, 11))
    blocks = build_overlapping_blocks(medium, (0, 1), _interaction(medium), block_size=8, overlap=3)
    assert len(blocks) == 2
    assert set().union(*(set(block.customer_ids) for block in blocks)) == set(medium)


def test_topology_selection_and_manual_validation():
    chip_info = {
        "calibration_time": "2026-08-01T00:00:00+08:00",
        "qubits_info": {
            str(index): {"qubit_index": index, "readout_fidelity": 0.98, "T1": 120, "T2": 100}
            for index in range(9)
        },
        "couplers_info": {
            str(index): {"qubits_index": [index, index + 1], "fidelity": 0.99 - index * 0.001}
            for index in range(8)
        },
    }
    selected = select_baihua_subgraph(chip_info, width=8)
    assert selected.width == 8
    assert selected.uncalibrated_couplings == 0
    assert len(selected.logical_edges) == 7
    manual = select_baihua_subgraph(chip_info, width=4, manual_qubits=[1, 2, 3, 4])
    assert manual.qubits == (1, 2, 3, 4)


def _proxy_fixture():
    customers = [
        Customer(1, 0.0, 1.0, 1),
        Customer(2, 0.0, 2.0, 1),
        Customer(3, 10.0, 1.0, 1),
        Customer(4, 10.0, 2.0, 1),
    ]
    assignments = {0: customers[:2], 1: customers[2:]}
    proxy = build_sparse_proxy_qubo(
        assignments=assignments,
        block_customers=customers,
        vehicle_pair=(0, 1),
        depot=(5.0, 0.0),
        vehicle_capacity=4,
        allowed_logical_edges=[(0, 1), (1, 2), (2, 3)],
    )
    return customers, assignments, proxy


def test_sparse_qubo_only_keeps_physical_edges():
    _customers, _assignments, proxy = _proxy_fixture()
    assert set(proxy.logical_edges) <= {(0, 1), (1, 2), (2, 3)}
    assert all(row.reason.startswith("pruned_") for row in proxy.pruned_interactions)


def test_p1_p2_p3_build_simulate_and_compile_audit():
    _customers, _assignments, proxy = _proxy_fixture()
    for depth in (1, 2, 3):
        parameters = pretrain_parameters(proxy, depth=depth, rounds=1)
        qasm = build_qaoa_qasm(proxy, depth, parameters.gamma, parameters.beta)
        counts = simulate_counts(proxy, parameters, shots=128, seed=depth)
        audit = compilation_audit(qasm, swap_count=0, mapping_verified=True)
        assert sum(counts.values()) == 128
        assert audit.passed
        assert audit.cnot_count == 2 * len(proxy.kept_interactions) * depth


def test_candidate_budget_and_acceptance_use_true_distance():
    customers, assignments, proxy = _proxy_fixture()
    counts = {format(value, "04b"): 1 for value in range(16)}
    batch, accepted = evaluate_counts(
        arm="test",
        counts=counts,
        proxy=proxy,
        assignments=assignments,
        block_customers=customers,
        depot=(5.0, 0.0),
        vehicle_capacity=4,
        candidate_k=3,
        routing_method="heuristic",
    )
    assert batch.evaluated_candidates == 3
    assert batch.unique_candidates == 16
    assert batch.best_of_shots is not None
    if accepted is not None:
        assert true_route_distance(accepted, (5.0, 0.0)) < true_route_distance(assignments, (5.0, 0.0))


def test_simulator_deepblock_integration_updates_and_logs_no_hardware():
    instance = generate_dispatch_instance(
        seed=7,
        num_customers=12,
        num_vehicles=3,
        vehicle_capacity=18,
    )
    clustering = capacity_constrained_kmeans(instance, seed=7)
    result = run_deepblock_arm(
        instance=instance,
        initial_assignments=clustering.assignments,
        arm="sim",
        config=DeepBlockConfig(
            pool_size=10,
            block_size=6,
            overlap=2,
            qaoa_depth=2,
            shots=64,
            candidate_k=4,
            routing_method="heuristic",
        ),
        seed=7,
    )
    assert result.attempted_subproblems >= 2
    assert result.final_distance <= result.baseline_distance + 1e-9
    assert sum(len(customers) for customers in result.assignments.values()) == 12

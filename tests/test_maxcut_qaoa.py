from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.maxcut_qaoa import (
    build_qaoa_openqasm,
    build_sparse_maxcut,
    cut_value,
    solve_maxcut_exact,
)


def test_six_qubit_qaoa_respects_gate_budget():
    instance = generate_dispatch_instance(
        seed=7,
        num_customers=8,
        num_vehicles=2,
        vehicle_capacity=20,
    )
    problem = build_sparse_maxcut(instance.customers[:6], max_edges=10)
    circuit = build_qaoa_openqasm(problem)

    assert problem.num_qubits == 6
    assert len(problem.edges) == 10
    assert circuit.cnot_count == 20
    assert circuit.qasm.count("cx ") == 20
    assert circuit.qasm.count("measure ") == 6


def test_exact_fallback_returns_a_valid_cut():
    instance = generate_dispatch_instance(
        seed=9,
        num_customers=6,
        num_vehicles=2,
        vehicle_capacity=20,
    )
    problem = build_sparse_maxcut(instance.customers, max_edges=10)
    bitstring = solve_maxcut_exact(problem)

    assert len(bitstring) == 6
    assert set(bitstring) <= {"0", "1"}
    assert cut_value(problem, bitstring) > 0

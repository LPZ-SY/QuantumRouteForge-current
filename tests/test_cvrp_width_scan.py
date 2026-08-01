from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.cvrp_width_scan import (
    SCAN_WIDTHS,
    analyze_counts,
    build_chain_qaoa_qasm,
    build_width_instance,
    classify_bitstring,
    greedy_classical_solution,
    qasm_gate_metrics,
    solve_exact,
)


def test_legacy_width_mapping_and_gate_budget_regression():
    expected = {
        6: (6, 2, 6, 10),
        8: (8, 2, 8, 14),
        10: (10, 2, 10, 18),
        12: (12, 2, 12, 22),
    }
    for width, expected_row in expected.items():
        instance = build_width_instance(width)
        qasm = build_chain_qaoa_qasm(instance)
        metrics = qasm_gate_metrics(qasm)
        assert (
            len(instance.customers),
            instance.vehicles,
            instance.decision_variables,
            metrics["cnot_count"],
        ) == expected_row
        assert metrics["two_qubit_gate_count"] == expected_row[-1]
        assert qasm.count("measure ") == width
        assert qasm.count("rx(0.8)") == width
        assert "rx(1.6)" not in qasm


def test_expanded_widths_are_real_customer_decisions():
    for width in (14, 16, 18, 20, 22):
        instance = build_width_instance(width)
        assert len(instance.customers) == width
        assert instance.decision_variables == width
        assert instance.cnot_count == 2 * (width - 1)
        optimum = solve_exact(instance)
        assert classify_bitstring(instance, optimum.bitstring) == "legal"


def test_filter_keeps_legal_poor_solutions_and_separates_categories():
    instance = build_width_instance(6)
    optimum = solve_exact(instance)
    heuristic = greedy_classical_solution(instance)
    legal_poor = next(
        format(state, "06b")
        for state in range(1, 1 << 6)
        if classify_bitstring(instance, format(state, "06b")) == "legal"
        and format(state, "06b") not in {optimum.bitstring, heuristic}
    )
    illegal = next(
        format(state, "06b")
        for state in range(1, (1 << 6) - 1)
        if classify_bitstring(instance, format(state, "06b")) == "illegal"
    )
    counts = {
        "000000": 20,
        illegal: 15,
        legal_poor: 10,
        optimum.bitstring: 5,
    }
    analysis = analyze_counts(instance, counts)
    assert analysis["raw_top1_category"] == "extreme"
    assert analysis["filtered_top1_bitstring"] == legal_poor
    assert analysis["extreme_ratio"] == 0.4
    assert analysis["illegal_ratio"] == 0.3
    assert analysis["feasible_ratio"] == 0.3


def test_supported_width_catalog_is_stable():
    assert SCAN_WIDTHS == (6, 8, 10, 12, 14, 16, 18, 20, 22)

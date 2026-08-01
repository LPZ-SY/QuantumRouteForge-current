from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .geometry import euclidean
from .models import Customer


@dataclass(frozen=True)
class WeightedEdge:
    left: int
    right: int
    weight: float


@dataclass(frozen=True)
class SparseMaxCutProblem:
    customers: list[Customer]
    edges: list[WeightedEdge]

    @property
    def num_qubits(self) -> int:
        return len(self.customers)

    @property
    def cnot_count(self) -> int:
        return 2 * len(self.edges)


@dataclass(frozen=True)
class QAOACircuit:
    qasm: str
    num_qubits: int
    edge_count: int
    cnot_count: int
    gamma: float
    beta: float


def build_sparse_maxcut(
    customers: Iterable[Customer],
    max_edges: int = 10,
) -> SparseMaxCutProblem:
    selected = list(customers)
    if len(selected) < 2:
        raise ValueError("A Max-Cut subproblem requires at least two customers.")
    if len(selected) > 6:
        raise ValueError("The hardware-safe Max-Cut subproblem is limited to 6 customers.")

    weighted_edges = [
        WeightedEdge(
            left=left,
            right=right,
            weight=euclidean(selected[left].point, selected[right].point),
        )
        for left, right in combinations(range(len(selected)), 2)
    ]
    weighted_edges.sort(
        key=lambda edge: (-edge.weight, edge.left, edge.right)
    )
    return SparseMaxCutProblem(
        customers=selected,
        edges=weighted_edges[: max(1, min(int(max_edges), len(weighted_edges)))],
    )


def build_qaoa_openqasm(
    problem: SparseMaxCutProblem,
    gamma: float = 1.1,
    beta: float = 0.8,
) -> QAOACircuit:
    """Build a p=1 Max-Cut QAOA circuit using explicit CX-RZ-CX blocks."""
    if problem.num_qubits > 6:
        raise ValueError("QAOA circuit exceeds the 6-qubit project limit.")
    if len(problem.edges) > 10:
        raise ValueError("QAOA circuit exceeds the 10-edge project limit.")

    max_weight = max((edge.weight for edge in problem.edges), default=1.0)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{problem.num_qubits}];",
        f"creg c[{problem.num_qubits}];",
    ]
    for qubit in range(problem.num_qubits):
        lines.append(f"h q[{qubit}];")

    # For Max-Cut, exp(-i*gamma*w*(1-ZZ)/2) is RZZ(-gamma*w)
    # up to a global phase. RZZ(theta) = CX-RZ(theta)-CX.
    for edge in problem.edges:
        normalized_weight = edge.weight / max(1e-12, max_weight)
        theta = -float(gamma) * normalized_weight
        lines.append(f"cx q[{edge.left}],q[{edge.right}];")
        lines.append(f"rz({theta:.12g}) q[{edge.right}];")
        lines.append(f"cx q[{edge.left}],q[{edge.right}];")

    for qubit in range(problem.num_qubits):
        lines.append(f"rx({2.0 * float(beta):.12g}) q[{qubit}];")
    for qubit in range(problem.num_qubits):
        lines.append(f"measure q[{qubit}] -> c[{qubit}];")

    return QAOACircuit(
        qasm="\n".join(lines) + "\n",
        num_qubits=problem.num_qubits,
        edge_count=len(problem.edges),
        cnot_count=problem.cnot_count,
        gamma=float(gamma),
        beta=float(beta),
    )


def bitstring_to_bits(bitstring: str, num_qubits: int) -> list[int]:
    cleaned = str(bitstring or "").replace(" ", "").strip()
    if not cleaned or any(bit not in "01" for bit in cleaned):
        raise ValueError("Bitstring must contain only 0 and 1.")
    cleaned = cleaned[-num_qubits:].zfill(num_qubits)
    # OpenQASM count keys display the highest classical bit on the left.
    return [int(bit) for bit in reversed(cleaned)]


def cut_value(problem: SparseMaxCutProblem, bitstring: str) -> float:
    bits = bitstring_to_bits(bitstring, problem.num_qubits)
    return sum(
        edge.weight
        for edge in problem.edges
        if bits[edge.left] != bits[edge.right]
    )


def solve_maxcut_exact(problem: SparseMaxCutProblem) -> str:
    """Deterministic local fallback used for tests and unavailable hardware."""
    best_bitstring = "0" * problem.num_qubits
    best_value = -1.0
    for state in range(1 << problem.num_qubits):
        displayed = format(state, f"0{problem.num_qubits}b")
        value = cut_value(problem, displayed)
        if value > best_value + 1e-12:
            best_bitstring = displayed
            best_value = value
    return best_bitstring

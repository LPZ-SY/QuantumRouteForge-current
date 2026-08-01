from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Iterable, Mapping, Sequence

from .geometry import euclidean
from .models import Customer, DispatchInstance
from .routing import build_route_plans
from .scenario import generate_dispatch_instance


SCAN_WIDTHS = (6, 8, 10, 12, 14, 16, 18, 20, 22)
SCAN_VEHICLES = 2
SCAN_SEED = 2026
CAPACITY_RATIO = 1.15
GAMMA = 1.1
BETA = 0.8


@dataclass(frozen=True)
class WidthScanInstance:
    """Two-vehicle CVRP reassignment problem used by the hardware scan.

    One logical bit is one real customer-to-vehicle decision:
    0 assigns the customer to vehicle 0 and 1 assigns it to vehicle 1.
    The objective is the repository's Max-Cut reassignment proxy, restricted
    to a customer-id chain so every cost edge can be embedded on one calibrated
    physical coupling without SWAP insertion.
    """

    width: int
    seed: int
    vehicles: int
    capacity: int
    dispatch: DispatchInstance
    edge_weights: tuple[float, ...]

    @property
    def customers(self) -> list[Customer]:
        return self.dispatch.customers

    @property
    def decision_variables(self) -> int:
        return self.width

    @property
    def cnot_count(self) -> int:
        return 2 * len(self.edge_weights)

    @property
    def total_demand(self) -> int:
        return self.dispatch.total_demand


@dataclass(frozen=True)
class PhysicalChain:
    width: int
    qubits: tuple[int, ...]
    coupling_fidelities: tuple[float, ...]
    calibration_time: str

    @property
    def minimum_fidelity(self) -> float:
        return min(self.coupling_fidelities, default=0.0)

    @property
    def average_fidelity(self) -> float:
        if not self.coupling_fidelities:
            return 0.0
        return sum(self.coupling_fidelities) / len(self.coupling_fidelities)

    @property
    def uncalibrated_couplings(self) -> int:
        return sum(fidelity <= 0.0 for fidelity in self.coupling_fidelities)

    @property
    def estimated_circuit_fidelity(self) -> float:
        # Every logical ZZ term is CX-RZ-CX, so each selected coupler is used
        # twice. This is the same coupling-only estimate used by the prior scan.
        if self.uncalibrated_couplings:
            return 0.0
        return math.prod(fidelity * fidelity for fidelity in self.coupling_fidelities)

    def payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "minimum_fidelity": self.minimum_fidelity,
            "average_fidelity": self.average_fidelity,
            "uncalibrated_couplings": self.uncalibrated_couplings,
            "estimated_circuit_fidelity": self.estimated_circuit_fidelity,
        }


@dataclass(frozen=True)
class ExactSolution:
    bitstring: str
    objective: float
    vehicle_loads: tuple[int, int]


def build_width_instance(
    width: int,
    seed: int = SCAN_SEED,
    vehicles: int = SCAN_VEHICLES,
    capacity_ratio: float = CAPACITY_RATIO,
) -> WidthScanInstance:
    if width not in SCAN_WIDTHS:
        raise ValueError(f"width must be one of {SCAN_WIDTHS}")
    if vehicles != 2:
        raise ValueError("The width scan uses the existing two-way Max-Cut encoding.")

    probe = generate_dispatch_instance(
        seed=seed,
        num_customers=width,
        num_vehicles=vehicles,
        vehicle_capacity=1,
    )
    capacity = max(1, math.ceil(probe.total_demand / vehicles * capacity_ratio))
    dispatch = generate_dispatch_instance(
        seed=seed,
        num_customers=width,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    weights = tuple(
        euclidean(left.point, right.point)
        for left, right in zip(dispatch.customers, dispatch.customers[1:])
    )
    return WidthScanInstance(
        width=width,
        seed=seed,
        vehicles=vehicles,
        capacity=capacity,
        dispatch=dispatch,
        edge_weights=weights,
    )


def canonical_bitstring(bitstring: str, width: int) -> str:
    cleaned = str(bitstring or "").replace(" ", "").strip()
    if not cleaned or any(bit not in "01" for bit in cleaned):
        raise ValueError("bitstring must contain only 0 and 1")
    return cleaned[-width:].zfill(width)


def logical_bits(bitstring: str, width: int) -> tuple[int, ...]:
    # OpenQASM count keys print the highest classical bit on the left.
    return tuple(int(bit) for bit in reversed(canonical_bitstring(bitstring, width)))


def displayed_bitstring(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in reversed(bits))


def vehicle_loads(instance: WidthScanInstance, bitstring: str) -> tuple[int, int]:
    bits = logical_bits(bitstring, instance.width)
    load0 = sum(c.demand for c, bit in zip(instance.customers, bits) if bit == 0)
    return load0, instance.total_demand - load0


def classify_bitstring(instance: WidthScanInstance, bitstring: str) -> str:
    canonical = canonical_bitstring(bitstring, instance.width)
    if canonical == "0" * instance.width or canonical == "1" * instance.width:
        return "extreme"
    load0, load1 = vehicle_loads(instance, canonical)
    if load0 > instance.capacity or load1 > instance.capacity:
        return "illegal"
    if load0 <= 0 or load1 <= 0:
        return "illegal"
    return "legal"


def cut_objective(instance: WidthScanInstance, bitstring: str) -> float:
    bits = logical_bits(bitstring, instance.width)
    return sum(
        weight
        for weight, left, right in zip(instance.edge_weights, bits, bits[1:])
        if left != right
    )


def solve_exact(instance: WidthScanInstance) -> ExactSolution:
    """Exact dynamic program over position, vehicle-0 load, and last bit."""
    # value: (objective, logical bit tuple)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    first_demand = instance.customers[0].demand
    states[(first_demand, 0)] = (0.0, (0,))
    states[(0, 1)] = (0.0, (1,))

    prefix_demand = first_demand
    for index in range(1, instance.width):
        demand = instance.customers[index].demand
        prefix_demand += demand
        weight = instance.edge_weights[index - 1]
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (load0, last_bit), (value, bits) in states.items():
            for bit in (0, 1):
                candidate_load0 = load0 + (demand if bit == 0 else 0)
                candidate_load1 = prefix_demand - candidate_load0
                if candidate_load0 > instance.capacity or candidate_load1 > instance.capacity:
                    continue
                candidate_value = value + (weight if bit != last_bit else 0.0)
                key = (candidate_load0, bit)
                old = next_states.get(key)
                candidate_bits = bits + (bit,)
                if (
                    old is None
                    or candidate_value > old[0] + 1e-12
                    or (
                        abs(candidate_value - old[0]) <= 1e-12
                        and candidate_bits < old[1]
                    )
                ):
                    next_states[key] = (candidate_value, candidate_bits)
        states = next_states

    feasible = [
        (value, bits, load0)
        for (load0, _last), (value, bits) in states.items()
        if 0 < load0 < instance.total_demand
        and instance.total_demand - load0 <= instance.capacity
    ]
    if not feasible:
        raise RuntimeError(f"width={instance.width} has no feasible assignment")
    value, bits, load0 = max(feasible, key=lambda item: (item[0], tuple(-x for x in item[1])))
    return ExactSolution(
        bitstring=displayed_bitstring(bits),
        objective=float(value),
        vehicle_loads=(load0, instance.total_demand - load0),
    )


def is_exact_optimum(
    instance: WidthScanInstance,
    bitstring: str,
    optimum: ExactSolution,
) -> bool:
    return (
        classify_bitstring(instance, bitstring) == "legal"
        and abs(cut_objective(instance, bitstring) - optimum.objective) <= 1e-9
    )


def solution_quality(
    instance: WidthScanInstance,
    bitstring: str,
    optimum: ExactSolution,
) -> float:
    if classify_bitstring(instance, bitstring) != "legal":
        return 0.0
    if optimum.objective <= 0:
        return 1.0
    return max(0.0, min(1.0, cut_objective(instance, bitstring) / optimum.objective))


def route_distance_nn_2opt(instance: WidthScanInstance, bitstring: str) -> float:
    bits = logical_bits(bitstring, instance.width)
    assignments = {
        0: [c for c, bit in zip(instance.customers, bits) if bit == 0],
        1: [c for c, bit in zip(instance.customers, bits) if bit == 1],
    }
    routes = build_route_plans(
        assignments=assignments,
        depot=instance.dispatch.depot,
        two_opt_rounds=2,
        routing_method="heuristic",
    )
    return float(sum(route.distance for route in routes))


def greedy_classical_solution(instance: WidthScanInstance) -> str:
    """Demand-balancing greedy assignment followed by feasible one-bit search."""
    bits = [0] * instance.width
    loads = [0, 0]
    for customer in sorted(
        instance.customers,
        key=lambda item: (-item.demand, item.customer_id),
    ):
        index = customer.customer_id - 1
        candidates = [
            vehicle
            for vehicle in (0, 1)
            if loads[vehicle] + customer.demand <= instance.capacity
        ]
        if not candidates:
            candidates = [0, 1]
        chosen = min(candidates, key=lambda vehicle: (loads[vehicle], vehicle))
        bits[index] = chosen
        loads[chosen] += customer.demand

    current = displayed_bitstring(bits)
    current_value = cut_objective(instance, current)
    improved = True
    while improved:
        improved = False
        best_candidate = current
        best_value = current_value
        logical = list(logical_bits(current, instance.width))
        for index in range(instance.width):
            candidate_bits = logical[:]
            candidate_bits[index] = 1 - candidate_bits[index]
            candidate = displayed_bitstring(candidate_bits)
            if classify_bitstring(instance, candidate) != "legal":
                continue
            value = cut_objective(instance, candidate)
            if value > best_value + 1e-12:
                best_candidate = candidate
                best_value = value
        if best_candidate != current:
            current = best_candidate
            current_value = best_value
            improved = True
    return current


def build_chain_qaoa_qasm(
    instance: WidthScanInstance,
    gamma: float = GAMMA,
    beta: float = BETA,
) -> str:
    max_weight = max(instance.edge_weights, default=1.0)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{instance.width}];",
        f"creg c[{instance.width}];",
    ]
    lines.extend(f"h q[{qubit}];" for qubit in range(instance.width))
    for index, weight in enumerate(instance.edge_weights):
        theta = -float(gamma) * weight / max(1e-12, max_weight)
        lines.append(f"cx q[{index}],q[{index + 1}];")
        lines.append(f"rz({theta:.12g}) q[{index + 1}];")
        lines.append(f"cx q[{index}],q[{index + 1}];")
    # Preserve the legacy real-hardware bridge convention: beta is the
    # physical RX rotation angle. The other Max-Cut helper uses the textbook
    # QAOA convention RX(2*beta), so changing this factor would silently alter
    # the historical 6--12-qubit experiment.
    lines.extend(f"rx({float(beta):.12g}) q[{qubit}];" for qubit in range(instance.width))
    lines.extend(
        f"measure q[{qubit}] -> c[{qubit}];"
        for qubit in range(instance.width)
    )
    return "\n".join(lines) + "\n"


def qasm_gate_metrics(qasm: str) -> dict[str, int]:
    """Return ASAP gate depth and CNOT count, excluding barriers/measurements."""
    depths: dict[int, int] = {}
    cnot_count = 0
    two_qubit_gate_count = 0
    gate_count = 0
    for raw_line in str(qasm or "").splitlines():
        line = raw_line.strip().lower()
        if (
            not line
            or line.startswith(("//", "openqasm", "include", "qreg", "creg", "barrier", "measure"))
        ):
            continue
        qubits = [int(value) for value in re.findall(r"q\[(\d+)\]", line)]
        if not qubits:
            continue
        layer = max((depths.get(qubit, 0) for qubit in qubits), default=0) + 1
        for qubit in qubits:
            depths[qubit] = layer
        gate_count += 1
        if line.startswith("cx "):
            cnot_count += 1
        if len(set(qubits)) == 2:
            two_qubit_gate_count += 1
    return {
        "depth": max(depths.values(), default=0),
        "cnot_count": cnot_count,
        "two_qubit_gate_count": two_qubit_gate_count,
        "gate_count": gate_count,
    }


def _coupler_rows(chip_info: Mapping[str, object]) -> list[tuple[int, int, float]]:
    rows = []
    couplers = chip_info.get("couplers_info", {})
    if not isinstance(couplers, Mapping):
        return rows
    for value in couplers.values():
        if not isinstance(value, Mapping):
            continue
        pair = value.get("qubits_index", ())
        if not isinstance(pair, Sequence) or len(pair) != 2:
            continue
        try:
            rows.append((int(pair[0]), int(pair[1]), float(value.get("fidelity") or 0.0)))
        except (TypeError, ValueError):
            continue
    return rows


def find_fidelity_greedy_chain(
    chip_info: Mapping[str, object],
    width: int,
    beam_width: int = 200_000,
) -> PhysicalChain:
    """Fidelity-first greedy beam search; zero-fidelity edges are forbidden."""
    rows = _coupler_rows(chip_info)
    adjacency: dict[int, list[tuple[int, float]]] = {}
    lookup: dict[frozenset[int], float] = {}
    for left, right, fidelity in rows:
        lookup[frozenset((left, right))] = fidelity
        if fidelity <= 0.0:
            continue
        adjacency.setdefault(left, []).append((right, fidelity))
        adjacency.setdefault(right, []).append((left, fidelity))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (-item[1], item[0]))

    # state: path, sum(log fidelity), minimum fidelity
    states: list[tuple[tuple[int, ...], float, float]] = [
        ((qubit,), 0.0, 1.0) for qubit in sorted(adjacency)
    ]
    for _depth in range(1, width):
        candidates: list[tuple[tuple[int, ...], float, float]] = []
        for path, log_product, minimum in states:
            for neighbor, fidelity in adjacency.get(path[-1], ()):
                if neighbor in path:
                    continue
                candidates.append(
                    (
                        path + (neighbor,),
                        log_product + math.log(fidelity),
                        min(minimum, fidelity),
                    )
                )
        if not candidates:
            raise RuntimeError(
                f"No fully calibrated Shenglian chain of width {width} was found."
            )
        # Optimize the weakest link first, then the product. This directly uses
        # calibration fidelity rather than truncating an unweighted long path.
        candidates.sort(key=lambda item: (item[2], item[1], item[0]), reverse=True)
        states = candidates[: max(1, int(beam_width))]

    path, _score, _minimum = max(
        states,
        key=lambda item: (item[2], item[1], item[0]),
    )
    fidelities = tuple(
        lookup[frozenset((left, right))]
        for left, right in zip(path, path[1:])
    )
    return PhysicalChain(
        width=width,
        qubits=path,
        coupling_fidelities=fidelities,
        calibration_time=str(chip_info.get("calibration_time") or ""),
    )


def normalize_counts(
    counts: Mapping[object, object],
    width: int,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_bitstring, raw_count in (counts or {}).items():
        try:
            bitstring = canonical_bitstring(str(raw_bitstring), width)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            normalized[bitstring] = normalized.get(bitstring, 0) + count
    return normalized


def analyze_counts(
    instance: WidthScanInstance,
    counts: Mapping[object, object],
) -> dict[str, object]:
    normalized = normalize_counts(counts, instance.width)
    if not normalized:
        raise ValueError("counts are empty")
    optimum = solve_exact(instance)
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    shots = sum(normalized.values())
    raw_bitstring, raw_count = ranked[0]
    legal_ranked = [
        (bitstring, count)
        for bitstring, count in ranked
        if classify_bitstring(instance, bitstring) == "legal"
    ]
    filtered_top1 = legal_ranked[0][0] if legal_ranked else ""
    top5 = legal_ranked[:5]

    category_shots = {"legal": 0, "illegal": 0, "extreme": 0}
    weighted_quality = 0.0
    legal_weighted_quality = 0.0
    optimal_shots = 0
    for bitstring, count in normalized.items():
        category = classify_bitstring(instance, bitstring)
        category_shots[category] += count
        quality = solution_quality(instance, bitstring, optimum)
        weighted_quality += count * quality
        if category == "legal":
            legal_weighted_quality += count * quality
        if is_exact_optimum(instance, bitstring, optimum):
            optimal_shots += count

    legal_shots = category_shots["legal"]
    top5_best_quality = max(
        (solution_quality(instance, bitstring, optimum) for bitstring, _ in top5),
        default=0.0,
    )
    top5_best_bitstring = max(
        (bitstring for bitstring, _ in top5),
        key=lambda bitstring: solution_quality(instance, bitstring, optimum),
        default="",
    )
    return {
        "shots": shots,
        "unique_bitstrings": len(normalized),
        "exact_optimum_bitstring": optimum.bitstring,
        "exact_optimum_objective": optimum.objective,
        "exact_optimum_loads": list(optimum.vehicle_loads),
        "exact_optimum_route_distance_nn_2opt": route_distance_nn_2opt(
            instance, optimum.bitstring
        ),
        "raw_top1_bitstring": raw_bitstring,
        "raw_top1_count": raw_count,
        "raw_top1_category": classify_bitstring(instance, raw_bitstring),
        "raw_top1_quality": solution_quality(instance, raw_bitstring, optimum),
        "raw_top1_route_distance_nn_2opt": (
            route_distance_nn_2opt(instance, raw_bitstring)
            if classify_bitstring(instance, raw_bitstring) == "legal"
            else None
        ),
        "filtered_top1_bitstring": filtered_top1,
        "filtered_top1_quality": (
            solution_quality(instance, filtered_top1, optimum)
            if filtered_top1
            else 0.0
        ),
        "filtered_top1_route_distance_nn_2opt": (
            route_distance_nn_2opt(instance, filtered_top1)
            if filtered_top1
            else None
        ),
        "top5_candidates": [
            {
                "bitstring": bitstring,
                "count": count,
                "quality": solution_quality(instance, bitstring, optimum),
                "is_exact_optimum": is_exact_optimum(instance, bitstring, optimum),
            }
            for bitstring, count in top5
        ],
        "top5_best_bitstring": top5_best_bitstring,
        "top5_best_quality": top5_best_quality,
        "top5_contains_exact_optimum": any(
            is_exact_optimum(instance, bitstring, optimum)
            for bitstring, _count in top5
        ),
        "optimal_solution_hit_rate": optimal_shots / shots,
        "feasible_ratio": category_shots["legal"] / shots,
        "illegal_ratio": category_shots["illegal"] / shots,
        "extreme_ratio": category_shots["extreme"] / shots,
        "hardware_mean_quality_unfiltered": weighted_quality / shots,
        "hardware_mean_quality_filtered": (
            legal_weighted_quality / legal_shots if legal_shots else 0.0
        ),
    }


def random_sampling_baseline(
    instance: WidthScanInstance,
    shots: int,
    repeats: int = 32,
    seed: int = SCAN_SEED,
) -> dict[str, float]:
    optimum = solve_exact(instance)
    rng = random.Random(seed * 10_000 + instance.width)
    total = max(1, int(shots)) * max(1, int(repeats))
    legal = 0
    illegal = 0
    extreme = 0
    legal_quality = 0.0
    all_quality = 0.0
    optimum_hits = 0
    for _ in range(total):
        state = rng.getrandbits(instance.width)
        bitstring = format(state, f"0{instance.width}b")
        category = classify_bitstring(instance, bitstring)
        quality = solution_quality(instance, bitstring, optimum)
        all_quality += quality
        if category == "legal":
            legal += 1
            legal_quality += quality
        elif category == "extreme":
            extreme += 1
        else:
            illegal += 1
        if is_exact_optimum(instance, bitstring, optimum):
            optimum_hits += 1
    return {
        "samples": float(total),
        "mean_quality_unfiltered": all_quality / total,
        "mean_quality_filtered": legal_quality / legal if legal else 0.0,
        "feasible_ratio": legal / total,
        "illegal_ratio": illegal / total,
        "extreme_ratio": extreme / total,
        "optimal_solution_hit_rate": optimum_hits / total,
    }


def instance_payload(instance: WidthScanInstance) -> dict[str, object]:
    return {
        "width": instance.width,
        "seed": instance.seed,
        "customers": instance.width,
        "vehicles": instance.vehicles,
        "decision_variables": instance.decision_variables,
        "actual_qubits": instance.width,
        "vehicle_capacity": instance.capacity,
        "total_demand": instance.total_demand,
        "depot": list(instance.dispatch.depot),
        "customer_data": [
            {
                "customer_id": customer.customer_id,
                "x": customer.x,
                "y": customer.y,
                "demand": customer.demand,
            }
            for customer in instance.customers
        ],
        "logical_edges": [
            {
                "left_customer_id": index + 1,
                "right_customer_id": index + 2,
                "weight": weight,
            }
            for index, weight in enumerate(instance.edge_weights)
        ],
        "objective_rule": (
            "capacity-feasible weighted Max-Cut on the customer-id chain; "
            "one customer-to-vehicle binary decision per logical qubit"
        ),
        "route_rule": "nearest-neighbor followed by two 2-opt passes",
    }


def classical_payload(instance: WidthScanInstance) -> dict[str, object]:
    optimum = solve_exact(instance)
    heuristic = greedy_classical_solution(instance)
    return {
        "exact": {
            "method": "dynamic_programming_position_load_last_bit",
            "bitstring": optimum.bitstring,
            "objective": optimum.objective,
            "quality": 1.0,
            "vehicle_loads": list(optimum.vehicle_loads),
            "route_distance_nn_2opt": route_distance_nn_2opt(
                instance, optimum.bitstring
            ),
        },
        "heuristic": {
            "method": "demand_balancing_greedy_plus_feasible_1flip_and_nn_2opt",
            "bitstring": heuristic,
            "objective": cut_objective(instance, heuristic),
            "quality": solution_quality(instance, heuristic, optimum),
            "vehicle_loads": list(vehicle_loads(instance, heuristic)),
            "route_distance_nn_2opt": route_distance_nn_2opt(instance, heuristic),
        },
    }


FILTER_RULES = {
    "extreme": (
        "exactly all-0 or all-1; classified first and excluded from filtered candidates"
    ),
    "illegal": (
        "non-extreme string with an empty vehicle or a vehicle load above capacity"
    ),
    "legal": (
        "both vehicles non-empty and within capacity; retained regardless of objective quality"
    ),
    "selection": (
        "filtered top-1/top-5 are ranked only by observed count after removing "
        "extreme and illegal strings; optimality is never used by the filter"
    ),
}

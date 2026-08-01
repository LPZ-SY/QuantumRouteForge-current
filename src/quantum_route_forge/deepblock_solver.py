from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Mapping, Sequence

from .baihua_topology import (
    PhysicalSubgraph,
    offline_identity_subgraph,
    select_baihua_subgraph,
)
from .candidate_evaluator import CandidateBatch, evaluate_counts, true_route_distance
from .deepblock_builder import (
    BoundaryCandidate,
    DeepBlock,
    build_interaction_graph,
    build_overlapping_blocks,
    rank_vehicle_pairs,
    scan_sequence,
    select_boundary_pool,
)
from .experiment_logger import ExperimentLogger
from .models import Customer, DispatchInstance
from .qaoa_depth_runner import (
    CompilationAudit,
    QAOAParameters,
    QAOARunResult,
    build_qaoa_qasm,
    compilation_audit,
    pretrain_parameters,
    random_counts,
    run_baihua_arm,
    run_simulator_arm,
    transpile_for_baihua,
)
from .sparse_proxy_qubo import SparseProxyQUBO, build_sparse_proxy_qubo


@dataclass(frozen=True)
class DeepBlockConfig:
    pool_size: int = 16
    block_size: int = 8
    overlap: int = 3
    qaoa_depth: int = 2
    shots: int = 4096
    candidate_k: int = 8
    max_sweeps: int = 1
    scan_order: str = "bidirectional"
    filter_extremes: bool = False
    routing_method: str = "heuristic"
    backend: str = "Baihua"
    max_cnot: int = 96
    max_depth: int = 240
    submit_hardware: bool = False
    confirm_hardware_submit: bool = False
    wait_hardware: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.block_size) <= 8:
            raise ValueError("block_size must be between 1 and 8")
        if int(self.qaoa_depth) not in {1, 2, 3}:
            raise ValueError("qaoa_depth must be 1, 2, or 3")
        if int(self.shots) < 1 or int(self.candidate_k) < 1:
            raise ValueError("shots and candidate_k must be positive")
        if self.scan_order not in {"forward", "bidirectional"}:
            raise ValueError("scan_order must be forward or bidirectional")


@dataclass(frozen=True)
class DeepBlockArmResult:
    arm: str
    baseline_distance: float
    final_distance: float
    accepted_moves: int
    attempted_subproblems: int
    blocks: tuple[DeepBlock, ...]
    candidate_pool: tuple[BoundaryCandidate, ...]
    assignments: dict[int, list[Customer]]
    status: str

    @property
    def improvement(self) -> float:
        return self.baseline_distance - self.final_distance

    def payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "baseline_distance": self.baseline_distance,
            "final_distance": self.final_distance,
            "improvement": self.improvement,
            "accepted_moves": self.accepted_moves,
            "attempted_subproblems": self.attempted_subproblems,
            "blocks": [block.payload() for block in self.blocks],
            "candidate_pool": [candidate.payload() for candidate in self.candidate_pool],
            "assignments": {
                str(vehicle): [customer.customer_id for customer in customers]
                for vehicle, customers in sorted(self.assignments.items())
            },
            "status": self.status,
        }


def _customers_by_id(instance: DispatchInstance) -> dict[int, Customer]:
    return {customer.customer_id: customer for customer in instance.customers}


def prepare_deepblocks(
    instance: DispatchInstance,
    assignments: Mapping[int, Sequence[Customer]],
    config: DeepBlockConfig,
) -> tuple[list[BoundaryCandidate], list[DeepBlock]]:
    pairs = rank_vehicle_pairs(assignments, instance.depot)
    if not pairs:
        return [], []
    # Prefer the closest pair with enough customers; fall back to the closest pair.
    vehicle_pair = max(
        pairs,
        key=lambda pair: (
            min(config.pool_size, len(assignments.get(pair[0], ())) + len(assignments.get(pair[1], ()))),
            -pairs.index(pair),
        ),
    )
    pool = select_boundary_pool(
        assignments,
        vehicle_pair=vehicle_pair,
        depot=instance.depot,
        pool_size=config.pool_size,
    )
    by_id = _customers_by_id(instance)
    customers = [by_id[row.customer_id] for row in pool]
    interactions = build_interaction_graph(
        customers,
        candidate_scores={row.customer_id: row.score for row in pool},
    )
    blocks = build_overlapping_blocks(
        [row.customer_id for row in pool],
        vehicle_pair=vehicle_pair,
        interactions=interactions,
        block_size=config.block_size,
        overlap=config.overlap,
        max_blocks=3,
    )
    return pool, blocks


def _slice_topology(topology: PhysicalSubgraph, width: int) -> PhysicalSubgraph:
    return PhysicalSubgraph(
        qubits=topology.qubits[:width],
        couplers=topology.couplers[: max(0, width - 1)],
        calibration_time=topology.calibration_time,
        score=topology.score,
        source=topology.source,
    )


def _diagnostic_run(
    arm: str,
    proxy: SparseProxyQUBO,
    shots: int,
    seed: int,
) -> QAOARunResult:
    counts = (
        random_counts(proxy.width, shots=shots, seed=seed)
        if arm == "random"
        else {format(value, f"0{proxy.width}b"): 1 for value in range(1 << proxy.width)}
    )
    parameters = QAOAParameters(
        depth=1,
        gamma=(0.0,),
        beta=(0.0,),
        optimizer="not_applicable",
        initial_value=0.0,
        final_value=0.0,
        evaluations=0,
    )
    return QAOARunResult(
        arm=arm,
        counts=counts,
        qasm="// classical diagnostic arm; no quantum circuit\n",
        physical_qasm="",
        parameters=parameters,
        compilation=compilation_audit("", swap_count=0, mapping_verified=True),
        shots=sum(counts.values()),
        seed=seed,
        message="Uniform random shots." if arm == "random" else "Exact enumeration of the local block.",
    )


def _instance_payload(instance: DispatchInstance) -> dict[str, object]:
    return {
        "depot": list(instance.depot),
        "num_vehicles": instance.num_vehicles,
        "vehicle_capacity": instance.vehicle_capacity,
        "customers": [asdict(customer) for customer in instance.customers],
    }


def run_deepblock_arm(
    *,
    instance: DispatchInstance,
    initial_assignments: Mapping[int, Sequence[Customer]],
    arm: str,
    config: DeepBlockConfig,
    seed: int,
    logger: ExperimentLogger | None = None,
    chip_info: Mapping[str, object] | None = None,
    manual_physical_qubits: Sequence[int] | None = None,
    api_token: str = "",
) -> DeepBlockArmResult:
    normalized_arm = str(arm).strip().lower()
    if normalized_arm not in {"random", "sim", "baihua", "exact"}:
        raise ValueError("arm must be random, sim, baihua, or exact")
    current = {vehicle: list(customers) for vehicle, customers in initial_assignments.items()}
    pool, blocks = prepare_deepblocks(instance, current, config)
    baseline_distance = true_route_distance(current, instance.depot, routing_method=config.routing_method)
    if not blocks:
        return DeepBlockArmResult(
            arm=normalized_arm,
            baseline_distance=baseline_distance,
            final_distance=baseline_distance,
            accepted_moves=0,
            attempted_subproblems=0,
            blocks=(),
            candidate_pool=tuple(pool),
            assignments=current,
            status="no_refinement_space",
        )

    by_id = _customers_by_id(instance)
    accepted_moves = 0
    iteration = 0
    status = "completed"
    parameter_transfer: QAOAParameters | None = None

    for sweep in range(max(1, int(config.max_sweeps))):
        accepted_this_sweep = 0
        sweep_blocks = list(blocks) if config.scan_order == "forward" else scan_sequence(blocks)
        for block in sweep_blocks:
            iteration += 1
            block_customers = [by_id[customer_id] for customer_id in block.customer_ids]
            width = len(block_customers)
            if chip_info:
                topology = select_baihua_subgraph(
                    chip_info,
                    width=width,
                    manual_qubits=(
                        list(manual_physical_qubits)[:width]
                        if manual_physical_qubits is not None
                        else None
                    ),
                )
            else:
                topology = offline_identity_subgraph(width)
            proxy = build_sparse_proxy_qubo(
                assignments=current,
                block_customers=block_customers,
                vehicle_pair=block.vehicle_pair,
                depot=instance.depot,
                vehicle_capacity=instance.vehicle_capacity,
                allowed_logical_edges=topology.logical_edges,
            )

            if normalized_arm in {"random", "exact"}:
                run_result = _diagnostic_run(
                    normalized_arm,
                    proxy,
                    shots=config.shots,
                    seed=seed * 10_000 + iteration,
                )
            else:
                parameters = pretrain_parameters(
                    proxy,
                    config.qaoa_depth,
                    initial_gamma=(parameter_transfer.gamma if parameter_transfer and parameter_transfer.depth == config.qaoa_depth else None),
                    initial_beta=(parameter_transfer.beta if parameter_transfer and parameter_transfer.depth == config.qaoa_depth else None),
                )
                parameter_transfer = parameters
                if normalized_arm == "sim":
                    run_result = run_simulator_arm(
                        proxy,
                        config.qaoa_depth,
                        shots=config.shots,
                        seed=seed * 10_000 + iteration,
                        parameters=parameters,
                    )
                else:
                    logical_qasm = build_qaoa_qasm(
                        proxy,
                        config.qaoa_depth,
                        parameters.gamma,
                        parameters.beta,
                    )
                    physical_qasm = logical_qasm
                    if chip_info:
                        try:
                            physical_qasm, raw_audit = transpile_for_baihua(
                                logical_qasm,
                                backend_name=config.backend,
                                target_qubits=topology.qubits,
                            )
                            audit = compilation_audit(
                                physical_qasm,
                                swap_count=raw_audit.swap_count,
                                mapping_verified=raw_audit.mapping_verified,
                                uncalibrated_couplings=topology.uncalibrated_couplings,
                                max_cnot=config.max_cnot,
                                max_depth=config.max_depth,
                            )
                        except Exception as exc:  # pragma: no cover - hardware runtime dependent
                            audit = compilation_audit(
                                logical_qasm,
                                mapping_verified=False,
                                uncalibrated_couplings=topology.uncalibrated_couplings,
                                max_cnot=config.max_cnot,
                                max_depth=config.max_depth,
                            )
                            status = f"compile_blocked:{type(exc).__name__}"
                    else:
                        audit = compilation_audit(
                            logical_qasm,
                            mapping_verified=False,
                            max_cnot=config.max_cnot,
                            max_depth=config.max_depth,
                        )
                    run_result = run_baihua_arm(
                        proxy=proxy,
                        parameters=parameters,
                        physical_qasm=physical_qasm,
                        compilation=audit,
                        shots=config.shots,
                        seed=seed * 10_000 + iteration,
                        backend=config.backend,
                        api_token=api_token or os.environ.get("QUAFU_API_TOKEN", ""),
                        submit_hardware=config.submit_hardware,
                        confirm_hardware_submit=config.confirm_hardware_submit,
                        wait=config.wait_hardware,
                    )

            assignments_before = {
                vehicle: list(customers) for vehicle, customers in current.items()
            }
            candidate_batch: CandidateBatch | None = None
            accepted_assignment = None
            if run_result.counts:
                candidate_batch, accepted_assignment = evaluate_counts(
                    arm=run_result.arm,
                    counts=run_result.counts,
                    proxy=proxy,
                    assignments=current,
                    block_customers=block_customers,
                    depot=instance.depot,
                    vehicle_capacity=instance.vehicle_capacity,
                    candidate_k=(1 << width) if normalized_arm == "exact" else config.candidate_k,
                    filter_extremes=config.filter_extremes,
                    routing_method=config.routing_method,
                )
            if accepted_assignment is not None:
                current = accepted_assignment
                accepted_moves += 1
                accepted_this_sweep += 1

            if logger is not None:
                batch_payload = candidate_batch.payload() if candidate_batch else None
                replay_record = {
                    "schema_version": 1,
                    "seed": seed,
                    "iteration": iteration,
                    "sweep": sweep + 1,
                    "block": block.payload(),
                    "instance": _instance_payload(instance),
                    "assignments_before": {
                        str(vehicle): [customer.customer_id for customer in customers]
                        for vehicle, customers in sorted(assignments_before.items())
                    },
                    "assignments_after": {
                        str(vehicle): [customer.customer_id for customer in customers]
                        for vehicle, customers in sorted(current.items())
                    },
                    "proxy": proxy.payload(),
                    "topology": topology.payload(),
                    "run": run_result.payload(),
                    "candidate_batch": batch_payload,
                    "fairness": {
                        "shots": config.shots,
                        "candidate_k": config.candidate_k,
                        "filter_extremes": config.filter_extremes,
                        "routing_method": config.routing_method,
                        "scan_order": config.scan_order,
                    },
                }
                accepted = candidate_batch.accepted if candidate_batch else None
                best = candidate_batch.best_of_shots if candidate_batch else None
                top_rows = list(candidate_batch.top_frequency) if candidate_batch else []

                def best_top_distance(limit: int) -> float | None:
                    values = [row.true_distance for row in top_rows[:limit] if row.feasible_after_repair]
                    return min(values) if values else None

                logger.log_subproblem(
                    seed=seed,
                    iteration=iteration,
                    block_id=block.block_id,
                    arm=run_result.arm,
                    counts=run_result.counts,
                    qasm=run_result.qasm,
                    physical_qasm=run_result.physical_qasm,
                    mapping=topology.payload(),
                    replay_record=replay_record,
                    metrics={
                        "sweep": sweep + 1,
                        "customer_ids": list(block.customer_ids),
                        "vehicle_pair": list(block.vehicle_pair),
                        "overlap_customer_ids": list(block.overlap_with_previous),
                        "qaoa_depth": config.qaoa_depth,
                        "gamma": list(run_result.parameters.gamma),
                        "beta": list(run_result.parameters.beta),
                        "shots": run_result.shots,
                        "candidate_k": (1 << width) if normalized_arm == "exact" else config.candidate_k,
                        "linear_h": list(proxy.linear),
                        "kept_j": [asdict(row) for row in proxy.kept_interactions],
                        "pruned_j": [asdict(row) for row in proxy.pruned_interactions],
                        "proxy_scale": proxy.scale,
                        "physical_qubits": list(topology.qubits),
                        "physical_couplers": [asdict(row) for row in topology.couplers],
                        "calibration_time": topology.calibration_time,
                        "cnot_count": run_result.compilation.cnot_count,
                        "swap_count": run_result.compilation.swap_count,
                        "circuit_depth": run_result.compilation.depth,
                        "two_qubit_gate_layers": run_result.compilation.two_qubit_gate_layers,
                        "compilation_passed": run_result.compilation.passed,
                        "unique_candidates": candidate_batch.unique_candidates if candidate_batch else 0,
                        "feasible_candidates": candidate_batch.feasible_candidates if candidate_batch else 0,
                        "evaluated_candidates": candidate_batch.evaluated_candidates if candidate_batch else 0,
                        "accepted": accepted is not None,
                        "accepted_improvement": accepted.improvement if accepted else 0.0,
                        "top1_best_distance": best_top_distance(1),
                        "top5_best_distance": best_top_distance(5),
                        "top8_best_distance": best_top_distance(8),
                        "best_of_shots_distance": best.true_distance if best else None,
                        "all_zero_probability": candidate_batch.all_zero_probability if candidate_batch else None,
                        "all_one_probability": candidate_batch.all_one_probability if candidate_batch else None,
                        "distribution_entropy": candidate_batch.distribution_entropy if candidate_batch else None,
                        "top_k_probability": candidate_batch.top_k_probability if candidate_batch else None,
                        "task_id": run_result.task_id,
                        "backend": run_result.backend,
                        "message": run_result.message,
                    },
                )
        if accepted_this_sweep == 0:
            break

    final_distance = true_route_distance(current, instance.depot, routing_method=config.routing_method)
    if normalized_arm == "baihua" and not config.submit_hardware:
        status = "dry_run_no_hardware_submission"
    return DeepBlockArmResult(
        arm=normalized_arm,
        baseline_distance=baseline_distance,
        final_distance=final_distance,
        accepted_moves=accepted_moves,
        attempted_subproblems=iteration,
        blocks=tuple(blocks),
        candidate_pool=tuple(pool),
        assignments=current,
        status=status,
    )

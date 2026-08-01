from __future__ import annotations

import math
import re

from .ablation import summarize_paired_ablation
from .assignment_bqm import build_assignment_bqm, decode_assignment
from .clustering import capacity_constrained_kmeans
from .maxcut_qaoa import solve_maxcut_exact
from .models import AssignmentMetadata, DispatchInstance, OptimizationResult
from .quafu_bridge import (
    bitstring_to_vehicle_hints,
    fetch_quafu_task_bitstring,
    submit_quafu_partition_job,
    submit_quafu_qasm_job,
)
from .routing import build_route_plans
from .solvers import solve_bqm_classical
from .tabu_qaoa import QuantumSample, run_tabu_qaoa


def _repair_capacity(assignments, vehicle_capacity: int) -> dict[int, list]:
    """Repair capacity violations by moving customers from overloaded routes."""
    assignments = {k: list(v) for k, v in assignments.items()}
    max_iters = 2000
    for _ in range(max_iters):
        loads = {v: sum(c.demand for c in clist) for v, clist in assignments.items()}
        overloaded = [v for v, load in loads.items() if load > vehicle_capacity]
        if not overloaded:
            return assignments

        moved = False
        for ov in sorted(overloaded, key=lambda v: loads[v], reverse=True):
            # Move higher-demand points first to reduce overload quickly.
            for customer in sorted(assignments[ov], key=lambda c: (-c.demand, c.customer_id)):
                candidates = [
                    v
                    for v in assignments.keys()
                    if v != ov and loads[v] + customer.demand <= vehicle_capacity
                ]
                if not candidates:
                    continue
                target = min(candidates, key=lambda v: loads[v])
                assignments[ov].remove(customer)
                assignments[target].append(customer)
                moved = True
                break
            if moved:
                break

        if not moved:
            # Feasibility should usually hold because global capacity is checked.
            break
    return assignments


def _normalize_bitstring(raw: str) -> tuple[str, str]:
    text = (raw or "").strip().replace(" ", "").replace("_", "")
    if not text:
        return "", ""
    if not re.fullmatch(r"[01]+", text):
        return "", "Manual bitstring ignored: only 0/1 are allowed."
    return text, ""


def _run_legacy_optimization(
    instance: DispatchInstance,
    mode: str,
    time_limit: int = 8,
    num_reads: int = 300,
    num_sweeps: int = 40,
    compactness_weight: float = 1.0,
    assignment_penalty: float = 90.0,
    capacity_penalty: float = 45.0,
    balance_weight: float = 1.2,
    two_opt_rounds: int = 2,
    quafu_token: str = "",
    quafu_backend: str = "",
    quafu_base_url: str = "",
    quafu_shots: int = 1024,
    quafu_wait: bool = True,
    quafu_max_qubits: int = 8,
    quafu_timeout_sec: int = 25,
    quafu_proxy_url: str = "",
    quafu_verify_ssl: bool = True,
    quafu_result_task_id: str = "",
    quafu_manual_bitstring: str = "",
    auto_repair_capacity: bool = False,
    routing_method: str = "heuristic",
) -> OptimizationResult:
    mode = (mode or "classical").lower().strip()
    use_quafu = mode in {"quantum", "quafu"}

    preferred_vehicle_by_customer: dict[int, int] = {}
    quafu_task_id = None
    quafu_backend_used = None
    quafu_bitstring = None
    quafu_endpoint = None
    quafu_msg = ""
    capacity_msg = ""
    used_mode = "classical"
    manual_bitstring, manual_bitstring_msg = _normalize_bitstring(quafu_manual_bitstring)
    requested_result_task_id = str(quafu_result_task_id or "").strip()
    lookup_task_msg = ""

    if not instance.feasible_capacity:
        if not auto_repair_capacity:
            min_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
            raise ValueError(
                "Total demand exceeds fleet capacity. "
                f"Required minimum capacity per vehicle is {min_capacity} "
                f"(demand={instance.total_demand}, vehicles={instance.num_vehicles})."
            )
        repaired_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
        instance = DispatchInstance(
            depot=instance.depot,
            customers=instance.customers,
            num_vehicles=instance.num_vehicles,
            vehicle_capacity=repaired_capacity,
        )
        capacity_msg = (
            f"Capacity auto-repaired to {repaired_capacity} "
            f"(original capacity was infeasible for demand {instance.total_demand})."
        )

    if use_quafu:
        if requested_result_task_id:
            job = fetch_quafu_task_bitstring(
                task_id=requested_result_task_id,
                api_token=quafu_token,
                base_url=quafu_base_url or None,
                timeout_sec=quafu_timeout_sec,
                proxy_url=quafu_proxy_url,
                verify_ssl=quafu_verify_ssl,
            )
            if not job.bitstring:
                lookup_task_msg = f"Task lookup ({requested_result_task_id}): {job.message}"
                job = submit_quafu_partition_job(
                    customers=instance.customers,
                    api_token=quafu_token,
                    backend=quafu_backend or None,
                    base_url=quafu_base_url or None,
                    shots=quafu_shots,
                    wait=quafu_wait,
                    max_qubits=quafu_max_qubits,
                    timeout_sec=quafu_timeout_sec,
                    proxy_url=quafu_proxy_url,
                    verify_ssl=quafu_verify_ssl,
                )
        else:
            job = submit_quafu_partition_job(
                customers=instance.customers,
                api_token=quafu_token,
                backend=quafu_backend or None,
                base_url=quafu_base_url or None,
                shots=quafu_shots,
                wait=quafu_wait,
                max_qubits=quafu_max_qubits,
                timeout_sec=quafu_timeout_sec,
                proxy_url=quafu_proxy_url,
                verify_ssl=quafu_verify_ssl,
            )
        quafu_task_id = job.task_id
        quafu_backend_used = job.backend
        quafu_bitstring = job.bitstring
        quafu_endpoint = job.endpoint
        quafu_msg = " | ".join(x for x in [lookup_task_msg, job.message] if x)

        selected_by_id = {c.customer_id: c for c in instance.customers}
        if job.selected_customer_ids:
            ordered_selected = [selected_by_id[cid] for cid in job.selected_customer_ids if cid in selected_by_id]
        else:
            # Fallback selection when backend accepts submission but does not expose selected ids in response.
            ranked = sorted(instance.customers, key=lambda c: (-c.demand, c.customer_id))
            ordered_selected = ranked[: max(2, min(quafu_max_qubits, len(ranked)))]

        if manual_bitstring and ordered_selected and instance.num_vehicles >= 2:
            preferred_vehicle_by_customer = bitstring_to_vehicle_hints(
                selected_customers=ordered_selected,
                bitstring=manual_bitstring,
                num_vehicles=instance.num_vehicles,
            )
            quafu_bitstring = manual_bitstring
            used_mode = "quafu_quantum_hybrid"
            quafu_msg = " | ".join(x for x in [quafu_msg, "Manual bitstring override applied."] if x)
        elif job.ok and job.bitstring and ordered_selected and instance.num_vehicles >= 2:
            preferred_vehicle_by_customer = bitstring_to_vehicle_hints(
                selected_customers=ordered_selected,
                bitstring=job.bitstring,
                num_vehicles=instance.num_vehicles,
            )
            used_mode = "quafu_quantum_hybrid"
        elif job.ok:
            used_mode = "quafu_submitted_classical_refine"
        else:
            used_mode = "quafu_unavailable_classical_fallback"

        if manual_bitstring_msg:
            quafu_msg = " | ".join(x for x in [quafu_msg, manual_bitstring_msg] if x)

    bqm = build_assignment_bqm(
        instance=instance,
        compactness_weight=compactness_weight,
        assignment_penalty=assignment_penalty,
        capacity_penalty=capacity_penalty,
        balance_weight=balance_weight,
        preferred_vehicle_by_customer=preferred_vehicle_by_customer,
    )
    run = solve_bqm_classical(
        bqm=bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
    )
    assignments = decode_assignment(
        sample=run.sample,
        customers=instance.customers,
        num_vehicles=instance.num_vehicles,
    )
    assignments = _repair_capacity(assignments, vehicle_capacity=instance.vehicle_capacity)
    routes = build_route_plans(
        assignments=assignments,
        depot=instance.depot,
        two_opt_rounds=two_opt_rounds,
        routing_method=routing_method,
        ortools_time_limit_sec=max(
            1,
            int(time_limit) // max(1, instance.num_vehicles),
        ),
        ortools_fallback=True,
    )
    total_distance = sum(r.distance for r in routes)

    return OptimizationResult(
        instance=instance,
        assignments=assignments,
        routes=routes,
        total_distance=total_distance,
        metadata=AssignmentMetadata(
            requested_mode=mode,
            used_mode=used_mode if use_quafu else run.used_mode,
            energy=run.energy,
            message=(
                " | ".join(
                    x
                    for x in [
                        run.message,
                        quafu_msg if use_quafu else "",
                        capacity_msg,
                    ]
                    if x
                )
            ),
            quantum_task_id=quafu_task_id,
            quantum_backend=quafu_backend_used,
            quantum_bitstring=quafu_bitstring,
            quantum_endpoint=quafu_endpoint,
            routing_method=routing_method,
        ),
    )


def _fit_bitstring(bitstring: str, num_qubits: int) -> str:
    cleaned, _message = _normalize_bitstring(bitstring)
    if not cleaned:
        return ""
    return cleaned[-num_qubits:].zfill(num_qubits)


def _run_hybrid_optimization(
    instance: DispatchInstance,
    mode: str,
    time_limit: int,
    two_opt_rounds: int,
    quafu_token: str,
    quafu_backend: str,
    quafu_base_url: str,
    quafu_shots: int,
    quafu_wait: bool,
    quafu_timeout_sec: int,
    quafu_proxy_url: str,
    quafu_verify_ssl: bool,
    quafu_result_task_id: str,
    quafu_manual_bitstring: str,
    auto_repair_capacity: bool,
    tabu_iterations: int,
    tabu_tenure: int,
    qaoa_subproblem_size: int,
    qaoa_max_edges: int,
    qaoa_gamma: float,
    qaoa_beta: float,
    clustering_seed: int,
    routing_method: str,
) -> OptimizationResult:
    qaoa_subproblem_size = min(
        max(2, int(qaoa_subproblem_size)),
        6,
        max(2, len(instance.customers)),
    )

    capacity_msg = ""
    if not instance.feasible_capacity:
        if not auto_repair_capacity:
            min_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
            raise ValueError(
                "Total demand exceeds fleet capacity. "
                f"Required minimum capacity per vehicle is {min_capacity} "
                f"(demand={instance.total_demand}, vehicles={instance.num_vehicles})."
            )
        repaired_capacity = math.ceil(instance.total_demand / instance.num_vehicles)
        instance = DispatchInstance(
            depot=instance.depot,
            customers=instance.customers,
            num_vehicles=instance.num_vehicles,
            vehicle_capacity=repaired_capacity,
        )
        capacity_msg = f"Capacity auto-repaired to {repaired_capacity}."

    clustering = capacity_constrained_kmeans(
        instance=instance,
        seed=clustering_seed,
    )
    manual_bitstring, manual_message = _normalize_bitstring(
        quafu_manual_bitstring
    )
    task_bitstring = ""
    lookup_message = ""
    lookup_backend = None
    lookup_endpoint = None
    requested_task_id = str(quafu_result_task_id or "").strip()
    if requested_task_id and quafu_token:
        lookup = fetch_quafu_task_bitstring(
            task_id=requested_task_id,
            api_token=quafu_token,
            base_url=quafu_base_url or None,
            timeout_sec=quafu_timeout_sec,
            proxy_url=quafu_proxy_url,
            verify_ssl=quafu_verify_ssl,
        )
        task_bitstring = lookup.bitstring or ""
        lookup_message = lookup.message
        lookup_backend = lookup.backend
        lookup_endpoint = lookup.endpoint

    def sampler(problem, circuit, iteration):
        fitted_manual = _fit_bitstring(manual_bitstring, problem.num_qubits)
        if fitted_manual:
            return QuantumSample(
                bitstring=fitted_manual,
                source="manual_bitstring",
                message="Manual bitstring override applied.",
            )

        fitted_task = _fit_bitstring(task_bitstring, problem.num_qubits)
        if fitted_task:
            return QuantumSample(
                bitstring=fitted_task,
                source="quafu_existing_task",
                task_id=requested_task_id or None,
                backend=lookup_backend,
                endpoint=lookup_endpoint,
                message=lookup_message,
                counts=lookup.counts,
            )

        if quafu_token:
            job = submit_quafu_qasm_job(
                qasm=circuit.qasm,
                selected_customers=problem.customers,
                api_token=quafu_token,
                backend=quafu_backend or None,
                base_url=quafu_base_url or None,
                shots=quafu_shots,
                wait=quafu_wait,
                timeout_sec=quafu_timeout_sec,
                proxy_url=quafu_proxy_url,
                verify_ssl=quafu_verify_ssl,
            )
            bitstring = _fit_bitstring(
                job.bitstring or "",
                problem.num_qubits,
            )
            if bitstring:
                return QuantumSample(
                    bitstring=bitstring,
                    source="quafu_hardware",
                    task_id=job.task_id,
                    backend=job.backend,
                    endpoint=job.endpoint,
                    message=job.message,
                    counts=job.counts,
                )
            return QuantumSample(
                bitstring=solve_maxcut_exact(problem),
                source=(
                    "quafu_submitted_local_fallback"
                    if job.ok
                    else "quafu_unavailable_local_fallback"
                ),
                task_id=job.task_id,
                backend=job.backend,
                endpoint=job.endpoint,
                message=job.message,
                counts=job.counts,
            )

        return QuantumSample(
            bitstring=solve_maxcut_exact(problem),
            source="local_exact_maxcut",
            message="Quafu token is empty; used local exact Max-Cut fallback.",
        )

    tabu_result = run_tabu_qaoa(
        assignments=clustering.assignments,
        depot=instance.depot,
        vehicle_capacity=instance.vehicle_capacity,
        iterations=tabu_iterations,
        subproblem_size=qaoa_subproblem_size,
        max_edges=min(max(1, int(qaoa_max_edges)), 10),
        gamma=qaoa_gamma,
        beta=qaoa_beta,
        tabu_tenure=tabu_tenure,
        seed=clustering_seed,
        sampler=sampler,
    )
    assignments = _repair_capacity(
        tabu_result.assignments,
        vehicle_capacity=instance.vehicle_capacity,
    )
    route_method = (routing_method or "ortools").lower().strip()
    routes = build_route_plans(
        assignments=assignments,
        depot=instance.depot,
        two_opt_rounds=two_opt_rounds,
        routing_method=route_method,
        ortools_time_limit_sec=max(
            1,
            int(time_limit) // max(1, instance.num_vehicles),
        ),
        ortools_fallback=True,
    )
    total_distance = sum(route.distance for route in routes)

    records = tabu_result.records
    accepted_moves = sum(1 for record in records if record.accepted)
    hardware_records = [
        record
        for record in records
        if record.sampler_source in {
            "quafu_hardware",
            "quafu_existing_task",
        }
    ]
    submitted_records = [
        record
        for record in records
        if record.task_id
    ]
    last_record = records[-1] if records else None
    last_quantum_record = (
        hardware_records[-1]
        if hardware_records
        else submitted_records[-1]
        if submitted_records
        else last_record
    )
    if hardware_records:
        used_mode = "tabu_qaoa_quafu_hybrid"
    elif submitted_records:
        used_mode = "tabu_qaoa_quafu_submitted_local_fallback"
    elif manual_bitstring:
        used_mode = "tabu_qaoa_manual_hybrid"
    else:
        used_mode = "tabu_qaoa_local_fallback"

    record_messages = []
    for record in records:
        if record.message and record.message not in record_messages:
            record_messages.append(record.message)
        if len(record_messages) >= 2:
            break
    message_parts = [
        f"Clustering={clustering.method}.",
        (
            f"QAOA window=adaptive, "
            f"max customers={qaoa_subproblem_size}."
        ),
        (
            f"Tabu-QAOA iterations={len(records)}, accepted={accepted_moves}, "
            f"proxy distance {tabu_result.initial_distance:.3f}"
            f" -> {tabu_result.best_distance:.3f}."
        ),
        f"Routing={route_method} (heuristic fallback enabled).",
        manual_message,
        capacity_msg,
        *record_messages,
    ]
    iteration_records = [
        {
            "iteration": record.iteration,
            "vehicle_pair": list(record.vehicle_pair),
            "vehicles": list(record.vehicle_pair),
            "selected_customer_ids": record.selected_customer_ids,
            "sub_k": record.sub_k,
            "n_edges": record.n_edges,
            "sub_cnot": record.sub_cnot,
            "bitstring": record.bitstring,
            "sampler_source": record.sampler_source,
            "accepted": record.accepted,
            "cut_before": record.cut_before,
            "cut_proposed": record.cut_proposed,
            "cut_after": record.cut_after,
            "classical_bitstring": record.classical_bitstring,
            "classical_cut": record.classical_cut,
            "proxy_before": record.proxy_before,
            "proxy_after": record.proxy_after,
            "candidate_distance": record.candidate_distance,
            "classical_candidate_distance": (
                record.classical_candidate_distance
            ),
            "quantum_improvement": record.quantum_improvement,
            "classical_improvement": record.classical_improvement,
            "paired_improvement_delta": (
                record.paired_improvement_delta
            ),
            "best_distance": record.best_distance,
            "task_id": record.task_id,
            "backend": record.backend,
            "endpoint": record.endpoint,
            "message": record.message,
            "counts": record.counts,
            "qasm": record.qasm,
            "edges": [
                {
                    "left": edge[0],
                    "right": edge[1],
                    "weight": edge[2],
                }
                for edge in (record.edges or [])
            ],
            "cnot_count": record.cnot_count,
        }
        for record in records
    ]

    return OptimizationResult(
        instance=instance,
        assignments=assignments,
        routes=routes,
        total_distance=total_distance,
        metadata=AssignmentMetadata(
            requested_mode=mode,
            used_mode=used_mode,
            energy=tabu_result.best_distance,
            message=" | ".join(part for part in message_parts if part),
            quantum_task_id=(
                last_quantum_record.task_id
                if last_quantum_record
                else requested_task_id or None
            ),
            quantum_backend=(
                last_quantum_record.backend
                if last_quantum_record
                else lookup_backend
            ),
            quantum_bitstring=(
                last_quantum_record.bitstring
                if last_quantum_record
                else manual_bitstring or task_bitstring or None
            ),
            quantum_endpoint=(
                last_quantum_record.endpoint
                if last_quantum_record
                else lookup_endpoint
            ),
            quantum_task_ids=tabu_result.quantum_task_ids,
            clustering_method=clustering.method,
            routing_method=route_method,
            qaoa_subproblem_policy="adaptive",
            qaoa_subproblem_size=qaoa_subproblem_size,
            tabu_iterations=len(records),
            tabu_accepted_moves=accepted_moves,
            initial_proxy_distance=tabu_result.initial_distance,
            best_proxy_distance=tabu_result.best_distance,
            quantum_counts=(
                last_quantum_record.counts
                if last_quantum_record
                else None
            ),
            quantum_iteration_records=iteration_records,
            stratified_ablation=summarize_paired_ablation(records),
        ),
    )


def run_optimization(
    instance: DispatchInstance,
    mode: str,
    time_limit: int = 8,
    num_reads: int = 300,
    num_sweeps: int = 40,
    compactness_weight: float = 1.0,
    assignment_penalty: float = 90.0,
    capacity_penalty: float = 45.0,
    balance_weight: float = 1.2,
    two_opt_rounds: int = 2,
    quafu_token: str = "",
    quafu_backend: str = "",
    quafu_base_url: str = "",
    quafu_shots: int = 1024,
    quafu_wait: bool = True,
    quafu_max_qubits: int = 6,
    quafu_timeout_sec: int = 25,
    quafu_proxy_url: str = "",
    quafu_verify_ssl: bool = True,
    quafu_result_task_id: str = "",
    quafu_manual_bitstring: str = "",
    auto_repair_capacity: bool = False,
    tabu_iterations: int = 20,
    tabu_tenure: int = 5,
    qaoa_subproblem_size: int = 6,
    qaoa_max_edges: int = 10,
    qaoa_gamma: float = 1.1,
    qaoa_beta: float = 0.8,
    clustering_seed: int = 2026,
    routing_method: str = "ortools",
) -> OptimizationResult:
    """Run the classical BQM baseline or the capacity-clustered Tabu-QAOA flow."""
    normalized_mode = (mode or "classical").lower().strip()
    if normalized_mode in {
        "quantum",
        "hybrid",
        "tabu_qaoa",
        "hybrid_local",
    }:
        return _run_hybrid_optimization(
            instance=instance,
            mode=normalized_mode,
            time_limit=time_limit,
            two_opt_rounds=two_opt_rounds,
            quafu_token=(
                ""
                if normalized_mode == "hybrid_local"
                else quafu_token
            ),
            quafu_backend=quafu_backend,
            quafu_base_url=quafu_base_url,
            quafu_shots=quafu_shots,
            quafu_wait=quafu_wait,
            quafu_timeout_sec=quafu_timeout_sec,
            quafu_proxy_url=quafu_proxy_url,
            quafu_verify_ssl=quafu_verify_ssl,
            quafu_result_task_id=quafu_result_task_id,
            quafu_manual_bitstring=quafu_manual_bitstring,
            auto_repair_capacity=auto_repair_capacity,
            tabu_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            qaoa_subproblem_size=max(
                2,
                min(
                    int(qaoa_subproblem_size),
                    int(quafu_max_qubits),
                    6,
                ),
            ),
            qaoa_max_edges=qaoa_max_edges,
            qaoa_gamma=qaoa_gamma,
            qaoa_beta=qaoa_beta,
            clustering_seed=clustering_seed,
            routing_method=routing_method,
        )

    return _run_legacy_optimization(
        instance=instance,
        mode=normalized_mode,
        time_limit=time_limit,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        compactness_weight=compactness_weight,
        assignment_penalty=assignment_penalty,
        capacity_penalty=capacity_penalty,
        balance_weight=balance_weight,
        two_opt_rounds=two_opt_rounds,
        quafu_token=quafu_token,
        quafu_backend=quafu_backend,
        quafu_base_url=quafu_base_url,
        quafu_shots=quafu_shots,
        quafu_wait=quafu_wait,
        quafu_max_qubits=quafu_max_qubits,
        quafu_timeout_sec=quafu_timeout_sec,
        quafu_proxy_url=quafu_proxy_url,
        quafu_verify_ssl=quafu_verify_ssl,
        quafu_result_task_id=quafu_result_task_id,
        quafu_manual_bitstring=quafu_manual_bitstring,
        auto_repair_capacity=auto_repair_capacity,
        routing_method=routing_method,
    )

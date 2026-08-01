from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization  # noqa: E402
from quantum_route_forge.env import quafu_token as load_quafu_token  # noqa: E402


def _emit_payload(payload: dict, output_json: str = "") -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    output_json = str(output_json or "").strip()
    if output_json:
        output_path = Path(output_json)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _run_once(
    seed: int,
    customers: int,
    vehicles: int,
    capacity: int,
    mode: str,
    time_limit: int,
    num_reads: int,
    quafu_token: str,
    quafu_backend: str,
    quafu_base_url: str,
    quafu_shots: int,
    quafu_wait: bool,
    quafu_max_qubits: int,
    quafu_timeout_sec: int,
    quafu_proxy_url: str,
    quafu_verify_ssl: bool,
    quafu_result_task_id: str,
    quafu_manual_bitstring: str,
    tabu_iterations: int,
    tabu_tenure: int,
    qaoa_subproblem_size: int,
    qaoa_max_edges: int,
    qaoa_gamma: float,
    qaoa_beta: float,
    routing_method: str,
) -> dict:
    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    result = run_optimization(
        instance=instance,
        mode=mode,
        time_limit=time_limit,
        num_reads=num_reads,
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
        tabu_iterations=tabu_iterations,
        tabu_tenure=tabu_tenure,
        qaoa_subproblem_size=qaoa_subproblem_size,
        qaoa_max_edges=qaoa_max_edges,
        qaoa_gamma=qaoa_gamma,
        qaoa_beta=qaoa_beta,
        clustering_seed=seed,
        routing_method=routing_method,
    )
    payload = {
        "mode_requested": mode,
        "mode_used": result.metadata.used_mode,
        "energy": result.metadata.energy,
        "message": result.metadata.message,
        "quantum_task_id": result.metadata.quantum_task_id,
        "quantum_backend": result.metadata.quantum_backend,
        "quantum_bitstring": result.metadata.quantum_bitstring,
        "quantum_endpoint": result.metadata.quantum_endpoint,
        "quantum_task_ids": result.metadata.quantum_task_ids or [],
        "clustering_method": result.metadata.clustering_method,
        "routing_method": result.metadata.routing_method,
        "qaoa_subproblem_policy": (
            result.metadata.qaoa_subproblem_policy
        ),
        "qaoa_subproblem_size": result.metadata.qaoa_subproblem_size,
        "tabu_iterations": result.metadata.tabu_iterations,
        "tabu_accepted_moves": result.metadata.tabu_accepted_moves,
        "initial_proxy_distance": result.metadata.initial_proxy_distance,
        "best_proxy_distance": result.metadata.best_proxy_distance,
        "quantum_counts": result.metadata.quantum_counts or {},
        "quantum_iteration_records": (
            result.metadata.quantum_iteration_records or []
        ),
        "stratified_ablation": (
            result.metadata.stratified_ablation or {}
        ),
        "num_customers": customers,
        "num_vehicles": vehicles,
        "vehicle_capacity": capacity,
        "total_demand": instance.total_demand,
        "total_distance": result.total_distance,
        "routes": [
            {
                "vehicle_id": route.vehicle_id,
                "stops": len(route.customers),
                "load": route.load,
                "distance": route.distance,
                "customer_ids": [c.customer_id for c in route.customers],
            }
            for route in result.routes
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Route Forge CLI runner.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--customers", type=int, default=48)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=37)
    parser.add_argument(
        "--mode",
        type=str,
        default="quantum",
        choices=["quantum", "classical", "hybrid_local"],
    )
    parser.add_argument("--time-limit", type=int, default=10)
    parser.add_argument("--num-reads", type=int, default=120)
    parser.add_argument("--quafu-token", type=str, default="")
    parser.add_argument("--quafu-backend", type=str, default="")
    parser.add_argument("--quafu-base-url", type=str, default="")
    parser.add_argument("--quafu-shots", type=int, default=1024)
    parser.add_argument("--quafu-max-qubits", type=int, default=6)
    parser.add_argument("--quafu-timeout-sec", type=int, default=25)
    parser.add_argument("--quafu-proxy-url", type=str, default="")
    parser.add_argument("--quafu-result-task-id", type=str, default="")
    parser.add_argument("--quafu-manual-bitstring", type=str, default="")
    parser.add_argument("--tabu-iterations", type=int, default=20)
    parser.add_argument("--tabu-tenure", type=int, default=5)
    parser.add_argument(
        "--qaoa-subproblem-size",
        type=int,
        default=6,
        help=(
            "maximum adaptive Max-Cut window size (2-6); "
            "smaller vehicle pairs are not filtered"
        ),
    )
    parser.add_argument("--qaoa-max-edges", type=int, default=10)
    parser.add_argument("--qaoa-gamma", type=float, default=1.1)
    parser.add_argument("--qaoa-beta", type=float, default=0.8)
    parser.add_argument(
        "--routing-method",
        type=str,
        choices=["ortools", "heuristic"],
        default="ortools",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="also write the complete structured result to this JSON path",
    )
    parser.add_argument(
        "--quafu-verify-ssl",
        type=str,
        default="true",
        choices=["true", "false"],
        help="set false only for diagnostics",
    )
    parser.add_argument(
        "--quafu-wait",
        type=str,
        default="true",
        choices=["true", "false"],
        help="wait for quantum execution result before continuing",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both quantum and classical modes on the same instance parameters.",
    )
    args = parser.parse_args()
    quafu_token = (
        (args.quafu_token or "").strip()
        or load_quafu_token()
    )
    quafu_wait = str(args.quafu_wait).lower() == "true"
    quafu_verify_ssl = str(args.quafu_verify_ssl).lower() == "true"

    try:
        if args.compare:
            q = _run_once(
                seed=args.seed,
                customers=args.customers,
                vehicles=args.vehicles,
                capacity=args.capacity,
                mode="quantum",
                time_limit=args.time_limit,
                num_reads=args.num_reads,
                quafu_token=quafu_token,
                quafu_backend=args.quafu_backend,
                quafu_base_url=args.quafu_base_url,
                quafu_shots=args.quafu_shots,
                quafu_wait=quafu_wait,
                quafu_max_qubits=args.quafu_max_qubits,
                quafu_timeout_sec=args.quafu_timeout_sec,
                quafu_proxy_url=args.quafu_proxy_url,
                quafu_verify_ssl=quafu_verify_ssl,
                quafu_result_task_id=args.quafu_result_task_id,
                quafu_manual_bitstring=args.quafu_manual_bitstring,
                tabu_iterations=args.tabu_iterations,
                tabu_tenure=args.tabu_tenure,
                qaoa_subproblem_size=args.qaoa_subproblem_size,
                qaoa_max_edges=args.qaoa_max_edges,
                qaoa_gamma=args.qaoa_gamma,
                qaoa_beta=args.qaoa_beta,
                routing_method=args.routing_method,
            )
            c = _run_once(
                seed=args.seed,
                customers=args.customers,
                vehicles=args.vehicles,
                capacity=args.capacity,
                mode="classical",
                time_limit=args.time_limit,
                num_reads=args.num_reads,
                quafu_token=quafu_token,
                quafu_backend=args.quafu_backend,
                quafu_base_url=args.quafu_base_url,
                quafu_shots=args.quafu_shots,
                quafu_wait=quafu_wait,
                quafu_max_qubits=args.quafu_max_qubits,
                quafu_timeout_sec=args.quafu_timeout_sec,
                quafu_proxy_url=args.quafu_proxy_url,
                quafu_verify_ssl=quafu_verify_ssl,
                quafu_result_task_id=args.quafu_result_task_id,
                quafu_manual_bitstring=args.quafu_manual_bitstring,
                tabu_iterations=args.tabu_iterations,
                tabu_tenure=args.tabu_tenure,
                qaoa_subproblem_size=args.qaoa_subproblem_size,
                qaoa_max_edges=args.qaoa_max_edges,
                qaoa_gamma=args.qaoa_gamma,
                qaoa_beta=args.qaoa_beta,
                routing_method=args.routing_method,
            )
            _emit_payload(
                {
                    "comparison": {
                        "quantum_total_distance": q["total_distance"],
                        "classical_total_distance": c["total_distance"],
                        "delta_quantum_minus_classical": q["total_distance"] - c["total_distance"],
                    },
                    "quantum": q,
                    "classical": c,
                },
                output_json=args.output_json,
            )
            return

        payload = _run_once(
            seed=args.seed,
            customers=args.customers,
            vehicles=args.vehicles,
            capacity=args.capacity,
            mode=args.mode,
            time_limit=args.time_limit,
            num_reads=args.num_reads,
            quafu_token=quafu_token,
            quafu_backend=args.quafu_backend,
            quafu_base_url=args.quafu_base_url,
            quafu_shots=args.quafu_shots,
            quafu_wait=quafu_wait,
            quafu_max_qubits=args.quafu_max_qubits,
            quafu_timeout_sec=args.quafu_timeout_sec,
            quafu_proxy_url=args.quafu_proxy_url,
            quafu_verify_ssl=quafu_verify_ssl,
            quafu_result_task_id=args.quafu_result_task_id,
            quafu_manual_bitstring=args.quafu_manual_bitstring,
            tabu_iterations=args.tabu_iterations,
            tabu_tenure=args.tabu_tenure,
            qaoa_subproblem_size=args.qaoa_subproblem_size,
            qaoa_max_edges=args.qaoa_max_edges,
            qaoa_gamma=args.qaoa_gamma,
            qaoa_beta=args.qaoa_beta,
            routing_method=args.routing_method,
        )
        _emit_payload(payload, output_json=args.output_json)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "mode_requested": args.mode,
                    "error": str(exc),
                    "hint": "Increase --capacity or --vehicles to satisfy total demand.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

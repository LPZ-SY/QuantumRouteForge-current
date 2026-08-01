from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization  # noqa: E402
from quantum_route_forge.quafu_bridge import fetch_quafu_task_bitstring  # noqa: E402


def _classical_distance(seed: int, customers: int, vehicles: int, capacity: int) -> float:
    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    result = run_optimization(
        instance=instance,
        mode="classical",
        num_reads=120,
    )
    return result.total_distance


def _submit_quantum_task(
    seed: int,
    customers: int,
    vehicles: int,
    capacity: int,
    token: str,
    backend: str,
    base_url: str,
    shots: int,
    max_qubits: int,
) -> tuple[str | None, str, str]:
    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    result = run_optimization(
        instance=instance,
        mode="quantum",
        num_reads=120,
        quafu_token=token,
        quafu_backend=backend,
        quafu_base_url=base_url,
        quafu_shots=shots,
        quafu_max_qubits=max_qubits,
        quafu_wait=False,
        quafu_timeout_sec=30,
    )
    return (
        result.metadata.quantum_task_id,
        result.metadata.quantum_backend or "",
        result.metadata.message,
    )


def _hybrid_distance_from_task(
    seed: int,
    customers: int,
    vehicles: int,
    capacity: int,
    token: str,
    base_url: str,
    task_id: str,
    timeout_sec: int,
) -> tuple[bool, float | None, str | None, str]:
    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    lookup = fetch_quafu_task_bitstring(
        task_id=task_id,
        api_token=token,
        base_url=base_url,
        timeout_sec=timeout_sec,
    )
    if not lookup.bitstring:
        return False, None, None, lookup.message

    result = run_optimization(
        instance=instance,
        mode="quantum",
        num_reads=120,
        quafu_token=token,
        quafu_base_url=base_url,
        quafu_result_task_id=task_id,
        quafu_wait=False,
        quafu_timeout_sec=timeout_sec,
    )
    return (
        result.metadata.used_mode == "quafu_quantum_hybrid",
        result.total_distance,
        result.metadata.quantum_bitstring,
        result.metadata.message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Quafu A/B benchmark: classical vs quantum-hybrid.")
    parser.add_argument("--customers", type=int, default=48)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--backend", type=str, default="Dongling")
    parser.add_argument("--quafu-base-url", type=str, default="https://quafu-sqc.baqis.ac.cn/")
    parser.add_argument("--quafu-shots", type=int, default=100)
    parser.add_argument("--quafu-max-qubits", type=int, default=2)
    parser.add_argument("--poll-rounds", type=int, default=15)
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--poll-timeout-sec", type=int, default=20)
    parser.add_argument("--output-csv", type=str, default="ab_results_quafu.csv")
    args = parser.parse_args()

    token = os.getenv("QUAFU_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("QUAFU_API_TOKEN is empty.")

    seeds = [args.seed_start + i for i in range(args.num_seeds)]
    rows: list[dict] = []

    print(f"[1/3] Running classical baselines for {len(seeds)} seeds...")
    for seed in seeds:
        dist = _classical_distance(seed, args.customers, args.vehicles, args.capacity)
        rows.append(
            {
                "seed": seed,
                "classical_distance": dist,
                "task_id": "",
                "task_backend": "",
                "task_submit_message": "",
                "hybrid_ready": False,
                "hybrid_distance": math.nan,
                "quantum_bitstring": "",
                "hybrid_message": "",
            }
        )

    print("[2/3] Submitting real quantum tasks (submit-only)...")
    for row in rows:
        task_id, task_backend, message = _submit_quantum_task(
            seed=int(row["seed"]),
            customers=args.customers,
            vehicles=args.vehicles,
            capacity=args.capacity,
            token=token,
            backend=args.backend,
            base_url=args.quafu_base_url,
            shots=args.quafu_shots,
            max_qubits=args.quafu_max_qubits,
        )
        row["task_id"] = task_id or ""
        row["task_backend"] = task_backend
        row["task_submit_message"] = message
        print(f"  seed={row['seed']} task_id={row['task_id']} backend={task_backend}")

    print("[3/3] Polling for finished tasks and hybrid replay...")
    pending = {str(r["seed"]) for r in rows if r["task_id"]}
    for round_id in range(1, args.poll_rounds + 1):
        if not pending:
            break
        print(f"  poll round {round_id}/{args.poll_rounds}, pending={len(pending)}")
        for row in rows:
            seed_key = str(row["seed"])
            if seed_key not in pending:
                continue
            task_id = str(row["task_id"])
            ok, hybrid_dist, bitstring, msg = _hybrid_distance_from_task(
                seed=int(row["seed"]),
                customers=args.customers,
                vehicles=args.vehicles,
                capacity=args.capacity,
                token=token,
                base_url=args.quafu_base_url,
                task_id=task_id,
                timeout_sec=args.poll_timeout_sec,
            )
            row["hybrid_message"] = msg
            if ok and hybrid_dist is not None and bitstring:
                row["hybrid_ready"] = True
                row["hybrid_distance"] = hybrid_dist
                row["quantum_bitstring"] = bitstring
                pending.remove(seed_key)
                print(
                    f"    finished seed={row['seed']} task_id={task_id} "
                    f"bitstring={bitstring} hybrid_distance={hybrid_dist:.3f}"
                )
        if pending:
            time.sleep(max(1, args.poll_interval_sec))

    out_path = ROOT / args.output_csv
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "seed",
                "classical_distance",
                "task_id",
                "task_backend",
                "task_submit_message",
                "hybrid_ready",
                "hybrid_distance",
                "quantum_bitstring",
                "hybrid_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    completed = [r for r in rows if r["hybrid_ready"]]
    print(f"\nSaved CSV: {out_path}")
    print(f"Completed hybrid seeds: {len(completed)}/{len(rows)}")
    if completed:
        deltas = [float(r["hybrid_distance"]) - float(r["classical_distance"]) for r in completed]
        wins = sum(1 for d in deltas if d < 0)
        print(
            "Hybrid - Classical delta: "
            f"mean={statistics.mean(deltas):.3f}, median={statistics.median(deltas):.3f}, "
            f"wins={wins}/{len(deltas)}"
        )
    else:
        print("No finished hybrid samples yet (all tasks still pending or unavailable).")


if __name__ == "__main__":
    main()

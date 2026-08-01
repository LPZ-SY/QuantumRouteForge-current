from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiment_utils import (
    ROOT,
    append_jsonl,
    build_bqm_for_instance,
    compute_capacity_metrics,
    coordinates_payload,
    ensure_dir,
    evaluate_sample,
    infer_capacity_from_seed,
    make_run_id,
    parse_int_list,
    routes_payload,
    write_csv,
)

import sys

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization
from quantum_route_forge.assignment_bqm import decode_assignment
from quantum_route_forge.solvers import solve_bqm_classical


def _quafu_status_from_used_mode(used_mode: str, has_token: bool, task_id: str | None, message: str, error: str) -> str:
    used = (used_mode or "").strip().lower()
    msg = (message or "").lower()
    if not has_token:
        return "token_missing"
    if error:
        return "failed"
    if "quafu_quantum_hybrid" in used:
        return "bitstring_assignment_refine"
    if "quafu_submitted_classical_refine" in used:
        return "submitted"
    if "fallback" in used:
        return "fallback"
    if task_id or "submitted to quafu" in msg:
        return "submitted"
    return "unknown"


def _serialize_route_summary(routes: list[dict[str, Any]]) -> str:
    brief = [
        {
            "vehicle_id": r.get("vehicle_id"),
            "stops": r.get("stops"),
            "load": r.get("load"),
            "distance": round(float(r.get("distance", 0.0)), 4),
        }
        for r in routes
    ]
    return json.dumps(brief, ensure_ascii=False)


def _run_random_mode(instance, bqm, seed: int, random_reads: int) -> tuple[dict[str, Any], float, float]:
    import random

    t0 = time.perf_counter()
    rng = random.Random(seed)
    best_sample = None
    best_energy = None
    for _ in range(max(1, random_reads)):
        sample = {str(v): int(rng.randint(0, 1)) for v in bqm.variables}
        energy = float(bqm.energy(sample))
        if best_energy is None or energy < best_energy:
            best_sample = sample
            best_energy = energy
    assign_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_info = evaluate_sample(instance=instance, bqm=bqm, sample=best_sample, two_opt_rounds=2)
    routing_runtime = time.perf_counter() - t1

    return eval_info, assign_runtime, routing_runtime


def _run_classical_mode(instance, bqm, num_reads: int) -> tuple[dict[str, Any], float, float]:
    t0 = time.perf_counter()
    run = solve_bqm_classical(bqm=bqm, num_reads=num_reads, num_sweeps=40)
    assign_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_info = evaluate_sample(instance=instance, bqm=bqm, sample=run.sample, two_opt_rounds=2)
    routing_runtime = time.perf_counter() - t1
    return eval_info, assign_runtime, routing_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="运行比赛实验（quick/full）")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--with-quafu", action="store_true")
    parser.add_argument("--no-quafu", action="store_true")
    parser.add_argument("--customers", type=str, default="")
    parser.add_argument("--vehicles", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results/competition_runs")
    parser.add_argument("--num-reads", type=int, default=120)
    parser.add_argument("--random-reads", type=int, default=120)
    parser.add_argument("--quafu-shots", type=int, default=100)
    parser.add_argument("--quafu-timeout-sec", type=int, default=25)
    args = parser.parse_args()

    if args.full:
        default_customers = [12, 24, 36, 48, 60]
        default_vehicles = [2, 3, 4]
        default_seeds = [2026, 2027, 2028, 2029, 2030]
        default_modes = ["classical", "random", "quantum"]
    else:
        default_customers = [12, 24, 48]
        default_vehicles = [2, 4]
        default_seeds = [2026, 2027]
        default_modes = ["classical", "random"]

    customers_list = parse_int_list(args.customers, default_customers)
    vehicles_list = parse_int_list(args.vehicles, default_vehicles)
    seeds = parse_int_list(args.seeds, default_seeds)

    token = (os.getenv("QUAFU_API_TOKEN", "") or "").strip()
    has_token = bool(token)
    quafu_base = (os.getenv("QUAFU_BASE_URL", "") or "").strip() or "https://quafu-sqc.baqis.ac.cn/"

    modes = list(default_modes)
    if args.no_quafu:
        modes = [m for m in modes if m != "quantum"]
    elif args.with_quafu:
        if "quantum" not in modes:
            modes.append("quantum")
    else:
        # 默认沿用 quick/full 内置列表。
        pass

    outdir = ensure_dir(Path(args.outdir))
    csv_path = outdir / "results.csv"
    jsonl_path = outdir / "results.jsonl"

    rows: list[dict[str, Any]] = []

    total = len(customers_list) * len(vehicles_list) * len(seeds) * len(modes)
    idx = 0

    for seed in seeds:
        for customers in customers_list:
            for vehicles in vehicles_list:
                capacity = infer_capacity_from_seed(seed=seed, customers=customers, vehicles=vehicles)
                instance = generate_dispatch_instance(
                    seed=seed,
                    num_customers=customers,
                    num_vehicles=vehicles,
                    vehicle_capacity=capacity,
                )
                bqm = build_bqm_for_instance(instance)
                num_vars = int(len(bqm.variables))
                num_quads = int(len(bqm.quadratic))

                for mode in modes:
                    idx += 1
                    print(f"[{idx}/{total}] seed={seed} customers={customers} vehicles={vehicles} mode={mode}")

                    run_id = make_run_id("comp")
                    row = {
                        "run_id": run_id,
                        "timestamp": "",
                        "seed": seed,
                        "customers": customers,
                        "vehicles": vehicles,
                        "capacity": capacity,
                        "mode": mode,
                        "requested_solver": mode,
                        "used_solver": "",
                        "quafu_backend": "",
                        "quafu_task_id": "",
                        "quafu_endpoint": "",
                        "quafu_status": "",
                        "is_quafu_submitted": False,
                        "is_fallback": False,
                        "is_token_missing": False,
                        "bqm_energy": "",
                        "total_route_distance": "",
                        "feasible": False,
                        "assignment_feasible": False,
                        "onehot_violation_count": "",
                        "capacity_violation_count": "",
                        "runtime_total_sec": "",
                        "runtime_assignment_sec": "",
                        "runtime_routing_sec": "",
                        "num_bqm_variables": num_vars,
                        "num_bqm_quadratic_terms": num_quads,
                        "route_summary": "",
                        "coordinates": json.dumps(coordinates_payload(instance), ensure_ascii=False),
                        "routes": "",
                        "error_message": "",
                    }

                    t_total_0 = time.perf_counter()
                    try:
                        if mode == "random":
                            eval_info, t_assign, t_routing = _run_random_mode(
                                instance=instance,
                                bqm=bqm,
                                seed=seed + customers * 101 + vehicles * 17,
                                random_reads=args.random_reads,
                            )
                            row["timestamp"] = eval_info.get("timestamp", "") or ""
                            row["used_solver"] = "random_best_of_reads"
                            row["bqm_energy"] = float(eval_info["assignment_energy"])
                            row["total_route_distance"] = float(eval_info["route_distance_after_2opt"])
                            row["assignment_feasible"] = bool(eval_info["assignment_feasible"])
                            row["feasible"] = bool(eval_info["assignment_feasible"] and eval_info["route_feasible"])
                            row["onehot_violation_count"] = int(eval_info["onehot_violation_count"])
                            row["capacity_violation_count"] = int(eval_info["capacity_violation_count"])
                            row["runtime_assignment_sec"] = round(float(t_assign), 6)
                            row["runtime_routing_sec"] = round(float(t_routing), 6)
                            row["routes"] = json.dumps(eval_info["routes_after_2opt"], ensure_ascii=False)
                            row["route_summary"] = _serialize_route_summary(eval_info["routes_after_2opt"])

                        elif mode == "classical":
                            eval_info, t_assign, t_routing = _run_classical_mode(
                                instance=instance,
                                bqm=bqm,
                                num_reads=args.num_reads,
                            )
                            row["used_solver"] = "classical_simulated_annealing"
                            row["bqm_energy"] = float(eval_info["assignment_energy"])
                            row["total_route_distance"] = float(eval_info["route_distance_after_2opt"])
                            row["assignment_feasible"] = bool(eval_info["assignment_feasible"])
                            row["feasible"] = bool(eval_info["assignment_feasible"] and eval_info["route_feasible"])
                            row["onehot_violation_count"] = int(eval_info["onehot_violation_count"])
                            row["capacity_violation_count"] = int(eval_info["capacity_violation_count"])
                            row["runtime_assignment_sec"] = round(float(t_assign), 6)
                            row["runtime_routing_sec"] = round(float(t_routing), 6)
                            row["routes"] = json.dumps(eval_info["routes_after_2opt"], ensure_ascii=False)
                            row["route_summary"] = _serialize_route_summary(eval_info["routes_after_2opt"])

                        elif mode == "quantum":
                            if not has_token:
                                row["used_solver"] = "token_missing"
                                row["is_token_missing"] = True
                                row["quafu_status"] = "token_missing"
                                row["error_message"] = "QUAFU_API_TOKEN is missing"
                            else:
                                t0 = time.perf_counter()
                                result = run_optimization(
                                    instance=instance,
                                    mode="quantum",
                                    time_limit=10,
                                    num_reads=args.num_reads,
                                    quafu_token=token,
                                    quafu_backend="Baihua",
                                    quafu_base_url=quafu_base,
                                    quafu_shots=max(100, int(args.quafu_shots)),
                                    quafu_wait=False,
                                    quafu_max_qubits=min(8, customers),
                                    quafu_timeout_sec=int(args.quafu_timeout_sec),
                                    quafu_proxy_url=(os.getenv("QUAFU_PROXY_URL", "") or "").strip(),
                                    quafu_verify_ssl=True,
                                )
                                t_assign = time.perf_counter() - t0

                                t1 = time.perf_counter()
                                # 路由在 run_optimization 内已完成，这里只做开销测量拆分。
                                _ = routes_payload(result.routes)
                                t_routing = time.perf_counter() - t1

                                row["used_solver"] = str(result.metadata.used_mode)
                                row["bqm_energy"] = float(result.metadata.energy)
                                row["total_route_distance"] = float(result.total_distance)

                                onehot_violation_count = 0
                                cap_viol, _over, _loads = compute_capacity_metrics(
                                    {r.vehicle_id - 1: list(r.customers) for r in result.routes},
                                    instance.vehicle_capacity,
                                )
                                row["onehot_violation_count"] = int(onehot_violation_count)
                                row["capacity_violation_count"] = int(cap_viol)
                                row["assignment_feasible"] = bool(cap_viol == 0)
                                row["feasible"] = bool(cap_viol == 0)

                                row["quafu_backend"] = result.metadata.quantum_backend or ""
                                row["quafu_task_id"] = result.metadata.quantum_task_id or ""
                                row["quafu_endpoint"] = result.metadata.quantum_endpoint or ""
                                row["is_quafu_submitted"] = bool(
                                    row["quafu_task_id"] or "Submitted to Quafu" in (result.metadata.message or "")
                                )
                                row["is_fallback"] = "fallback" in (result.metadata.used_mode or "").lower()
                                row["quafu_status"] = _quafu_status_from_used_mode(
                                    used_mode=result.metadata.used_mode,
                                    has_token=has_token,
                                    task_id=result.metadata.quantum_task_id,
                                    message=result.metadata.message,
                                    error="",
                                )

                                row["runtime_assignment_sec"] = round(float(t_assign), 6)
                                row["runtime_routing_sec"] = round(float(t_routing), 6)
                                r_payload = routes_payload(result.routes)
                                row["routes"] = json.dumps(r_payload, ensure_ascii=False)
                                row["route_summary"] = _serialize_route_summary(r_payload)
                        else:
                            row["used_solver"] = "unknown_mode"
                            row["error_message"] = f"Unsupported mode: {mode}"

                    except Exception as exc:
                        row["error_message"] = f"{type(exc).__name__}: {exc}"
                        row["used_solver"] = row["used_solver"] or "failed"
                        if mode == "quantum":
                            row["quafu_status"] = "failed"

                    row["timestamp"] = row["timestamp"] or time.strftime("%Y-%m-%dT%H:%M:%S")
                    runtime_total = time.perf_counter() - t_total_0
                    row["runtime_total_sec"] = round(float(runtime_total), 6)

                    if row["runtime_assignment_sec"] == "":
                        row["runtime_assignment_sec"] = round(float(runtime_total), 6)
                    if row["runtime_routing_sec"] == "":
                        row["runtime_routing_sec"] = 0.0

                    if mode != "quantum":
                        row["quafu_status"] = row["quafu_status"] or "not_applicable"

                    rows.append(row)
                    append_jsonl(jsonl_path, row)

    write_csv(csv_path, rows)
    print(f"\n写入完成: {csv_path}")
    print(f"写入完成: {jsonl_path}")


if __name__ == "__main__":
    main()

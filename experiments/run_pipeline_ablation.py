from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

from experiment_utils import (
    ROOT,
    append_jsonl,
    build_bqm_for_instance,
    ensure_dir,
    evaluate_sample,
    has_quafu_bitstrings_available,
    infer_capacity_from_seed,
    parse_int_list,
    read_quafu_counts_from_raw_result,
    write_csv,
)

import sys

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization
from quantum_route_forge.solvers import solve_bqm_classical


def _pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100.0


def _status_from_used_mode(used_mode: str, has_token: bool) -> str:
    if not has_token:
        return "token_missing"
    used = (used_mode or "").lower()
    if "quafu_submitted_classical_refine" in used:
        return "submitted"
    if "quafu_quantum_hybrid" in used:
        return "bitstring_used"
    if "fallback" in used:
        return "fallback"
    if "classical" in used:
        return "fallback"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="优化流程消融实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--customers", type=str, default="")
    parser.add_argument("--vehicles", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results/pipeline_ablation")
    args = parser.parse_args()

    if args.full:
        default_customers = [12, 24, 36, 48, 60]
        default_vehicles = [2, 3, 4]
        default_seeds = [2026, 2027, 2028, 2029, 2030]
    else:
        default_customers = [12, 24, 48]
        default_vehicles = [2, 4]
        default_seeds = [2026, 2027]

    customers_list = parse_int_list(args.customers, default_customers)
    vehicles_list = parse_int_list(args.vehicles, default_vehicles)
    seeds = parse_int_list(args.seeds, default_seeds)

    outdir = ensure_dir(Path(args.outdir))
    csv_path = outdir / "pipeline_ablation.csv"
    jsonl_path = outdir / "pipeline_ablation.jsonl"

    rows: list[dict[str, Any]] = []

    token = (os.getenv("QUAFU_API_TOKEN", "") or "").strip()
    has_token = bool(token)
    quafu_base = (os.getenv("QUAFU_BASE_URL", "") or "").strip() or "https://quafu-sqc.baqis.ac.cn/"

    quafu_available = has_quafu_bitstrings_available()
    quafu_counts = read_quafu_counts_from_raw_result() if quafu_available else {}

    total = len(customers_list) * len(vehicles_list) * len(seeds)
    idx = 0

    for seed in seeds:
        for customers in customers_list:
            for vehicles in vehicles_list:
                idx += 1
                print(f"[{idx}/{total}] seed={seed} customers={customers} vehicles={vehicles}")
                capacity = infer_capacity_from_seed(seed=seed, customers=customers, vehicles=vehicles)
                instance = generate_dispatch_instance(
                    seed=seed,
                    num_customers=customers,
                    num_vehicles=vehicles,
                    vehicle_capacity=capacity,
                )
                bqm = build_bqm_for_instance(instance)

                # 1) random assignment only
                rng = random.Random(seed * 100 + customers + vehicles)
                random_sample = {str(v): int(rng.randint(0, 1)) for v in bqm.variables}
                random_eval = evaluate_sample(instance=instance, bqm=bqm, sample=random_sample, two_opt_rounds=2)

                row1 = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "capacity": capacity,
                    "method": "random_assignment_only",
                    "assignment_energy": float(random_eval["assignment_energy"]),
                    "assignment_feasible": bool(random_eval["assignment_feasible"]),
                    "route_distance_before_refine": float(random_eval["route_distance_before_2opt"]),
                    "route_distance_after_refine": float(random_eval["route_distance_before_2opt"]),
                    "improvement_percent": 0.0,
                    "quafu_status": "not_applicable",
                    "bitstring_available": False,
                }
                rows.append(row1)
                append_jsonl(jsonl_path, row1)

                # 2) random assignment + route refinement
                row2 = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "capacity": capacity,
                    "method": "random_assignment_refine",
                    "assignment_energy": float(random_eval["assignment_energy"]),
                    "assignment_feasible": bool(random_eval["assignment_feasible"]),
                    "route_distance_before_refine": float(random_eval["route_distance_before_2opt"]),
                    "route_distance_after_refine": float(random_eval["route_distance_after_2opt"]),
                    "improvement_percent": _pct(
                        float(random_eval["route_distance_before_2opt"]),
                        float(random_eval["route_distance_after_2opt"]),
                    ),
                    "quafu_status": "not_applicable",
                    "bitstring_available": False,
                }
                rows.append(row2)
                append_jsonl(jsonl_path, row2)

                # 3/4) classical BQM
                c_run = solve_bqm_classical(bqm=bqm, num_reads=120, num_sweeps=40)
                c_eval = evaluate_sample(instance=instance, bqm=bqm, sample=c_run.sample, two_opt_rounds=2)

                row3 = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "capacity": capacity,
                    "method": "classical_bqm_assignment_only",
                    "assignment_energy": float(c_eval["assignment_energy"]),
                    "assignment_feasible": bool(c_eval["assignment_feasible"]),
                    "route_distance_before_refine": float(c_eval["route_distance_before_2opt"]),
                    "route_distance_after_refine": float(c_eval["route_distance_before_2opt"]),
                    "improvement_percent": 0.0,
                    "quafu_status": "not_applicable",
                    "bitstring_available": False,
                }
                rows.append(row3)
                append_jsonl(jsonl_path, row3)

                row4 = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "capacity": capacity,
                    "method": "classical_bqm_assignment_refine",
                    "assignment_energy": float(c_eval["assignment_energy"]),
                    "assignment_feasible": bool(c_eval["assignment_feasible"]),
                    "route_distance_before_refine": float(c_eval["route_distance_before_2opt"]),
                    "route_distance_after_refine": float(c_eval["route_distance_after_2opt"]),
                    "improvement_percent": _pct(
                        float(c_eval["route_distance_before_2opt"]),
                        float(c_eval["route_distance_after_2opt"]),
                    ),
                    "quafu_status": "not_applicable",
                    "bitstring_available": False,
                }
                rows.append(row4)
                append_jsonl(jsonl_path, row4)

                # 5) Quafu submission only
                if has_token:
                    try:
                        q_result = run_optimization(
                            instance=instance,
                            mode="quantum",
                            time_limit=8,
                            num_reads=80,
                            quafu_token=token,
                            quafu_backend="Baihua",
                            quafu_base_url=quafu_base,
                            quafu_shots=100,
                            quafu_wait=False,
                            quafu_max_qubits=min(8, customers),
                            quafu_timeout_sec=25,
                            quafu_proxy_url=(os.getenv("QUAFU_PROXY_URL", "") or "").strip(),
                            quafu_verify_ssl=True,
                        )
                        q_status = _status_from_used_mode(q_result.metadata.used_mode, has_token=True)
                    except Exception:
                        q_status = "failed"
                else:
                    q_status = "token_missing"

                row5 = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "capacity": capacity,
                    "method": "quafu_submission_only",
                    "assignment_energy": "",
                    "assignment_feasible": "",
                    "route_distance_before_refine": "",
                    "route_distance_after_refine": "",
                    "improvement_percent": "",
                    "quafu_status": q_status,
                    "bitstring_available": bool(quafu_counts),
                }
                rows.append(row5)
                append_jsonl(jsonl_path, row5)

                # 6) Quafu bitstring assignment + refinement（若可用）
                if quafu_counts:
                    x_vars = [v for v in bqm.variables if str(v).startswith("x_c")]
                    bitstring = sorted(quafu_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
                    if len(bitstring) >= len(x_vars):
                        q_sample = {str(v): 0 for v in bqm.variables}
                        for i, var in enumerate(x_vars):
                            q_sample[var] = 1 if bitstring[-1 - i] == "1" else 0
                        q_eval = evaluate_sample(instance=instance, bqm=bqm, sample=q_sample, two_opt_rounds=2)
                        row6 = {
                            "seed": seed,
                            "customers": customers,
                            "vehicles": vehicles,
                            "capacity": capacity,
                            "method": "quafu_bitstring_assignment_refine",
                            "assignment_energy": float(q_eval["assignment_energy"]),
                            "assignment_feasible": bool(q_eval["assignment_feasible"]),
                            "route_distance_before_refine": float(q_eval["route_distance_before_2opt"]),
                            "route_distance_after_refine": float(q_eval["route_distance_after_2opt"]),
                            "improvement_percent": _pct(
                                float(q_eval["route_distance_before_2opt"]),
                                float(q_eval["route_distance_after_2opt"]),
                            ),
                            "quafu_status": "bitstring_used",
                            "bitstring_available": True,
                        }
                        rows.append(row6)
                        append_jsonl(jsonl_path, row6)

    write_csv(csv_path, rows)
    print(f"\n写入完成: {csv_path}")
    print(f"写入完成: {jsonl_path}")


if __name__ == "__main__":
    main()

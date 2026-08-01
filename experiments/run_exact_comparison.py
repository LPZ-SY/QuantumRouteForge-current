from __future__ import annotations

import argparse
import itertools
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

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.assignment_bqm import assignment_var
from quantum_route_forge.solvers import solve_bqm_classical


def _exact_best_over_assignments(bqm, customer_ids: list[int], vehicles: int):
    import dimod

    best_energy = None
    best_assignment = None

    for choices in itertools.product(range(vehicles), repeat=len(customer_ids)):
        fixed = bqm.copy()
        sample_fixed = {}
        for idx, cid in enumerate(customer_ids):
            chosen_v = choices[idx]
            for v in range(vehicles):
                var = assignment_var(cid, v)
                val = 1 if v == chosen_v else 0
                sample_fixed[var] = val
                if var in fixed.variables:
                    fixed.fix_variable(var, val)

        if len(fixed.variables) > 24:
            # 保险阈值，避免 slack 变量过多导致枚举爆炸。
            continue

        if len(fixed.variables) == 0:
            energy = float(fixed.offset)
        else:
            ss = dimod.ExactSolver().sample(fixed)
            energy = float(ss.first.energy)

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_assignment = sample_fixed

    return best_energy, best_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="小规模精确最优对照实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--customers", type=str, default="")
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results/exact_runs")
    parser.add_argument("--num-reads", type=int, default=200)
    args = parser.parse_args()

    if args.full:
        default_customers = [6, 8, 10]
        default_seeds = [2026, 2027, 2028]
    else:
        default_customers = [6, 8, 10]
        default_seeds = [2026, 2027, 2028]

    customers_list = parse_int_list(args.customers, default_customers)
    seeds = parse_int_list(args.seeds, default_seeds)
    vehicles = int(args.vehicles)

    outdir = ensure_dir(Path(args.outdir))
    csv_path = outdir / "exact_comparison.csv"
    jsonl_path = outdir / "exact_comparison.jsonl"

    rows: list[dict[str, Any]] = []

    quafu_available = has_quafu_bitstrings_available()
    quafu_counts = read_quafu_counts_from_raw_result() if quafu_available else {}

    total = len(customers_list) * len(seeds)
    idx = 0

    for seed in seeds:
        for customers in customers_list:
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
            customer_ids = [c.customer_id for c in instance.customers]

            exact_energy, exact_sample = _exact_best_over_assignments(
                bqm=bqm,
                customer_ids=customer_ids,
                vehicles=vehicles,
            )
            if exact_energy is None or exact_sample is None:
                row = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                    "exact_energy": "",
                    "method": "exact_optimum",
                    "method_best_energy": "",
                    "energy_gap": "",
                    "approximation_ratio": "",
                    "feasible": False,
                    "route_distance_after_refinement": "",
                    "error_message": "exact solver skipped due variable explosion",
                }
                rows.append(row)
                append_jsonl(jsonl_path, row)
                continue

            exact_eval = evaluate_sample(instance=instance, bqm=bqm, sample=exact_sample, two_opt_rounds=2)
            base_exact = {
                "seed": seed,
                "customers": customers,
                "vehicles": vehicles,
                "exact_energy": float(exact_energy),
            }

            exact_row = {
                **base_exact,
                "method": "exact_optimum",
                "method_best_energy": float(exact_energy),
                "energy_gap": 0.0,
                "approximation_ratio": 1.0,
                "feasible": bool(exact_eval["assignment_feasible"]),
                "route_distance_after_refinement": float(exact_eval["route_distance_after_2opt"]),
                "error_message": "",
            }
            rows.append(exact_row)
            append_jsonl(jsonl_path, exact_row)

            # random best
            import random

            rng = random.Random(seed * 100 + customers)
            best_random = None
            best_random_energy = None
            for _ in range(300):
                s = {str(v): int(rng.randint(0, 1)) for v in bqm.variables}
                e = float(bqm.energy(s))
                if best_random_energy is None or e < best_random_energy:
                    best_random = s
                    best_random_energy = e
            random_eval = evaluate_sample(instance=instance, bqm=bqm, sample=best_random, two_opt_rounds=2)
            random_row = {
                **base_exact,
                "method": "random_best",
                "method_best_energy": float(best_random_energy),
                "energy_gap": float(best_random_energy - exact_energy),
                "approximation_ratio": float(best_random_energy / exact_energy) if exact_energy != 0 else "",
                "feasible": bool(random_eval["assignment_feasible"]),
                "route_distance_after_refinement": float(random_eval["route_distance_after_2opt"]),
                "error_message": "",
            }
            rows.append(random_row)
            append_jsonl(jsonl_path, random_row)

            # classical SA best
            c_run = solve_bqm_classical(bqm=bqm, num_reads=max(100, args.num_reads), num_sweeps=60)
            c_eval = evaluate_sample(instance=instance, bqm=bqm, sample=c_run.sample, two_opt_rounds=2)
            c_row = {
                **base_exact,
                "method": "classical_sa_best",
                "method_best_energy": float(c_run.energy),
                "energy_gap": float(c_run.energy - exact_energy),
                "approximation_ratio": float(c_run.energy / exact_energy) if exact_energy != 0 else "",
                "feasible": bool(c_eval["assignment_feasible"]),
                "route_distance_after_refinement": float(c_eval["route_distance_after_2opt"]),
                "error_message": "",
            }
            rows.append(c_row)
            append_jsonl(jsonl_path, c_row)

            # quafu best（若可用）
            x_vars = [v for v in bqm.variables if str(v).startswith("x_c")]
            if quafu_counts:
                best_q_sample = None
                best_q_energy = None
                for bitstring, _count in sorted(quafu_counts.items(), key=lambda kv: kv[1], reverse=True):
                    if len(bitstring) < len(x_vars):
                        continue
                    sample = {str(v): 0 for v in bqm.variables}
                    for i, var in enumerate(x_vars):
                        sample[var] = 1 if bitstring[-1 - i] == "1" else 0
                    e = float(bqm.energy(sample))
                    if best_q_energy is None or e < best_q_energy:
                        best_q_energy = e
                        best_q_sample = sample

                if best_q_sample is not None:
                    q_eval = evaluate_sample(instance=instance, bqm=bqm, sample=best_q_sample, two_opt_rounds=2)
                    q_row = {
                        **base_exact,
                        "method": "quafu_best",
                        "method_best_energy": float(best_q_energy),
                        "energy_gap": float(best_q_energy - exact_energy),
                        "approximation_ratio": float(best_q_energy / exact_energy) if exact_energy != 0 else "",
                        "feasible": bool(q_eval["assignment_feasible"]),
                        "route_distance_after_refinement": float(q_eval["route_distance_after_2opt"]),
                        "error_message": "",
                    }
                    rows.append(q_row)
                    append_jsonl(jsonl_path, q_row)

    write_csv(csv_path, rows)
    print(f"\n写入完成: {csv_path}")
    print(f"写入完成: {jsonl_path}")


if __name__ == "__main__":
    main()

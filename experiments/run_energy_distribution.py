from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiment_utils import (
    ROOT,
    append_jsonl,
    build_bqm_for_instance,
    compute_capacity_metrics,
    compute_onehot_violation_count,
    ensure_dir,
    has_quafu_bitstrings_available,
    infer_capacity_from_seed,
    parse_int_list,
    read_quafu_counts_from_raw_result,
    summarize_energy,
    write_csv,
)

import sys

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.assignment_bqm import decode_assignment


def _evaluate_samples(method: str, samples: list[dict[str, int]], bqm, instance) -> dict[str, Any]:
    energies: list[float] = []
    onehot_bad = 0
    cap_bad = 0
    feasible = 0

    for sample in samples:
        energy = float(bqm.energy(sample))
        energies.append(energy)

        onehot_violation = compute_onehot_violation_count(sample, instance.customers, instance.num_vehicles)
        assignments = decode_assignment(sample, instance.customers, instance.num_vehicles)
        cap_violation, _over, _loads = compute_capacity_metrics(assignments, instance.vehicle_capacity)

        if onehot_violation > 0:
            onehot_bad += 1
        if cap_violation > 0:
            cap_bad += 1
        if onehot_violation == 0 and cap_violation == 0:
            feasible += 1

    s = summarize_energy(energies)
    n = max(1, len(samples))
    return {
        "method": method,
        "sample_count": len(samples),
        "best_energy": s["best"],
        "mean_energy": s["mean"],
        "median_energy": s["median"],
        "std_energy": s["std"],
        "feasible_sample_rate": feasible / n,
        "onehot_violation_rate": onehot_bad / n,
        "capacity_violation_rate": cap_bad / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BQM 能量分布实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--customers", type=str, default="")
    parser.add_argument("--vehicles", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="results/energy_distribution")
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
    csv_path = outdir / "energy_distribution.csv"
    jsonl_path = outdir / "energy_distribution.jsonl"

    rows: list[dict[str, Any]] = []

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
                x_vars = [v for v in bqm.variables if str(v).startswith("x_c")]

                rng = random.Random(seed * 1000 + customers * 10 + vehicles)
                random_samples = []
                for _ in range(max(20, int(args.sample_count))):
                    sample = {str(v): int(rng.randint(0, 1)) for v in bqm.variables}
                    random_samples.append(sample)

                row_random = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                }
                row_random.update(_evaluate_samples("random_bitstrings", random_samples, bqm, instance))
                rows.append(row_random)
                append_jsonl(jsonl_path, row_random)

                import dimod

                sampler = dimod.SimulatedAnnealingSampler()
                ss = sampler.sample(
                    bqm,
                    num_reads=max(20, int(args.sample_count)),
                    num_sweeps=40,
                )
                classical_samples = [dict(rec.sample) for rec in ss.data(['sample'])]
                row_classical = {
                    "seed": seed,
                    "customers": customers,
                    "vehicles": vehicles,
                }
                row_classical.update(_evaluate_samples("classical_sa", classical_samples, bqm, instance))
                rows.append(row_classical)
                append_jsonl(jsonl_path, row_classical)

                if quafu_counts:
                    quafu_samples: list[dict[str, int]] = []
                    for bitstring, _count in sorted(quafu_counts.items(), key=lambda kv: kv[1], reverse=True)[: int(args.sample_count)]:
                        # 仅当长度可覆盖 x 变量时才纳入该实例评估。
                        if len(bitstring) < len(x_vars):
                            continue
                        sample = {str(v): 0 for v in bqm.variables}
                        for i, var in enumerate(x_vars):
                            sample[var] = 1 if bitstring[-1 - i] == "1" else 0
                        quafu_samples.append(sample)
                    if quafu_samples:
                        row_quafu = {
                            "seed": seed,
                            "customers": customers,
                            "vehicles": vehicles,
                        }
                        row_quafu.update(_evaluate_samples("quafu_measured_bitstrings", quafu_samples, bqm, instance))
                        rows.append(row_quafu)
                        append_jsonl(jsonl_path, row_quafu)

    write_csv(csv_path, rows)

    note_path = outdir / "energy_distribution_note.md"
    if not quafu_counts:
        note_path.write_text(
            "当前流程尚未获得可用于闭环优化的 Quafu measured bitstrings，本实验仅比较随机采样与经典模拟退火采样的 BQM 能量分布。\n",
            encoding="utf-8",
        )
    else:
        note_path.write_text(
            "本实验比较随机采样、经典模拟退火采样与 Quafu measured bitstrings 的 BQM 能量分布。\n",
            encoding="utf-8",
        )

    print(f"\n写入完成: {csv_path}")
    print(f"写入完成: {jsonl_path}")
    print(f"写入完成: {note_path}")


if __name__ == "__main__":
    main()

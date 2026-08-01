from __future__ import annotations

import argparse
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

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.solvers import solve_bqm_classical


def _summarize_topk(
    method: str,
    k: int,
    candidates: list[dict[str, int]],
    bqm,
    instance,
) -> dict[str, Any]:
    selected = candidates[:k]
    if not selected:
        return {
            "method": method,
            "k": k,
            "best_assignment_energy": "",
            "best_route_distance": "",
            "feasible_rate": 0.0,
            "repair_rate": 0.0,
            "average_improvement_after_refinement": 0.0,
            "candidate_count": 0,
        }

    evals = [evaluate_sample(instance=instance, bqm=bqm, sample=s, two_opt_rounds=2) for s in selected]
    best_assignment = min(float(e["assignment_energy"]) for e in evals)
    best_route = min(float(e["route_distance_after_2opt"]) for e in evals)
    feasible_rate = sum(1 for e in evals if e["assignment_feasible"]) / len(evals)
    repair_rate = sum(1 for e in evals if e["repaired"]) / len(evals)
    avg_impr = sum(float(e["improvement_after_2opt"]) for e in evals) / len(evals)

    return {
        "method": method,
        "k": k,
        "best_assignment_energy": best_assignment,
        "best_route_distance": best_route,
        "feasible_rate": feasible_rate,
        "repair_rate": repair_rate,
        "average_improvement_after_refinement": avg_impr,
        "candidate_count": len(evals),
    }


def _top_candidates_by_energy(candidates: list[dict[str, int]], bqm, limit: int) -> list[dict[str, int]]:
    scored = [(float(bqm.energy(s)), s) for s in candidates]
    scored.sort(key=lambda x: x[0])
    out: list[dict[str, int]] = []
    seen = set()
    for _energy, sample in scored:
        key = tuple(sorted(sample.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="top-k 候选解细化实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--customers", type=str, default="")
    parser.add_argument("--vehicles", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results/topk_refinement")
    parser.add_argument("--sample-pool", type=int, default=200)
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
    csv_path = outdir / "topk_refinement.csv"
    jsonl_path = outdir / "topk_refinement.jsonl"

    rows: list[dict[str, Any]] = []

    quafu_available = has_quafu_bitstrings_available()
    quafu_counts = read_quafu_counts_from_raw_result() if quafu_available else {}

    ks = [1, 5, 10]
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

                # random 候选池
                rng = random.Random(seed * 1000 + customers * 13 + vehicles)
                random_pool = []
                for _ in range(max(20, args.sample_pool)):
                    random_pool.append({str(v): int(rng.randint(0, 1)) for v in bqm.variables})
                random_top = _top_candidates_by_energy(random_pool, bqm, limit=max(ks))

                # classical SA 候选池
                import dimod

                ss = dimod.SimulatedAnnealingSampler().sample(
                    bqm,
                    num_reads=max(20, args.sample_pool),
                    num_sweeps=40,
                )
                classical_pool = [dict(rec.sample) for rec in ss.data(['sample'])]
                classical_top = _top_candidates_by_energy(classical_pool, bqm, limit=max(ks))

                # quafu 候选池（若可用）
                quafu_top: list[dict[str, int]] = []
                if quafu_counts:
                    sorted_counts = sorted(quafu_counts.items(), key=lambda kv: kv[1], reverse=True)
                    for bitstring, _count in sorted_counts:
                        if len(bitstring) < len(x_vars):
                            continue
                        sample = {str(v): 0 for v in bqm.variables}
                        for i, var in enumerate(x_vars):
                            sample[var] = 1 if bitstring[-1 - i] == "1" else 0
                        quafu_top.append(sample)
                    quafu_top = _top_candidates_by_energy(quafu_top, bqm, limit=max(ks))

                for k in ks:
                    base = {
                        "seed": seed,
                        "customers": customers,
                        "vehicles": vehicles,
                        "capacity": capacity,
                    }

                    r = dict(base)
                    r.update(_summarize_topk("random_topk_refine", k, random_top, bqm, instance))
                    rows.append(r)
                    append_jsonl(jsonl_path, r)

                    c = dict(base)
                    c.update(_summarize_topk("classical_sa_topk_refine", k, classical_top, bqm, instance))
                    rows.append(c)
                    append_jsonl(jsonl_path, c)

                    if quafu_top:
                        q = dict(base)
                        q.update(_summarize_topk("quafu_topk_refine", k, quafu_top, bqm, instance))
                        rows.append(q)
                        append_jsonl(jsonl_path, q)

    write_csv(csv_path, rows)

    note_path = outdir / "topk_refinement_note.md"
    note_path.write_text(
        "top-k 候选解细化实验体现了混合量子-经典流程的核心思想：采样侧提供多个候选客户-车辆分配方案，经典侧负责可行性修复、路线构造和局部细化，最终从候选解集合中选择较优路线。\n",
        encoding="utf-8",
    )

    print(f"\n写入完成: {csv_path}")
    print(f"写入完成: {jsonl_path}")
    print(f"写入完成: {note_path}")


if __name__ == "__main__":
    main()

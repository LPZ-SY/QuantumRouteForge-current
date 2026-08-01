from __future__ import annotations

import argparse
import json
import random
import time
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
    make_run_id,
    now_iso,
    read_quafu_counts_from_raw_result,
    write_csv,
)

import sys

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.solvers import solve_bqm_classical


def main() -> None:
    parser = argparse.ArgumentParser(description="Quafu 闭环最小实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--customers", type=int, default=12)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="results/quantum_closed_loop")
    args = parser.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    csv_path = outdir / "closed_loop.csv"
    jsonl_path = outdir / "closed_loop.jsonl"
    unavailable_path = outdir / "closed_loop_unavailable.md"

    if not has_quafu_bitstrings_available():
        unavailable_path.write_text(
            "当前 Quafu 诊断结果显示 measured bitstring 不可用或未进入闭环，因此不生成 closed_loop.csv/jsonl，以避免伪造量子采样结果。\n",
            encoding="utf-8",
        )
        print(f"写入: {unavailable_path}")
        return

    quafu_counts = read_quafu_counts_from_raw_result()
    if not quafu_counts:
        unavailable_path.write_text(
            "诊断文件存在但未解析到有效 measurement counts，因此不生成 closed_loop.csv/jsonl。\n",
            encoding="utf-8",
        )
        print(f"写入: {unavailable_path}")
        return

    seed = int(args.seed)
    customers = int(args.customers)
    vehicles = int(args.vehicles)
    capacity = infer_capacity_from_seed(seed=seed, customers=customers, vehicles=vehicles)

    instance = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
    )
    bqm = build_bqm_for_instance(instance)
    x_vars = [v for v in bqm.variables if str(v).startswith("x_c")]

    rows: list[dict[str, Any]] = []

    def _record(
        mode: str,
        source: str,
        sample: dict[str, int],
        bitstring: str = "",
        count: float = 0.0,
        prob: float = 0.0,
    ):
        t0 = time.perf_counter()
        eval_info = evaluate_sample(instance=instance, bqm=bqm, sample=sample, two_opt_rounds=2)
        runtime = time.perf_counter() - t0
        row = {
            "run_id": make_run_id("closed"),
            "timestamp": now_iso(),
            "seed": seed,
            "customers": customers,
            "vehicles": vehicles,
            "capacity": capacity,
            "mode": mode,
            "bitstring_source": source,
            "bitstring": bitstring,
            "count": count,
            "probability": prob,
            "assignment_energy": float(eval_info["assignment_energy"]),
            "assignment_feasible": bool(eval_info["assignment_feasible"]),
            "onehot_violation_count": int(eval_info["onehot_violation_count"]),
            "capacity_violation_count": int(eval_info["capacity_violation_count"]),
            "total_over_capacity": int(eval_info["total_over_capacity"]),
            "repaired": bool(eval_info["repaired"]),
            "repair_summary": eval_info["repair_summary"],
            "route_distance_before_2opt": float(eval_info["route_distance_before_2opt"]),
            "route_distance_after_2opt": float(eval_info["route_distance_after_2opt"]),
            "total_route_distance": float(eval_info["route_distance_after_2opt"]),
            "total_runtime_sec": float(runtime),
            "quafu_task_id": "",
            "quafu_backend": "",
            "quafu_status": "bitstring_used",
            "error_message": "",
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)

    # random
    rng = random.Random(seed * 123)
    random_sample = {str(v): int(rng.randint(0, 1)) for v in bqm.variables}
    _record("random_bitstring_refine", "random", random_sample, bitstring="")

    # classical SA
    c_run = solve_bqm_classical(bqm=bqm, num_reads=120, num_sweeps=40)
    _record("classical_sa_refine", "classical_sa", c_run.sample, bitstring="")

    # quafu top-k
    counts_sorted = sorted(quafu_counts.items(), key=lambda kv: kv[1], reverse=True)
    total_counts = sum(float(v) for _, v in counts_sorted) or 1.0

    for k in [1, 5, 10]:
        for bitstring, cnt in counts_sorted[:k]:
            if len(bitstring) < len(x_vars):
                continue
            sample = {str(v): 0 for v in bqm.variables}
            for i, var in enumerate(x_vars):
                sample[var] = 1 if bitstring[-1 - i] == "1" else 0
            _record(
                f"quafu_top{k}_refine",
                f"quafu_top{k}",
                sample,
                bitstring=bitstring,
                count=float(cnt),
                prob=float(cnt) / total_counts,
            )

    write_csv(csv_path, rows)
    print(f"写入: {csv_path}")
    print(f"写入: {jsonl_path}")


if __name__ == "__main__":
    main()

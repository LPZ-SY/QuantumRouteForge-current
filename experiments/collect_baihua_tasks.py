from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_baihua_deepblock import analyze, write_report  # noqa: E402
from quantum_route_forge.candidate_evaluator import (  # noqa: E402
    evaluate_counts,
    true_route_distance,
)
from quantum_route_forge.env import load_env_file, quafu_token  # noqa: E402
from quantum_route_forge.experiment_logger import (  # noqa: E402
    atomic_json,
    read_jsonl,
    write_csv,
)
from quantum_route_forge.models import Customer  # noqa: E402
from replay_baihua_counts import _proxy  # noqa: E402


FINISHED = {"finished", "completed", "done", "success", "failed", "error", "cancelled"}


def _collect(manager, task_id: str, timeout_sec: int, interval_sec: float) -> tuple[dict[str, int], str, str]:
    deadline = time.monotonic() + max(1, int(timeout_sec))
    status = "unknown"
    error = ""
    while True:
        raw = manager.result(int(task_id))
        counts: dict[str, int] = {}
        if isinstance(raw, Mapping):
            status = str(raw.get("status") or "unknown")
            error = str(raw.get("error") or "")
            raw_counts = raw.get("corrected") or raw.get("count") or {}
            if isinstance(raw_counts, Mapping):
                for key, value in raw_counts.items():
                    try:
                        count = int(value)
                    except (TypeError, ValueError):
                        continue
                    if count > 0:
                        counts[str(key).replace(" ", "")] = count
        if counts or status.strip().lower() in FINISHED or time.monotonic() >= deadline:
            return counts, status, error
        time.sleep(max(0.2, float(interval_sec)))


def _assignment_payload(assignments) -> dict[str, list[int]]:
    return {
        str(vehicle): [customer.customer_id for customer in customers]
        for vehicle, customers in sorted(assignments.items())
    }


def collect_run(input_dir: Path, timeout_sec: int, interval_sec: float) -> dict[str, object]:
    load_env_file(ROOT / ".env")
    from quark import Task

    manager = Task(quafu_token(path=ROOT / ".env"))
    replay_files = sorted((input_dir / "replay").glob("seed_*_iter_*_baihua_p*.json"))
    metric_rows = read_jsonl(input_dir / "subproblem_metrics.jsonl")
    metrics_by_replay = {str(row.get("replay_file")): row for row in metric_rows}
    collected = pending = failed = 0
    trajectory_consistent = True
    previous_after: dict[str, list[int]] | None = None

    for path in replay_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(record.get("run", {}).get("task_id") or "")
        if not task_id:
            failed += 1
            continue
        counts, status, error = _collect(manager, task_id, timeout_sec, interval_sec)
        if not counts:
            pending += status.strip().lower() not in FINISHED
            failed += status.strip().lower() in {"failed", "error", "cancelled", "finished", "completed", "done", "success"}
            continue

        customers = [Customer(**row) for row in record["instance"]["customers"]]
        by_id = {customer.customer_id: customer for customer in customers}
        assignments = {
            int(vehicle): [by_id[int(customer_id)] for customer_id in customer_ids]
            for vehicle, customer_ids in record["assignments_before"].items()
        }
        if previous_after is not None and record["assignments_before"] != previous_after:
            trajectory_consistent = False
        proxy = _proxy(record["proxy"])
        block_customers = [by_id[customer_id] for customer_id in proxy.customer_ids]
        batch, accepted_assignment = evaluate_counts(
            arm=record["run"]["arm"],
            counts=counts,
            proxy=proxy,
            assignments=assignments,
            block_customers=block_customers,
            depot=tuple(float(value) for value in record["instance"]["depot"]),
            vehicle_capacity=int(record["instance"]["vehicle_capacity"]),
            candidate_k=int(record["fairness"]["candidate_k"]),
            filter_extremes=bool(record["fairness"]["filter_extremes"]),
            routing_method=str(record["fairness"]["routing_method"]),
        )
        after = accepted_assignment if accepted_assignment is not None else assignments
        after_payload = _assignment_payload(after)
        if accepted_assignment is not None and record["assignments_after"] != after_payload:
            trajectory_consistent = False
        previous_after = after_payload

        record["run"]["counts"] = counts
        record["run"]["message"] = (
            f"Collected task status={status}; {len(counts)} outcomes over {sum(counts.values())} shots."
        )
        record["candidate_batch"] = batch.payload()
        record["assignments_after"] = after_payload
        record["collection"] = {
            "status": status,
            "error": error,
            "trajectory_consistent": trajectory_consistent,
        }
        atomic_json(path, record)
        stem = path.stem
        atomic_json(input_dir / "counts" / f"{stem}.json", counts)

        relative = f"replay/{path.name}"
        metric = metrics_by_replay.get(relative)
        if metric is not None:
            accepted = batch.accepted
            best = batch.best_of_shots
            top_rows = list(batch.top_frequency)
            metric.update(
                {
                    "unique_candidates": batch.unique_candidates,
                    "feasible_candidates": batch.feasible_candidates,
                    "evaluated_candidates": batch.evaluated_candidates,
                    "accepted": accepted is not None,
                    "accepted_improvement": accepted.improvement if accepted else 0.0,
                    "top1_best_distance": min((row.true_distance for row in top_rows[:1]), default=None),
                    "top5_best_distance": min((row.true_distance for row in top_rows[:5]), default=None),
                    "top8_best_distance": min((row.true_distance for row in top_rows[:8]), default=None),
                    "best_of_shots_distance": best.true_distance if best else None,
                    "all_zero_probability": batch.all_zero_probability,
                    "all_one_probability": batch.all_one_probability,
                    "distribution_entropy": batch.distribution_entropy,
                    "top_k_probability": batch.top_k_probability,
                    "message": record["run"]["message"],
                }
            )
        collected += 1

    (input_dir / "subproblem_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    write_csv(input_dir / "subproblem_metrics.csv", metric_rows)

    instance_rows = read_jsonl(input_dir / "instance_summary.jsonl")
    hardware_rows = [row for row in instance_rows if row.get("arm") == "baihua"]
    if hardware_rows and replay_files:
        replay_records = [json.loads(path.read_text(encoding="utf-8")) for path in replay_files]
        accepted_moves = sum(
            bool((record.get("candidate_batch") or {}).get("accepted"))
            for record in replay_records
        )
        if trajectory_consistent and previous_after is not None:
            last = replay_records[-1]
            customers = [Customer(**row) for row in last["instance"]["customers"]]
            by_id = {customer.customer_id: customer for customer in customers}
            final_assignments = {
                int(vehicle): [by_id[int(customer_id)] for customer_id in customer_ids]
                for vehicle, customer_ids in previous_after.items()
            }
            final_distance = true_route_distance(
                final_assignments,
                tuple(float(value) for value in last["instance"]["depot"]),
                routing_method=str(last["fairness"]["routing_method"]),
            )
            row = hardware_rows[0]
            baseline = float(row["baseline_distance"])
            exact_distance = float(row["exact_distance"])
            headroom = baseline - exact_distance
            row.update(
                {
                    "final_distance": final_distance,
                    "improvement_vs_baseline": baseline - final_distance,
                    "accepted_moves": accepted_moves,
                    "acceptance_rate": accepted_moves / max(1, len(replay_files)),
                    "headroom_utilization": (
                        (baseline - final_distance) / headroom if headroom > 1e-9 else None
                    ),
                    "status": "hardware_counts_collected",
                }
            )
        else:
            hardware_rows[0]["status"] = "hardware_counts_collected_but_closed_loop_resubmission_required"

    random_by_seed = {
        int(row["seed"]): float(row["final_distance"])
        for row in instance_rows
        if row.get("arm") == "random"
    }
    for row in instance_rows:
        random_distance = random_by_seed.get(int(row["seed"]))
        if random_distance is not None:
            row["difference_vs_random"] = float(row["final_distance"]) - random_distance
    (input_dir / "instance_summary.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in instance_rows),
        encoding="utf-8",
    )
    write_csv(input_dir / "instance_summary.csv", instance_rows)
    analysis = analyze(input_dir)
    atomic_json(input_dir / "analysis.json", analysis)
    write_report(input_dir, analysis)
    summary = {
        "tasks": len(replay_files),
        "collected": collected,
        "pending": pending,
        "failed": failed,
        "trajectory_consistent": trajectory_consistent,
    }
    atomic_json(input_dir / "collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect submitted Baihua DeepBlock task counts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--interval-sec", type=float, default=3.0)
    args = parser.parse_args()
    print(json.dumps(collect_run(args.input, args.timeout_sec, args.interval_sec), ensure_ascii=False))


if __name__ == "__main__":
    main()

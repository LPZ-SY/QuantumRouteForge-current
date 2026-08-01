from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.candidate_evaluator import evaluate_counts  # noqa: E402
from quantum_route_forge.experiment_logger import atomic_json, write_csv  # noqa: E402
from quantum_route_forge.models import Customer  # noqa: E402
from quantum_route_forge.sparse_proxy_qubo import (  # noqa: E402
    ProxyInteraction,
    SparseProxyQUBO,
)


def _proxy(payload: dict) -> SparseProxyQUBO:
    return SparseProxyQUBO(
        customer_ids=tuple(int(value) for value in payload["customer_ids"]),
        vehicle_pair=tuple(int(value) for value in payload["vehicle_pair"]),
        linear=tuple(float(value) for value in payload["linear"]),
        quadratic=tuple(
            ProxyInteraction(
                left=int(row["left"]),
                right=int(row["right"]),
                coefficient=float(row["coefficient"]),
                kept=bool(row["kept"]),
                reason=str(row["reason"]),
            )
            for row in payload["quadratic"]
        ),
        constant=float(payload.get("constant", 0.0)),
        current_bitstring=str(payload["current_bitstring"]),
        scale=float(payload.get("scale", 1.0)),
    )


def replay_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    customer_rows = [Customer(**row) for row in record["instance"]["customers"]]
    by_id = {customer.customer_id: customer for customer in customer_rows}
    assignments = {
        int(vehicle): [by_id[int(customer_id)] for customer_id in customer_ids]
        for vehicle, customer_ids in record["assignments_before"].items()
    }
    proxy = _proxy(record["proxy"])
    block_customers = [by_id[customer_id] for customer_id in proxy.customer_ids]
    counts = record["run"].get("counts") or {}
    expected_batch = record.get("candidate_batch")
    if not counts:
        passed = expected_batch is None and record["assignments_before"] == record["assignments_after"]
        return {
            "replay_file": path.name,
            "seed": record["seed"],
            "iteration": record["iteration"],
            "arm": record["run"]["arm"],
            "passed": passed,
            "reason": "dry_run_no_counts" if passed else "dry_run_record_mismatch",
        }
    fairness = record["fairness"]
    candidate_k = int(fairness["candidate_k"])
    if record["run"]["arm"] == "exact":
        candidate_k = 1 << proxy.width
    batch, accepted_assignment = evaluate_counts(
        arm=record["run"]["arm"],
        counts=counts,
        proxy=proxy,
        assignments=assignments,
        block_customers=block_customers,
        depot=tuple(float(value) for value in record["instance"]["depot"]),
        vehicle_capacity=int(record["instance"]["vehicle_capacity"]),
        candidate_k=candidate_k,
        filter_extremes=bool(fairness["filter_extremes"]),
        routing_method=str(fairness["routing_method"]),
    )
    expected_accepted = (expected_batch or {}).get("accepted")
    expected_best = (expected_batch or {}).get("best_of_shots")
    accepted_match = (
        (batch.accepted is None and expected_accepted is None)
        or (
            batch.accepted is not None
            and expected_accepted is not None
            and batch.accepted.bitstring == expected_accepted["bitstring"]
            and math.isclose(batch.accepted.true_distance, float(expected_accepted["true_distance"]), abs_tol=1e-9)
        )
    )
    best_match = (
        (batch.best_of_shots is None and expected_best is None)
        or (
            batch.best_of_shots is not None
            and expected_best is not None
            and batch.best_of_shots.bitstring == expected_best["bitstring"]
            and math.isclose(batch.best_of_shots.true_distance, float(expected_best["true_distance"]), abs_tol=1e-9)
        )
    )
    actual_after = (
        {
            str(vehicle): [customer.customer_id for customer in customers]
            for vehicle, customers in sorted(accepted_assignment.items())
        }
        if accepted_assignment is not None
        else record["assignments_before"]
    )
    assignment_match = actual_after == record["assignments_after"]
    return {
        "replay_file": path.name,
        "seed": record["seed"],
        "iteration": record["iteration"],
        "arm": record["run"]["arm"],
        "passed": accepted_match and best_match and assignment_match,
        "accepted_match": accepted_match,
        "best_of_shots_match": best_match,
        "assignment_match": assignment_match,
        "accepted_bitstring": batch.accepted.bitstring if batch.accepted else "",
        "best_of_shots_bitstring": batch.best_of_shots.bitstring if batch.best_of_shots else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay of Baihua DeepBlock counts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    files = sorted((args.input / "replay").glob("seed_*_iter_*.json"))
    rows = [replay_record(path) for path in files]
    summary = {
        "records": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "rows": rows,
    }
    atomic_json(args.input / "replay" / "verification.json", summary)
    write_csv(args.input / "replay" / "verification.csv", rows)
    if args.verify and summary["failed"]:
        raise SystemExit(f"Replay verification failed for {summary['failed']} record(s).")
    print(json.dumps({key: summary[key] for key in ("records", "passed", "failed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

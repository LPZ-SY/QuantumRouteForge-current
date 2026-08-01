from __future__ import annotations

import csv
import json
import math
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge.assignment_bqm import assignment_var, build_assignment_bqm, decode_assignment
from quantum_route_forge.geometry import cycle_distance
from quantum_route_forge.models import Customer, DispatchInstance
from quantum_route_forge.routing import build_route_plans


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int_list(text: str, default: list[int]) -> list[int]:
    text = (text or "").strip()
    if not text:
        return default
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or default


def infer_capacity_from_seed(seed: int, customers: int, vehicles: int, ratio: float = 1.15) -> int:
    # Demands are independent from capacity in scenario generator, so we can probe with any positive capacity.
    from quantum_route_forge import generate_dispatch_instance

    probe = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=1,
    )
    base = probe.total_demand / max(1, vehicles)
    return max(1, math.ceil(base * ratio))


def random_sample_for_bqm(bqm, rng: random.Random) -> dict[str, int]:
    return {str(v): int(rng.randint(0, 1)) for v in bqm.variables}


def compute_onehot_violation_count(sample: dict[str, int], customers: Iterable[Customer], vehicles: int) -> int:
    violations = 0
    for c in customers:
        s = 0
        for v in range(vehicles):
            s += int(sample.get(assignment_var(c.customer_id, v), 0))
        if s != 1:
            violations += 1
    return violations


def compute_capacity_metrics(assignments: dict[int, list[Customer]], capacity: int) -> tuple[int, int, dict[int, int]]:
    loads: dict[int, int] = {}
    violation_count = 0
    total_over_capacity = 0
    for v, clist in assignments.items():
        load = int(sum(c.demand for c in clist))
        loads[v] = load
        over = max(0, load - capacity)
        if over > 0:
            violation_count += 1
            total_over_capacity += over
    return violation_count, total_over_capacity, loads


def repair_assignment_if_needed(
    assignments: dict[int, list[Customer]],
    capacity: int,
    max_iters: int = 2000,
) -> tuple[dict[int, list[Customer]], bool, str]:
    assignments = {k: list(v) for k, v in assignments.items()}

    for _ in range(max_iters):
        cap_viol, _, loads = compute_capacity_metrics(assignments, capacity)
        if cap_viol == 0:
            return assignments, False, "no_repair_needed"

        overloaded = [v for v, load in loads.items() if load > capacity]
        moved = False
        for ov in sorted(overloaded, key=lambda x: loads[x], reverse=True):
            for customer in sorted(assignments[ov], key=lambda c: (-c.demand, c.customer_id)):
                targets = [
                    t
                    for t in assignments.keys()
                    if t != ov and loads[t] + customer.demand <= capacity
                ]
                if not targets:
                    continue
                target = min(targets, key=lambda t: loads[t])
                assignments[ov].remove(customer)
                assignments[target].append(customer)
                moved = True
                break
            if moved:
                break

        if not moved:
            return assignments, True, "repair_attempted_but_stuck"

    return assignments, True, "repair_iter_limit_reached"


def serialize_assignments(assignments: dict[int, list[Customer]]) -> dict[str, list[int]]:
    return {
        str(v): [int(c.customer_id) for c in clist]
        for v, clist in sorted(assignments.items(), key=lambda kv: kv[0])
    }


def routes_payload(routes) -> list[dict[str, Any]]:
    payload = []
    for r in routes:
        payload.append(
            {
                "vehicle_id": int(r.vehicle_id),
                "stops": len(r.customers),
                "load": int(r.load),
                "distance": float(r.distance),
                "customer_ids": [int(c.customer_id) for c in r.customers],
            }
        )
    return payload


def coordinates_payload(instance: DispatchInstance) -> dict[str, Any]:
    return {
        "depot": [float(instance.depot[0]), float(instance.depot[1])],
        "customers": [
            {
                "customer_id": int(c.customer_id),
                "x": float(c.x),
                "y": float(c.y),
                "demand": int(c.demand),
            }
            for c in instance.customers
        ],
    }


def evaluate_sample(
    instance: DispatchInstance,
    bqm,
    sample: dict[str, int],
    two_opt_rounds: int = 2,
) -> dict[str, Any]:
    # Allow partial samples (e.g., only x variables) by zero-filling unspecified vars.
    completed_sample = {str(v): int(sample.get(str(v), 0)) for v in bqm.variables}

    assignment_energy = float(bqm.energy(completed_sample))
    decoded = decode_assignment(completed_sample, instance.customers, instance.num_vehicles)
    onehot_violation_count = compute_onehot_violation_count(
        completed_sample, instance.customers, instance.num_vehicles
    )
    cap_violation_before, over_before, loads_before = compute_capacity_metrics(decoded, instance.vehicle_capacity)
    repaired_assignment, repaired, repair_summary = repair_assignment_if_needed(decoded, instance.vehicle_capacity)
    cap_violation_after, over_after, loads_after = compute_capacity_metrics(repaired_assignment, instance.vehicle_capacity)

    routes_before = build_route_plans(
        assignments=repaired_assignment,
        depot=instance.depot,
        two_opt_rounds=0,
    )
    routes_after = build_route_plans(
        assignments=repaired_assignment,
        depot=instance.depot,
        two_opt_rounds=two_opt_rounds,
    )
    route_distance_before = float(sum(r.distance for r in routes_before))
    route_distance_after = float(sum(r.distance for r in routes_after))

    return {
        "assignment_energy": assignment_energy,
        "decoded_assignment": serialize_assignments(repaired_assignment),
        "onehot_violation_count": int(onehot_violation_count),
        "capacity_violation_count": int(cap_violation_after),
        "capacity_violation_count_before_repair": int(cap_violation_before),
        "total_over_capacity": int(over_after),
        "total_over_capacity_before_repair": int(over_before),
        "onehot_feasible": bool(onehot_violation_count == 0),
        "capacity_feasible": bool(cap_violation_after == 0),
        "assignment_feasible": bool(onehot_violation_count == 0 and cap_violation_after == 0),
        "repaired": bool(repaired),
        "repair_summary": repair_summary,
        "vehicle_loads": {str(k): int(v) for k, v in sorted(loads_after.items(), key=lambda kv: kv[0])},
        "vehicle_loads_before_repair": {str(k): int(v) for k, v in sorted(loads_before.items(), key=lambda kv: kv[0])},
        "vehicle_route_lengths": {str(r.vehicle_id): float(r.distance) for r in routes_after},
        "route_distance_before_2opt": route_distance_before,
        "route_distance_after_2opt": route_distance_after,
        "improvement_after_2opt": route_distance_before - route_distance_after,
        "route_feasible": bool(cap_violation_after == 0),
        "routes_before_2opt": routes_payload(routes_before),
        "routes_after_2opt": routes_payload(routes_after),
    }


def build_bqm_for_instance(instance: DispatchInstance):
    return build_assignment_bqm(instance=instance)


def best_random_sample(bqm, seed: int, sample_count: int) -> tuple[dict[str, int], float]:
    rng = random.Random(seed)
    best_sample = None
    best_energy = None
    for _ in range(max(1, sample_count)):
        s = random_sample_for_bqm(bqm, rng)
        e = float(bqm.energy(s))
        if best_energy is None or e < best_energy:
            best_energy = e
            best_sample = s
    assert best_sample is not None and best_energy is not None
    return best_sample, best_energy


def has_quafu_bitstrings_available(summary_path: Path | None = None) -> bool:
    path = summary_path or (ROOT / "results" / "quafu_diagnostics" / "diagnostic_summary.json")
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("has_bitstrings"))


def read_quafu_counts_from_raw_result(raw_result_path: Path | None = None) -> dict[str, float]:
    path = raw_result_path or (ROOT / "results" / "quafu_diagnostics" / "raw_quafu_result_response.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    counts: dict[str, float] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                k = str(key).replace(" ", "")
                if set(k) <= {"0", "1"} and k:
                    try:
                        counts[k] = counts.get(k, 0.0) + float(value)
                    except Exception:
                        pass
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return counts


def summarize_energy(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "best": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    n = len(values)
    vals = sorted(values)
    mean = sum(vals) / n
    med = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in vals) / n
    return {
        "best": float(min(vals)),
        "mean": float(mean),
        "median": float(med),
        "std": float(math.sqrt(var)),
    }

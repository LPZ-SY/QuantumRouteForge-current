from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.baihua_topology import calibration_snapshot  # noqa: E402
from quantum_route_forge.clustering import capacity_constrained_kmeans  # noqa: E402
from quantum_route_forge.deepblock_solver import (  # noqa: E402
    DeepBlockConfig,
    run_deepblock_arm,
)
from quantum_route_forge.experiment_logger import ExperimentLogger  # noqa: E402
from quantum_route_forge.models import DispatchInstance  # noqa: E402
from quantum_route_forge.scenario import generate_dispatch_instance  # noqa: E402


DEFAULT_OUTDIR = ROOT / "results" / "baihua_deepblock_p2"


def parse_seeds(text: str) -> list[int]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(values) == 1:
        count = int(values[0])
        return list(range(1, count + 1))
    return [int(value) for value in values]


def _parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_instance(seed: int, customers: int, vehicles: int, capacity: int) -> DispatchInstance:
    if capacity > 0:
        return generate_dispatch_instance(
            seed=seed,
            num_customers=customers,
            num_vehicles=vehicles,
            vehicle_capacity=capacity,
        )
    probe = generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=1,
    )
    inferred = max(1, math.ceil(probe.total_demand / vehicles * 1.15))
    return generate_dispatch_instance(
        seed=seed,
        num_customers=customers,
        num_vehicles=vehicles,
        vehicle_capacity=inferred,
    )


def _load_calibration(
    *,
    backend: str,
    calibration_json: str,
    fetch_calibration: bool,
) -> dict[str, Any] | None:
    if calibration_json:
        return json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    if not fetch_calibration:
        return None
    try:
        from quark.circuit import Backend
    except ImportError as exc:
        raise RuntimeError("QuarkStudio is required to fetch Baihua calibration.") from exc
    return dict(Backend(backend).chip_info)


def run_experiment(
    *,
    seeds: Sequence[int],
    arms: Sequence[str],
    customers: int,
    vehicles: int,
    capacity: int,
    config: DeepBlockConfig,
    outdir: Path,
    chip_info: dict[str, Any] | None = None,
    manual_physical_qubits: Sequence[int] | None = None,
    api_token: str = "",
    include_exact: bool = True,
) -> list[dict[str, object]]:
    logger = ExperimentLogger(outdir)
    effective_arms = [arm.strip().lower() for arm in arms if arm.strip()]
    if include_exact and "exact" not in effective_arms:
        effective_arms.append("exact")
    logger.write_config(
        {
            "seeds": list(seeds),
            "customers": customers,
            "vehicles": vehicles,
            "capacity": capacity,
            "arms": effective_arms,
            "deepblock": as_config_payload(config),
            "manual_physical_qubits": list(manual_physical_qubits or ()),
            "hardware_submit_default": False,
        }
    )
    if chip_info:
        logger.write_calibration(calibration_snapshot(chip_info))
    else:
        logger.write_calibration(
            {
                "backend": config.backend,
                "calibration_time": None,
                "source": "not_requested_simulator_or_offline_dry_run",
                "qubits": [],
                "couplers": [],
            }
        )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        instance = _build_instance(int(seed), customers, vehicles, capacity)
        clustering = capacity_constrained_kmeans(instance, seed=int(seed))
        results = {}
        # Exact runs first so headroom is available for every reported arm.
        ordered_arms = sorted(effective_arms, key=lambda arm: (arm != "exact", arm))
        for arm in ordered_arms:
            results[arm] = run_deepblock_arm(
                instance=instance,
                initial_assignments=clustering.assignments,
                arm=arm,
                config=config,
                seed=int(seed),
                logger=logger,
                chip_info=chip_info,
                manual_physical_qubits=manual_physical_qubits,
                api_token=api_token,
            )
        exact = results.get("exact")
        baseline = next(iter(results.values())).baseline_distance
        exact_distance = exact.final_distance if exact else baseline
        random_distance = (
            results["random"].final_distance if "random" in results else None
        )
        headroom = baseline - exact_distance
        has_headroom = headroom > 1e-9
        for arm, result in results.items():
            row = {
                "seed": int(seed),
                "arm": arm,
                "clustering_method": clustering.method,
                "baseline_distance": baseline,
                "exact_distance": exact_distance,
                "has_refinement_space": has_headroom,
                "final_distance": result.final_distance,
                "improvement_vs_baseline": result.improvement,
                "difference_vs_random": (
                    result.final_distance - random_distance
                    if random_distance is not None
                    else None
                ),
                "accepted_moves": result.accepted_moves,
                "acceptance_rate": (
                    result.accepted_moves / result.attempted_subproblems
                    if result.attempted_subproblems
                    else 0.0
                ),
                "headroom_utilization": (
                    result.improvement / headroom if has_headroom else None
                ),
                "status": result.status,
                "vehicle_capacity": instance.vehicle_capacity,
                "total_demand": instance.total_demand,
                "attempted_subproblems": result.attempted_subproblems,
            }
            rows.append(row)
            logger.log_instance(row)
    logger.finalize_csv()
    # Produce the protocol report immediately; the standalone analysis command
    # can be rerun later after merging or extending results.
    from analyze_baihua_deepblock import analyze, write_report

    analysis = analyze(outdir)
    (outdir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(outdir, analysis)
    return rows


def as_config_payload(config: DeepBlockConfig) -> dict[str, object]:
    return {
        field: getattr(config, field)
        for field in config.__dataclass_fields__
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baihua 8-qubit overlapping DeepBlock CVRP experiment")
    parser.add_argument("--seeds", default="3", help="Single value means seed count; comma list means explicit seeds.")
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=0, help="0 infers a 1.15 balanced-capacity ratio.")
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--qaoa-depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-sweeps", type=int, default=1)
    parser.add_argument("--scan-order", choices=("forward", "bidirectional"), default="bidirectional")
    parser.add_argument("--arms", default="random,sim,baihua")
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--filter-extremes", action="store_true")
    parser.add_argument("--routing-method", choices=("heuristic", "ortools"), default="heuristic")
    parser.add_argument("--backend", default="Baihua")
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--fetch-calibration", action="store_true")
    parser.add_argument("--physical-qubits", default="")
    parser.add_argument("--max-cnot", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=240)
    parser.add_argument("--submit-hardware", action="store_true")
    parser.add_argument("--confirm-hardware-submit", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.submit_hardware and not args.confirm_hardware_submit:
        raise SystemExit("Refusing hardware submission without --confirm-hardware-submit.")
    _load_env_file(ROOT / ".env")
    chip_info = _load_calibration(
        backend=args.backend,
        calibration_json=args.calibration_json,
        fetch_calibration=args.fetch_calibration or args.submit_hardware,
    )
    config = DeepBlockConfig(
        pool_size=args.pool_size,
        block_size=args.block_size,
        overlap=args.overlap,
        qaoa_depth=args.qaoa_depth,
        shots=args.shots,
        candidate_k=args.candidate_k,
        max_sweeps=args.max_sweeps,
        scan_order=args.scan_order,
        filter_extremes=args.filter_extremes,
        routing_method=args.routing_method,
        backend=args.backend,
        max_cnot=args.max_cnot,
        max_depth=args.max_depth,
        submit_hardware=args.submit_hardware,
        confirm_hardware_submit=args.confirm_hardware_submit,
        wait_hardware=not args.no_wait,
    )
    rows = run_experiment(
        seeds=parse_seeds(args.seeds),
        arms=[value.strip() for value in args.arms.split(",") if value.strip()],
        customers=args.customers,
        vehicles=args.vehicles,
        capacity=args.capacity,
        config=config,
        outdir=args.outdir,
        chip_info=chip_info,
        manual_physical_qubits=_parse_ints(args.physical_qubits) or None,
        api_token=os.environ.get("QUAFU_API_TOKEN", ""),
        include_exact=not args.no_exact,
    )
    print(json.dumps({"outdir": str(args.outdir), "instance_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

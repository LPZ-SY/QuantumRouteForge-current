from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from run_baihua_deepblock import DeepBlockConfig, parse_seeds, run_experiment


def _specs(shots: int) -> list[tuple[str, DeepBlockConfig]]:
    common = dict(
        pool_size=16,
        block_size=8,
        shots=shots,
        max_sweeps=1,
        routing_method="heuristic",
    )
    return [
        ("reference_p1_k8_o3_bidir", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=8, overlap=3, scan_order="bidirectional")),
        ("p1_k16_o3_bidir", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=16, overlap=3, scan_order="bidirectional")),
        ("p1_k32_o3_bidir", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=32, overlap=3, scan_order="bidirectional")),
        ("p1_k64_o3_bidir", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=64, overlap=3, scan_order="bidirectional")),
        ("p2_k8_o3_bidir", DeepBlockConfig(**common, qaoa_depth=2, candidate_k=8, overlap=3, scan_order="bidirectional")),
        ("p2_k32_o3_bidir", DeepBlockConfig(**common, qaoa_depth=2, candidate_k=32, overlap=3, scan_order="bidirectional")),
        ("p1_k32_o0_bidir", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=32, overlap=0, scan_order="bidirectional")),
        ("p1_k32_o3_forward", DeepBlockConfig(**common, qaoa_depth=1, candidate_k=32, overlap=3, scan_order="forward")),
    ]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["configuration"]), str(row["arm"])), []).append(row)
    output: list[dict[str, object]] = []
    for (configuration, arm), values in sorted(groups.items()):
        improvements = [float(row["improvement_vs_baseline"]) for row in values]
        headroom = [
            float(row["headroom_utilization"])
            for row in values
            if row["headroom_utilization"] not in (None, "") and math.isfinite(float(row["headroom_utilization"]))
        ]
        output.append(
            {
                "configuration": configuration,
                "arm": arm,
                "seeds": len(values),
                "mean_improvement": statistics.fmean(improvements),
                "median_improvement": statistics.median(improvements),
                "mean_headroom_utilization": statistics.fmean(headroom) if headroom else None,
                "mean_accepted_moves": statistics.fmean(float(row["accepted_moves"]) for row in values),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired simulation ablations for Baihua DeepBlock")
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    combined: list[dict[str, object]] = []
    for name, config in _specs(args.shots):
        target = args.outdir / name
        run_experiment(
            seeds=seeds,
            arms=("random", "sim"),
            customers=args.customers,
            vehicles=args.vehicles,
            capacity=args.capacity,
            config=config,
            outdir=target,
            include_exact=True,
        )
        config_rows = _read_rows(target / "instance_summary.csv")
        for row in config_rows:
            row["configuration"] = name
            combined.append(row)
        manifest.append({"configuration": name, "outdir": str(target), "rows": len(config_rows)})
        print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)
    from quantum_route_forge.experiment_logger import atomic_json, write_csv

    summary = _aggregate(combined)
    write_csv(args.outdir / "ablation_instances.csv", combined)
    write_csv(args.outdir / "ablation_summary.csv", summary)
    atomic_json(args.outdir / "ablation_manifest.json", manifest)
    atomic_json(args.outdir / "ablation_summary.json", summary)
    print(json.dumps({"configurations": len(manifest), "rows": len(combined)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_baihua_deepblock import (
    DEFAULT_OUTDIR,
    DeepBlockConfig,
    _load_calibration,
    _load_env_file,
    _parse_ints,
    parse_seeds,
    run_experiment,
    ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baihua DeepBlock p=1/2/3 depth scan")
    parser.add_argument("--seeds", default="3")
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--scan-order", choices=("forward", "bidirectional"), default="bidirectional")
    parser.add_argument("--arms", default="random,sim,baihua")
    parser.add_argument("--backend", default="Baihua")
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--fetch-calibration", action="store_true")
    parser.add_argument("--physical-qubits", default="")
    parser.add_argument("--routing-method", choices=("heuristic", "ortools"), default="heuristic")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR.parent / "baihua_depth_scan")
    args = parser.parse_args()

    _load_env_file(ROOT / ".env")
    chip_info = _load_calibration(
        backend=args.backend,
        calibration_json=args.calibration_json,
        fetch_calibration=args.fetch_calibration,
    )
    summaries = []
    for depth in _parse_ints(args.depths):
        if depth not in {1, 2, 3}:
            raise SystemExit(f"Unsupported depth p={depth}")
        outdir = args.outdir / f"p{depth}"
        rows = run_experiment(
            seeds=parse_seeds(args.seeds),
            arms=[value.strip() for value in args.arms.split(",") if value.strip()],
            customers=args.customers,
            vehicles=args.vehicles,
            capacity=args.capacity,
            config=DeepBlockConfig(
                pool_size=args.pool_size,
                block_size=args.block_size,
                overlap=args.overlap,
                qaoa_depth=depth,
                shots=args.shots,
                candidate_k=args.candidate_k,
                scan_order=args.scan_order,
                routing_method=args.routing_method,
                backend=args.backend,
            ),
            outdir=outdir,
            chip_info=chip_info,
            manual_physical_qubits=_parse_ints(args.physical_qubits) or None,
            include_exact=True,
        )
        summaries.append({"depth": depth, "outdir": str(outdir), "instance_rows": len(rows)})
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "depth_scan_index.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()

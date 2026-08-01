from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


REQUIRED_FIELDS = {
    "iteration",
    "sub_k",
    "n_edges",
    "sub_cnot",
    "vehicles",
    "quantum_improvement",
    "classical_improvement",
    "paired_improvement_delta",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "quantum" in payload and isinstance(payload["quantum"], dict):
        payload = payload["quantum"]
    records = payload.get("quantum_iteration_records")
    if not isinstance(records, list):
        raise ValueError(
            "Input JSON does not contain quantum_iteration_records."
        )
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Iteration record {index} is not an object.")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Iteration record {index} is missing: "
                f"{', '.join(sorted(missing))}."
            )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["sub_k"]), []).append(record)

    rows = []
    for sub_k in sorted(grouped):
        layer = grouped[sub_k]
        rows.append(
            {
                "sub_k": sub_k,
                "iterations": len(layer),
                "mean_n_edges": fmean(
                    float(record["n_edges"]) for record in layer
                ),
                "mean_sub_cnot": fmean(
                    float(record["sub_cnot"]) for record in layer
                ),
                "mean_quantum_improvement": fmean(
                    float(record["quantum_improvement"])
                    for record in layer
                ),
                "mean_classical_improvement": fmean(
                    float(record["classical_improvement"])
                    for record in layer
                ),
                "mean_delta_b_minus_a": fmean(
                    float(record["paired_improvement_delta"])
                    for record in layer
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "sub_k",
        "iterations",
        "mean_n_edges",
        "mean_sub_cnot",
        "mean_quantum_improvement",
        "mean_classical_improvement",
        "mean_delta_b_minus_a",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired quantum/classical Tabu-QAOA ablation "
            "records by actual subproblem size."
        )
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    records = _load_records(args.input_json)
    rows = summarize(records)
    result = {
        "source": str(args.input_json.resolve()),
        "total_iterations": len(records),
        "layers": rows,
        "delta_definition": (
            "mean quantum proxy improvement minus mean paired classical "
            "shadow proxy improvement; positive favors B"
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    print(serialized)


if __name__ == "__main__":
    main()

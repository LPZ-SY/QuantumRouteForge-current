from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_default(value: object) -> object:
    if hasattr(value, "payload"):
        return value.payload()  # type: ignore[no-any-return, attr-defined]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, default=_json_default)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


class ExperimentLogger:
    """Structured DeepBlock artifact writer with one replay record per arm."""

    def __init__(self, outdir: Path | str):
        self.outdir = Path(outdir)
        if (self.outdir / "config.json").exists():
            raise FileExistsError(
                f"Experiment output already exists at {self.outdir}. "
                "Choose a new --outdir to avoid mixing independent runs."
            )
        for name in ("counts", "circuits", "mappings", "replay"):
            (self.outdir / name).mkdir(parents=True, exist_ok=True)
        self.instance_jsonl = self.outdir / "instance_summary.jsonl"
        self.subproblem_jsonl = self.outdir / "subproblem_metrics.jsonl"

    def write_config(self, config: Mapping[str, object]) -> None:
        atomic_json(
            self.outdir / "config.json",
            {
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                **dict(config),
            },
        )

    def write_calibration(self, snapshot: Mapping[str, object]) -> None:
        atomic_json(self.outdir / "calibration_snapshot.json", dict(snapshot))

    def log_subproblem(
        self,
        *,
        seed: int,
        iteration: int,
        block_id: str,
        arm: str,
        counts: Mapping[str, int],
        qasm: str,
        physical_qasm: str,
        mapping: Mapping[str, object],
        replay_record: Mapping[str, object],
        metrics: Mapping[str, object],
    ) -> None:
        stem = f"seed_{int(seed)}_iter_{int(iteration):03d}_{block_id}_{arm}"
        atomic_json(self.outdir / "counts" / f"{stem}.json", dict(counts))
        (self.outdir / "circuits" / f"{stem}.qasm").write_text(qasm, encoding="utf-8")
        if physical_qasm:
            (self.outdir / "circuits" / f"{stem}_physical.qasm").write_text(
                physical_qasm,
                encoding="utf-8",
            )
        atomic_json(self.outdir / "mappings" / f"{stem}.json", dict(mapping))
        atomic_json(self.outdir / "replay" / f"{stem}.json", dict(replay_record))
        append_jsonl(
            self.subproblem_jsonl,
            {
                "seed": int(seed),
                "iteration": int(iteration),
                "block_id": block_id,
                "arm": arm,
                **dict(metrics),
                "counts_file": f"counts/{stem}.json",
                "replay_file": f"replay/{stem}.json",
            },
        )

    def log_instance(self, row: Mapping[str, object]) -> None:
        append_jsonl(self.instance_jsonl, row)

    def finalize_csv(self) -> None:
        write_csv(self.outdir / "instance_summary.csv", read_jsonl(self.instance_jsonl))
        write_csv(self.outdir / "subproblem_metrics.csv", read_jsonl(self.subproblem_jsonl))

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.cvrp_width_scan import (  # noqa: E402
    FILTER_RULES,
    SCAN_SEED,
    analyze_counts,
    build_chain_qaoa_qasm,
    build_width_instance,
    classical_payload,
    find_fidelity_greedy_chain,
    instance_payload,
    qasm_gate_metrics,
    random_sampling_baseline,
)


DEFAULT_WIDTHS = (14, 16, 18, 20, 22)
DEFAULT_OUTDIR = ROOT / "results" / "cvrp_width_scan_20260731"


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or not os.environ.get(key, "").strip():
            os.environ[key] = value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _parse_widths(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def _fetch_chip_info(chip: str) -> dict[str, Any]:
    from quark.circuit import Backend

    backend = Backend(chip)
    return dict(backend.chip_info)


def _task_manager():
    # The desktop process can retain an expired token after .env is refreshed.
    # For reproducible hardware runs the repository-local, gitignored .env is
    # the explicit credential source.
    _load_env_file(ROOT / ".env", override=True)
    token = os.getenv("QUAFU_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QUAFU_API_TOKEN is empty")
    from quark import Task

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            return Task(token)
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"QuarkStudio verification failed after 5 attempts: {last_error}"
    ) from last_error


def _run_task_with_retry(manager, task: dict[str, Any]):
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return manager.run(task)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"QuarkStudio submission failed after 3 attempts: {last_error}"
    ) from last_error


def _result_with_retry(manager, task_id: int):
    for attempt in range(1, 4):
        try:
            return manager.result(task_id)
        except Exception:
            if attempt < 3:
                time.sleep(attempt)
    return {}


def prepare(
    widths: list[int],
    outdir: Path,
    chip: str,
    seed: int,
    shots: int,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    from quark.circuit import Backend, Transpiler

    backend = Backend(chip)
    chip_info = dict(backend.chip_info)
    _atomic_json(outdir / "calibration_snapshot.json", chip_info)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "chip": chip,
        "seed": seed,
        "shots": shots,
        "filter_rules": FILTER_RULES,
        "widths": {},
    }
    for width in widths:
        instance = build_width_instance(width=width, seed=seed)
        chain = find_fidelity_greedy_chain(chip_info, width=width)
        qasm = build_chain_qaoa_qasm(instance)
        logical_metrics = qasm_gate_metrics(qasm)
        precompiled = Transpiler(backend).run(
            qasm,
            target_qubits=list(chain.qubits),
            optimize_level=1,
            niter=5,
        )
        physical_qasm = str(precompiled.to_openqasm2)
        physical_metrics = qasm_gate_metrics(physical_qasm)
        row = {
            "status": "prepared",
            "instance": instance_payload(instance),
            "classical": classical_payload(instance),
            "physical_chain": chain.payload(),
            "circuit": {
                "qasm_file": f"width_{width}.qasm",
                "physical_qasm_file": f"width_{width}_physical.qasm",
                "logical_cnot_count": logical_metrics["cnot_count"],
                "logical_depth": logical_metrics["depth"],
                "logical_gate_count": logical_metrics["gate_count"],
                "physical_native_two_qubit_gate_count": physical_metrics[
                    "two_qubit_gate_count"
                ],
                "physical_depth": physical_metrics["depth"],
                "physical_gate_count": physical_metrics["gate_count"],
                "swap_count": 0,
                "precompiled_locally": True,
                "gamma": 1.1,
                "physical_rx_angle": 0.8,
                "mixer_convention": "legacy_hardware_bridge_rx_beta_not_rx_2beta",
            },
            "hardware": {
                "task_id": "",
                "shots_requested": shots,
            },
        }
        payload["widths"][str(width)] = row
        (outdir / f"width_{width}.qasm").write_text(qasm, encoding="utf-8")
        (outdir / f"width_{width}_physical.qasm").write_text(
            physical_qasm,
            encoding="utf-8",
        )
        _atomic_json(outdir / f"instance_width_{width}.json", row["instance"])
    _atomic_json(outdir / "scan.json", payload)
    return payload


def submit(
    payload: dict[str, Any],
    outdir: Path,
    chip: str,
) -> dict[str, Any]:
    manager = _task_manager()
    for width_text, row in payload["widths"].items():
        hardware = row["hardware"]
        if hardware.get("task_id"):
            continue
        width = int(width_text)
        chain = row["physical_chain"]["qubits"]
        if row["physical_chain"]["uncalibrated_couplings"]:
            row["status"] = "skipped_uncalibrated_chain"
            continue
        physical_qasm_file = row["circuit"].get(
            "physical_qasm_file",
            f"width_{width}.qasm",
        )
        qasm = (outdir / physical_qasm_file).read_text(encoding="utf-8")
        task = {
            "chip": chip,
            "name": f"QRF_CVRP_premapped_chain_w{width}",
            "circuit": qasm,
            "shots": int(hardware["shots_requested"]),
            "compile": False,
            "options": {
                "compiler": None,
                "correct": False,
                "open_dd": None,
                "target_qubits": [],
            },
        }
        task_id = _run_task_with_retry(manager, task)
        if not isinstance(task_id, int):
            row["status"] = "submission_failed"
            hardware["submission_response"] = task_id
        else:
            hardware["task_id"] = str(task_id)
            hardware["submitted_at"] = datetime.now().astimezone().isoformat()
            row["status"] = "submitted"
            print(f"submitted width={width} task_id={task_id}", flush=True)
        _atomic_json(outdir / "scan.json", payload)
    return payload


def collect(
    payload: dict[str, Any],
    outdir: Path,
    timeout_sec: int,
    poll_interval_sec: int,
) -> dict[str, Any]:
    manager = _task_manager()
    deadline = time.monotonic() + max(1, timeout_sec)
    pending = {
        width
        for width, row in payload["widths"].items()
        if row["hardware"].get("task_id") and row["status"] != "finished"
    }
    while pending and time.monotonic() < deadline:
        for width_text in list(pending):
            row = payload["widths"][width_text]
            task_id = int(row["hardware"]["task_id"])
            result = _result_with_retry(manager, task_id)
            if not isinstance(result, dict) or not result:
                continue
            status = str(result.get("status") or "").lower()
            if status in {"failed", "error", "cancelled", "canceled"}:
                row["status"] = status
                row["hardware"]["error"] = result.get("error")
                pending.remove(width_text)
                continue
            counts = result.get("count")
            if status not in {"finished", "completed", "done", "success"} or not isinstance(counts, dict):
                continue

            width = int(width_text)
            raw_path = outdir / f"raw_result_width_{width}_task_{task_id}.json"
            _atomic_json(raw_path, result)
            instance = build_width_instance(width=width, seed=int(payload["seed"]))
            analysis = analyze_counts(instance, counts)
            random_baseline = random_sampling_baseline(
                instance,
                shots=int(analysis["shots"]),
                repeats=32,
                seed=int(payload["seed"]),
            )
            executed_qasm = str(
                result.get("transpiled")
                or result.get("circuit")
                or (
                    outdir
                    / row["circuit"].get(
                        "physical_qasm_file",
                        f"width_{width}.qasm",
                    )
                ).read_text(encoding="utf-8")
            )
            compiled_metrics = qasm_gate_metrics(executed_qasm)
            measured_physical_qubits = sorted(
                {
                    int(value)
                    for value in re.findall(
                        r"measure\s+q\[(\d+)\]",
                        executed_qasm,
                        flags=re.IGNORECASE,
                    )
                }
            )
            heuristic_quality = float(row["classical"]["heuristic"]["quality"])
            analysis["random_baseline"] = random_baseline
            analysis["signal_filtered_mean_minus_random"] = (
                float(analysis["hardware_mean_quality_filtered"])
                - random_baseline["mean_quality_filtered"]
            )
            analysis["gap_top5_minus_classical_heuristic"] = (
                float(analysis["top5_best_quality"]) - heuristic_quality
            )
            row["circuit"]["compiled_depth"] = compiled_metrics["depth"]
            row["circuit"]["compiled_cnot_count"] = compiled_metrics["cnot_count"]
            row["circuit"]["compiled_two_qubit_gate_count"] = compiled_metrics[
                "two_qubit_gate_count"
            ]
            row["circuit"]["compiled_gate_count"] = compiled_metrics["gate_count"]
            row["circuit"]["executed_measured_physical_qubits"] = (
                measured_physical_qubits
            )
            row["circuit"]["mapping_verified"] = (
                measured_physical_qubits
                == sorted(int(qubit) for qubit in row["physical_chain"]["qubits"])
            )
            row["hardware"]["shots_actual"] = analysis["shots"]
            row["hardware"]["raw_result_file"] = raw_path.name
            row["analysis"] = analysis
            row["status"] = "finished"
            pending.remove(width_text)
            print(
                f"finished width={width} task_id={task_id} "
                f"signal={analysis['signal_filtered_mean_minus_random']:.6f}",
                flush=True,
            )
            _atomic_json(outdir / "scan.json", payload)
        if pending:
            print(f"pending widths={','.join(sorted(pending, key=int))}", flush=True)
            time.sleep(max(1, poll_interval_sec))
    _atomic_json(outdir / "scan.json", payload)
    return payload


def write_summary(payload: dict[str, Any], outdir: Path) -> None:
    rows = []
    for width_text, row in sorted(payload["widths"].items(), key=lambda item: int(item[0])):
        instance = row["instance"]
        chain = row["physical_chain"]
        circuit = row["circuit"]
        analysis = row.get("analysis", {})
        random_baseline = analysis.get("random_baseline", {})
        rows.append(
            {
                "width": int(width_text),
                "customers": instance["customers"],
                "vehicles": instance["vehicles"],
                "decision_variables": instance["decision_variables"],
                "actual_qubits": instance["actual_qubits"],
                "physical_qubit_mapping": json.dumps(chain["qubits"]),
                "cnot": circuit["logical_cnot_count"],
                "native_two_qubit_gates": circuit.get(
                    "compiled_two_qubit_gate_count",
                    circuit.get("physical_native_two_qubit_gate_count"),
                ),
                "circuit_depth": circuit.get(
                    "compiled_depth",
                    circuit.get("physical_depth", circuit["logical_depth"]),
                ),
                "mapping_verified": circuit.get("mapping_verified"),
                "min_coupling_fidelity": chain["minimum_fidelity"],
                "avg_coupling_fidelity": chain["average_fidelity"],
                "uncalibrated_couplings": chain["uncalibrated_couplings"],
                "estimated_circuit_fidelity": chain["estimated_circuit_fidelity"],
                "shots": analysis.get("shots", row["hardware"]["shots_requested"]),
                "raw_top1_quality": analysis.get("raw_top1_quality"),
                "raw_top1_category": analysis.get("raw_top1_category"),
                "filtered_top1_quality": analysis.get("filtered_top1_quality"),
                "top5_best_quality": analysis.get("top5_best_quality"),
                "top5_contains_exact_optimum": analysis.get("top5_contains_exact_optimum"),
                "hardware_mean_quality_filtered": analysis.get(
                    "hardware_mean_quality_filtered"
                ),
                "random_baseline": random_baseline.get("mean_quality_filtered"),
                "classical_heuristic": row["classical"]["heuristic"]["quality"],
                "exact_optimum_objective": row["classical"]["exact"]["objective"],
                "exact_optimum_bitstring": row["classical"]["exact"]["bitstring"],
                "exact_route_distance_nn_2opt": row["classical"]["exact"][
                    "route_distance_nn_2opt"
                ],
                "heuristic_route_distance_nn_2opt": row["classical"]["heuristic"][
                    "route_distance_nn_2opt"
                ],
                "signal": analysis.get("signal_filtered_mean_minus_random"),
                "optimal_solution_hit_rate": analysis.get("optimal_solution_hit_rate"),
                "feasible_ratio": analysis.get("feasible_ratio"),
                "illegal_ratio": analysis.get("illegal_ratio"),
                "extreme_ratio": analysis.get("extreme_ratio"),
                "gap_vs_classical_heuristic": analysis.get(
                    "gap_top5_minus_classical_heuristic"
                ),
                "task_id": row["hardware"].get("task_id"),
                "status": row["status"],
            }
        )
    with (outdir / "width_scan.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(outdir / "width_scan.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shenglian CVRP logical-width scan")
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--seed", type=int, default=SCAN_SEED)
    parser.add_argument("--chip", default="Shenglian")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--action",
        choices=("prepare", "submit", "collect", "run"),
        default="run",
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-interval-sec", type=int, default=10)
    args = parser.parse_args()

    widths = _parse_widths(args.widths)
    scan_path = args.outdir / "scan.json"
    if args.action in {"prepare", "run"} or not scan_path.exists():
        payload = prepare(widths, args.outdir, args.chip, args.seed, args.shots)
    else:
        payload = json.loads(scan_path.read_text(encoding="utf-8"))
    if args.action in {"submit", "run"}:
        payload = submit(payload, args.outdir, args.chip)
    if args.action in {"collect", "run"}:
        payload = collect(
            payload,
            args.outdir,
            args.timeout_sec,
            args.poll_interval_sec,
        )
    write_summary(payload, args.outdir)
    print(f"summary={args.outdir / 'width_scan.csv'}", flush=True)


if __name__ == "__main__":
    main()

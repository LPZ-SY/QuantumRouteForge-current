from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge.env import (
    env_value,
    normalize_local_proxy_environment,
    quafu_token,
)


BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Quafu queue status and optionally submit a Bell task."
    )
    parser.add_argument(
        "--submit",
        "--submit-bell",
        dest="submit",
        action="store_true",
    )
    parser.add_argument("--chip", default="")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    token = quafu_token()
    chip = (args.chip or "").strip() or env_value(
        "QUAFU_CHIP",
        "Dongling",
    )
    if not token:
        raise SystemExit("QUAFU_API_TOKEN is not configured.")
    if args.shots <= 0 or args.shots % 1024 != 0:
        raise SystemExit("--shots must be a positive multiple of 1024.")

    normalize_local_proxy_environment()
    try:
        from quark import Task
    except ImportError:
        raise SystemExit(
            "quarkstudio is not installed. Use Python 3.12 and run: "
            "pip install -U quarkstudio"
        )

    manager = Task(token)
    print(f"endpoint={getattr(Task, 'URL', '?')}")
    print(f"token_loaded_length={len(token)}")
    status = manager.status()
    print(f"queue_status={status}")

    if not args.submit:
        print("Read-only check complete.")
        return

    task = {
        "chip": chip,
        "name": "QRF_connectivity_check",
        "circuit": BELL_QASM,
        "shots": args.shots,
        "options": {
            "compiler": "qsteed",
            "correct": False,
            "open_dd": None,
            "target_qubits": [],
        },
    }
    task_id = manager.run(task)
    print(f"task_id={task_id}")
    result = manager.result(task_id, timeout=max(5.0, args.timeout))
    if not isinstance(result, dict):
        print(f"unexpected_result_type={type(result).__name__}")
        return
    print(f"status={result.get('status')}")
    counts = result.get("corrected") or result.get("count") or {}
    print(f"counts={counts}")


if __name__ == "__main__":
    main()

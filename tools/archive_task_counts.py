from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge.env import (
    normalize_local_proxy_environment,
    quafu_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch QuarkStudio task results and archive raw JSON."
    )
    parser.add_argument("task_ids", nargs="+")
    parser.add_argument(
        "--outdir",
        default="results/real_hardware_counts",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    token = quafu_token()
    if not token:
        raise SystemExit("QUAFU_API_TOKEN is not configured.")
    normalize_local_proxy_environment()

    try:
        from quark import Task
    except ImportError:
        raise SystemExit(
            "quarkstudio is not installed; use the Python 3.12 environment."
        )

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manager = Task(token)
    for raw_task_id in args.task_ids:
        task_id = str(raw_task_id).strip()
        if not task_id.isdigit():
            print(f"skip invalid task id: {task_id!r}")
            continue
        result = manager.result(
            int(task_id),
            timeout=max(5, int(args.timeout)),
        )
        output_path = outdir / f"task_{task_id}.json"
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        counts = {}
        status = "unknown"
        if isinstance(result, dict):
            counts = result.get("corrected") or result.get("count") or {}
            status = result.get("status") or "unknown"
        print(
            f"task_id={task_id} status={status} "
            f"outcomes={len(counts)} shots={sum(counts.values()) if counts else 0} "
            f"saved={output_path}"
        )


if __name__ == "__main__":
    main()

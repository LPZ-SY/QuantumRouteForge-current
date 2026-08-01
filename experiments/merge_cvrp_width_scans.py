from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed CVRP width scans")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    by_width: dict[int, dict[str, object]] = {}
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                width = int(row["width"])
                if width in by_width:
                    raise SystemExit(f"duplicate width {width} in {path}")
                if row.get("status") != "finished":
                    raise SystemExit(f"width {width} is not finished in {path}")
                if row.get("mapping_verified", "").lower() != "true":
                    raise SystemExit(f"width {width} mapping is not verified in {path}")
                by_width[width] = dict(row)

    rows = [by_width[width] for width in sorted(by_width)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

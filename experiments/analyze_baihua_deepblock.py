from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


def _float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = list(values)
    if not rows:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(rows)
    std = statistics.stdev(rows) if len(rows) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
    return {
        "n": len(rows),
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _paired_pvalue(differences: list[float]) -> float | None:
    if len(differences) < 2 or all(abs(value - differences[0]) <= 1e-15 for value in differences):
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(stats.ttest_1samp(differences, popmean=0.0, alternative="two-sided").pvalue)


def analyze(input_dir: Path) -> dict[str, object]:
    with (input_dir / "instance_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in rows}
    seeds = sorted({seed for seed, _arm in by_seed_arm})
    arms = sorted({arm for _seed, arm in by_seed_arm})
    no_headroom_ratio = (
        sum(
            not _bool(next(row["has_refinement_space"] for (row_seed, _), row in by_seed_arm.items() if row_seed == seed))
            for seed in seeds
        )
        / len(seeds)
        if seeds
        else 0.0
    )
    arm_results: dict[str, object] = {}
    for arm in arms:
        arm_rows = [by_seed_arm[(seed, arm)] for seed in seeds if (seed, arm) in by_seed_arm]
        all_improvements = [_float(row["improvement_vs_baseline"]) or 0.0 for row in arm_rows]
        headroom_rows = [row for row in arm_rows if _bool(row["has_refinement_space"])]
        utilization = [
            value
            for row in headroom_rows
            if (value := _float(row.get("headroom_utilization"))) is not None
        ]
        random_differences: list[float] = []
        wins = ties = losses = 0
        for row in arm_rows:
            seed = int(row["seed"])
            random_row = by_seed_arm.get((seed, "random"))
            if random_row is None or arm == "random":
                continue
            arm_distance = _float(row["final_distance"])
            random_distance = _float(random_row["final_distance"])
            if arm_distance is None or random_distance is None:
                continue
            difference = arm_distance - random_distance
            random_differences.append(difference)
            if difference < -1e-9:
                wins += 1
            elif difference > 1e-9:
                losses += 1
            else:
                ties += 1
        arm_results[arm] = {
            "all_instances_improvement": _summary(all_improvements),
            "headroom_instances_improvement": _summary(
                [_float(row["improvement_vs_baseline"]) or 0.0 for row in headroom_rows]
            ),
            "headroom_utilization": _summary(utilization),
            "difference_vs_random": _summary(random_differences),
            "paired_two_sided_ttest_pvalue": _paired_pvalue(random_differences),
            "win_tie_loss_vs_random": {"win": wins, "tie": ties, "loss": losses},
            "statuses": sorted({row.get("status", "") for row in arm_rows}),
        }
    return {
        "statistical_unit": "independent_seed",
        "seeds": seeds,
        "arms": arm_results,
        "no_refinement_space_ratio": no_headroom_ratio,
        "all_instance_count": len(seeds),
        "headroom_instance_count": sum(
            _bool(next(row["has_refinement_space"] for (row_seed, _), row in by_seed_arm.items() if row_seed == seed))
            for seed in seeds
        ),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(input_dir: Path, analysis: dict[str, object]) -> None:
    lines = [
        "# Baihua DeepBlock experiment report",
        "",
        "Statistical unit: independent seed. All instances are retained; the headroom subset is reported alongside the full result.",
        "",
        f"- All instances: {analysis['all_instance_count']}",
        f"- Instances with refinement space: {analysis['headroom_instance_count']}",
        f"- No-refinement-space ratio: {_fmt(analysis['no_refinement_space_ratio'])}",
        "",
        "## Arm summary",
        "",
        "| Arm | All-instance mean improvement | Headroom mean improvement | Mean headroom utilization | Mean distance difference vs random | Win/Tie/Loss vs random | Two-sided paired p-value |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, payload in analysis["arms"].items():
        wtl = payload["win_tie_loss_vs_random"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _fmt(payload["all_instances_improvement"]["mean"]),
                    _fmt(payload["headroom_instances_improvement"]["mean"]),
                    _fmt(payload["headroom_utilization"]["mean"]),
                    _fmt(payload["difference_vs_random"]["mean"]),
                    f"{wtl['win']}/{wtl['tie']}/{wtl['loss']}",
                    _fmt(payload["paired_two_sided_ttest_pvalue"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These results measure candidate quality and recovered local-refinement headroom under a fixed sampling/evaluation budget. They do not establish quantum speedup over OR-Tools, exact enumeration, or other classical algorithms.",
            "",
        ]
    )
    (input_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Baihua DeepBlock paired seed results")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-by-headroom", action="store_true", help="Retained for protocol-compatible CLI; both layers are always reported.")
    args = parser.parse_args()
    payload = analyze(args.input)
    (args.input / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.input, payload)
    print(json.dumps({"input": str(args.input), "seeds": payload["all_instance_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

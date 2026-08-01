from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.candidate_evaluator import evaluate_counts  # noqa: E402
from quantum_route_forge.experiment_logger import atomic_json, write_csv  # noqa: E402
from quantum_route_forge.models import Customer  # noqa: E402
from replay_baihua_counts import _proxy  # noqa: E402


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _ranks(left)
    y = _ranks(right)
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    return numerator / (x_norm * y_norm) if x_norm > 0 and y_norm > 0 else None


def analyze_record(path: Path, candidate_ks: list[int]) -> list[dict[str, object]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not str(record["run"]["arm"]).startswith("baihua") or not record["run"].get("counts"):
        return []
    customers = [Customer(**row) for row in record["instance"]["customers"]]
    by_id = {customer.customer_id: customer for customer in customers}
    assignments = {
        int(vehicle): [by_id[int(customer_id)] for customer_id in customer_ids]
        for vehicle, customer_ids in record["assignments_before"].items()
    }
    proxy = _proxy(record["proxy"])
    block_customers = [by_id[customer_id] for customer_id in proxy.customer_ids]
    fairness = record["fairness"]
    batch, _ = evaluate_counts(
        arm="baihua",
        counts=record["run"]["counts"],
        proxy=proxy,
        assignments=assignments,
        block_customers=block_customers,
        depot=tuple(float(value) for value in record["instance"]["depot"]),
        vehicle_capacity=int(record["instance"]["vehicle_capacity"]),
        candidate_k=1 << proxy.width,
        filter_extremes=bool(fairness["filter_extremes"]),
        routing_method=str(fairness["routing_method"]),
    )
    ranked = list(batch.top_frequency)
    finite = [row for row in ranked if math.isfinite(row.true_distance)]
    correlation = _spearman(
        [row.proxy_energy for row in finite],
        [row.true_distance for row in finite],
    )
    best_improvement = max((row.improvement for row in finite), default=0.0)
    source_config = json.loads((path.parents[1] / "config.json").read_text(encoding="utf-8"))
    depth = int(source_config["deepblock"]["qaoa_depth"])
    rows: list[dict[str, object]] = []
    for candidate_k in candidate_ks:
        selected = [
            row
            for row in ranked[:candidate_k]
            if row.feasible_after_repair and row.improvement > 1e-9
        ]
        improvement = max((row.improvement for row in selected), default=0.0)
        probability_mass = sum(row.probability for row in ranked[:candidate_k])
        rows.append(
            {
                "source": path.parents[1].name,
                "depth": depth,
                "seed": int(record["seed"]),
                "iteration": int(record["iteration"]),
                "block_id": record["block"]["block_id"],
                "candidate_k": candidate_k,
                "unique_candidates": batch.unique_candidates,
                "online_hit": improvement > 1e-9,
                "online_best_improvement": improvement,
                "best_of_all_sampled_improvement": max(0.0, best_improvement),
                "candidate_probability_mass": probability_mass,
                "proxy_true_spearman": correlation,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["depth"]), int(row["candidate_k"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (depth, candidate_k), values in sorted(grouped.items()):
        improvements = [float(row["online_best_improvement"]) for row in values]
        correlations = [
            float(row["proxy_true_spearman"])
            for row in values
            if row["proxy_true_spearman"] is not None
        ]
        summaries.append(
            {
                "depth": depth,
                "candidate_k": candidate_k,
                "subproblems": len(values),
                "hit_count": sum(bool(row["online_hit"]) for row in values),
                "hit_rate": statistics.fmean(bool(row["online_hit"]) for row in values),
                "mean_best_improvement": statistics.fmean(improvements),
                "median_best_improvement": statistics.median(improvements),
                "mean_probability_mass": statistics.fmean(
                    float(row["candidate_probability_mass"]) for row in values
                ),
                "mean_proxy_true_spearman": statistics.fmean(correlations) if correlations else None,
            }
        )
    return summaries


def write_report(outdir: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Baihua sampled-candidate budget diagnostic",
        "",
        "Each row is a logged hardware subproblem evaluated at its recorded pre-decision state. This is a counterfactual candidate-budget diagnostic, not a closed-loop final-route result.",
        "",
        "| Depth | k | Improving blocks | Hit rate | Mean local improvement | Top-k probability mass | Proxy/true Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        corr = row["mean_proxy_true_spearman"]
        lines.append(
            f"| p={row['depth']} | {row['candidate_k']} | {row['hit_count']}/{row['subproblems']} | "
            f"{float(row['hit_rate']):.1%} | {float(row['mean_best_improvement']):.6f} | "
            f"{float(row['mean_probability_mass']):.1%} | {float(corr):.4f} |"
        )
    lines.extend(
        [
            "",
            "The proxy energy is retained for circuit construction, but the near-zero aggregate rank correlation means it is not used to claim reliable downstream route ranking.",
            "",
        ]
    )
    (outdir / "candidate_budget_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate budgets using existing Baihua counts")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-ks", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    candidate_ks = sorted({int(value) for value in args.candidate_ks.split(",") if value.strip()})
    rows: list[dict[str, object]] = []
    for input_dir in args.inputs:
        for path in sorted((input_dir / "replay").glob("seed_*_baihua*.json")):
            rows.extend(analyze_record(path, candidate_ks))
    summaries = summarize(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "candidate_budget_subproblems.csv", rows)
    write_csv(args.outdir / "candidate_budget_summary.csv", summaries)
    atomic_json(args.outdir / "candidate_budget_summary.json", summaries)
    write_report(args.outdir, summaries)
    print(json.dumps({"subproblem_rows": len(rows), "summary_rows": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

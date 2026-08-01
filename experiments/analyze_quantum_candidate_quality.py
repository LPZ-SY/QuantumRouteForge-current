from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_route_forge.candidate_evaluator import evaluate_counts  # noqa: E402
from quantum_route_forge.experiment_logger import atomic_json, write_csv  # noqa: E402
from quantum_route_forge.models import Customer  # noqa: E402
from quantum_route_forge.qaoa_depth_runner import QAOAParameters, simulate_counts  # noqa: E402
from quantum_route_forge.sparse_proxy_qubo import bitstring_to_bits  # noqa: E402
from replay_baihua_counts import _proxy  # noqa: E402


def _mass(counts: dict[str, int], selected: set[str]) -> float:
    shots = sum(int(value) for value in counts.values())
    return sum(int(counts.get(bitstring, 0)) for bitstring in selected) / max(1, shots)


def _sign_flip_pvalue(values: list[float]) -> float | None:
    if not values:
        return None
    observed = abs(statistics.fmean(values))
    permutations = [
        abs(statistics.fmean(value * sign for value, sign in zip(values, signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in permutations) / len(permutations)


def _bootstrap_ci(values: list[float], samples: int = 20_000) -> tuple[float, float]:
    rng = random.Random(20260801)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def analyze_record(
    path: Path,
    *,
    energy_tail_fraction: float,
    route_improvement_fraction: float,
    ideal_shots: int,
) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    proxy = _proxy(record["proxy"])
    width = proxy.width
    all_bitstrings = [format(value, f"0{width}b") for value in range(1 << width)]
    energy_ranked = sorted(
        all_bitstrings,
        key=lambda bitstring: (proxy.energy(bitstring_to_bits(bitstring, width)), bitstring),
    )
    tail_size = max(1, math.ceil(energy_tail_fraction * len(all_bitstrings)))
    low_energy = set(energy_ranked[:tail_size])
    energy_threshold = max(proxy.energy(bitstring_to_bits(bitstring, width)) for bitstring in low_energy)

    customers = [Customer(**row) for row in record["instance"]["customers"]]
    by_id = {customer.customer_id: customer for customer in customers}
    assignments = {
        int(vehicle): [by_id[int(customer_id)] for customer_id in customer_ids]
        for vehicle, customer_ids in record["assignments_before"].items()
    }
    block_customers = [by_id[customer_id] for customer_id in proxy.customer_ids]
    uniform_counts = {bitstring: 1 for bitstring in all_bitstrings}
    fairness = record["fairness"]
    uniform_batch, _ = evaluate_counts(
        arm="uniform_exact",
        counts=uniform_counts,
        proxy=proxy,
        assignments=assignments,
        block_customers=block_customers,
        depot=tuple(float(value) for value in record["instance"]["depot"]),
        vehicle_capacity=int(record["instance"]["vehicle_capacity"]),
        candidate_k=1 << width,
        filter_extremes=False,
        routing_method=str(fairness["routing_method"]),
    )
    rows = {row.bitstring: row for row in uniform_batch.top_frequency}
    baseline = next(iter(rows.values())).true_distance + next(iter(rows.values())).improvement
    improving = {
        bitstring
        for bitstring, row in rows.items()
        if row.feasible_after_repair
        and row.improvement / max(1e-12, baseline) >= route_improvement_fraction
    }
    combined = low_energy & improving

    parameter_payload = record["run"]["parameters"]
    parameters = QAOAParameters(
        depth=int(parameter_payload["depth"]),
        gamma=tuple(float(value) for value in parameter_payload["gamma"]),
        beta=tuple(float(value) for value in parameter_payload["beta"]),
        optimizer=str(parameter_payload["optimizer"]),
        initial_value=float(parameter_payload["initial_value"]),
        final_value=float(parameter_payload["final_value"]),
        evaluations=int(parameter_payload["evaluations"]),
    )
    ideal_counts = simulate_counts(
        proxy,
        parameters,
        shots=ideal_shots,
        seed=int(record["seed"]) * 10_000 + int(record["iteration"]),
    )
    hardware_counts = {str(key): int(value) for key, value in record["run"]["counts"].items()}
    uniform_mass_low = len(low_energy) / len(all_bitstrings)
    uniform_mass_improving = len(improving) / len(all_bitstrings)
    uniform_mass_combined = len(combined) / len(all_bitstrings)
    hardware_mass_low = _mass(hardware_counts, low_energy)
    hardware_mass_improving = _mass(hardware_counts, improving)
    hardware_mass_combined = _mass(hardware_counts, combined)
    ideal_mass_low = _mass(ideal_counts, low_energy)
    ideal_mass_improving = _mass(ideal_counts, improving)
    ideal_mass_combined = _mass(ideal_counts, combined)
    return {
        "source": path.parents[1].name,
        "seed": int(record["seed"]),
        "iteration": int(record["iteration"]),
        "block_id": record["block"]["block_id"],
        "width": width,
        "shots": sum(hardware_counts.values()),
        "energy_tail_fraction": energy_tail_fraction,
        "energy_tail_states": tail_size,
        "energy_threshold": energy_threshold,
        "route_improvement_fraction": route_improvement_fraction,
        "improving_states": len(improving),
        "combined_states": len(combined),
        "hardware_low_energy_mass": hardware_mass_low,
        "uniform_low_energy_mass": uniform_mass_low,
        "ideal_low_energy_mass": ideal_mass_low,
        "hardware_low_energy_enrichment": hardware_mass_low / uniform_mass_low,
        "hardware_minus_uniform_low_energy_mass": hardware_mass_low - uniform_mass_low,
        "hardware_improving_mass": hardware_mass_improving,
        "uniform_improving_mass": uniform_mass_improving,
        "ideal_improving_mass": ideal_mass_improving,
        "hardware_improving_enrichment": (
            hardware_mass_improving / uniform_mass_improving
            if uniform_mass_improving > 0
            else None
        ),
        "hardware_minus_uniform_improving_mass": hardware_mass_improving - uniform_mass_improving,
        "hardware_combined_mass": hardware_mass_combined,
        "uniform_combined_mass": uniform_mass_combined,
        "ideal_combined_mass": ideal_mass_combined,
        "hardware_combined_enrichment": (
            hardware_mass_combined / uniform_mass_combined
            if uniform_mass_combined > 0
            else None
        ),
        "hardware_minus_uniform_combined_mass": hardware_mass_combined - uniform_mass_combined,
        "task_positive_low_energy_enrichment": hardware_mass_low > uniform_mass_low,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seed_rows: list[dict[str, object]] = []
    metrics = ("low_energy", "improving", "combined")
    for seed, values in sorted(by_seed.items()):
        payload: dict[str, object] = {"seed": seed, "subproblems": len(values)}
        for metric in metrics:
            for source in ("hardware", "uniform", "ideal"):
                payload[f"{source}_{metric}_mass"] = statistics.fmean(
                    float(row[f"{source}_{metric}_mass"]) for row in values
                )
            payload[f"hardware_minus_uniform_{metric}_mass"] = (
                float(payload[f"hardware_{metric}_mass"])
                - float(payload[f"uniform_{metric}_mass"])
            )
        seed_rows.append(payload)
    tests: dict[str, object] = {}
    for metric in metrics:
        differences = [float(row[f"hardware_minus_uniform_{metric}_mass"]) for row in seed_rows]
        low, high = _bootstrap_ci(differences)
        tests[metric] = {
            "statistical_unit": "independent_seed",
            "n": len(seed_rows),
            "mean_hardware_mass": statistics.fmean(float(row[f"hardware_{metric}_mass"]) for row in seed_rows),
            "mean_uniform_mass": statistics.fmean(float(row[f"uniform_{metric}_mass"]) for row in seed_rows),
            "mean_ideal_mass": statistics.fmean(float(row[f"ideal_{metric}_mass"]) for row in seed_rows),
            "mean_hardware_minus_uniform": statistics.fmean(differences),
            "bootstrap_ci95": [low, high],
            "exact_sign_flip_two_sided_pvalue": _sign_flip_pvalue(differences),
            "positive_seed_count": sum(value > 0 for value in differences),
        }
    return {"seed_rows": seed_rows, "tests": tests}


def write_report(outdir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    test = summary["tests"]
    low = test["low_energy"]
    improving = test["improving"]
    combined = test["combined"]
    low_ci = low["bootstrap_ci95"]
    confirmatory_pass = low_ci[0] > 0 and float(low["exact_sign_flip_two_sided_pvalue"]) < 0.05
    lines = [
        "# Independent quantum-candidate quality analysis",
        "",
        "## Frozen candidate-quality definition",
        "",
        "A quantum-native high-quality candidate is a bitstring in the exact bottom 10% of proxy-QUBO energies for its own 8-bit subproblem. The rank threshold is computed from the QUBO only, before reading hardware counts, and is therefore comparable across differently scaled blocks.",
        "",
        "## Existing formal-count result (exploratory)",
        "",
        "| Primary event | Hardware mass | Uniform mass | Ideal-QAOA mass | Hardware-uniform | 95% seed bootstrap CI | Exact sign-flip p | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ci = low["bootstrap_ci95"]
    lines.append(
        f"| Bottom-10% BQM energy | {100*float(low['mean_hardware_mass']):.2f}% | {100*float(low['mean_uniform_mass']):.2f}% | "
        f"{100*float(low['mean_ideal_mass']):.2f}% | {100*float(low['mean_hardware_minus_uniform']):+.2f} pp | "
        f"[{100*float(ci[0]):+.2f}, {100*float(ci[1]):+.2f}] pp | {float(low['exact_sign_flip_two_sided_pvalue']):.5f} | "
        f"{low['positive_seed_count']}/{low['n']} |"
    )
    lines.extend(
        [
            "",
            f"Confirmatory decision rule: the bottom-10% energy-mass difference must have a seed-bootstrap lower bound above zero and an exact paired p-value below 0.05. Existing-data pass: **{confirmatory_pass}**.",
            "",
            "This analysis is exploratory because the threshold was introduced after the existing hardware data had already been collected. The exact definition and decision rule above are now frozen for a future confirmatory hardware batch; they must not be tuned after seeing that batch.",
            "",
            "This gate is intentionally independent of capacity repair, route construction, top-k acceptance, and final CVRP distance. Passing it establishes a positive quantum-sampling contribution in candidate quality relative to uniform sampling; it is not a quantum-speedup claim.",
            "",
            "## Optional downstream diagnostic (not part of the gate)",
            "",
            f"The >=0.5% route-improvement mass was {100*float(improving['mean_hardware_mass']):.2f}% on hardware versus {100*float(improving['mean_uniform_mass']):.2f}% under uniform sampling. The combined low-energy-and-route-improvement mass was {100*float(combined['mean_hardware_mass']):.2f}% versus {100*float(combined['mean_uniform_mass']):.2f}%. These values diagnose objective alignment only and do not alter the quantum-contribution decision.",
            "",
        ]
    )
    (outdir / "quantum_candidate_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure(outdir: Path, summary: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    tests = summary["tests"]
    metric = "low_energy"
    sources = ("hardware", "uniform", "ideal")
    colors = {"hardware": "#6A4C93", "uniform": "#8D99AE", "ideal": "#1982C4"}
    width = 0.58
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for offset, source in enumerate(sources):
        value = 100 * float(tests[metric][f"mean_{source}_mass"])
        ax.bar(
            [offset],
            [value],
            width,
            label=source,
            color=colors[source],
        )
        ax.text(offset, value + 1, f"{value:.2f}%", ha="center")
    ax.set_xticks(range(len(sources)), sources)
    ax.set_ylabel("Probability mass (%)")
    ax.set_title("Probability mass in the pre-defined bottom-10% BQM-energy region")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_quantum_candidate_quality.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent quantum candidate-quality test")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--energy-tail-fraction", type=float, default=0.10)
    parser.add_argument("--route-improvement-fraction", type=float, default=0.005)
    parser.add_argument("--ideal-shots", type=int, default=262_144)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.energy_tail_fraction < 1:
        raise SystemExit("energy-tail-fraction must be between 0 and 1")
    if args.route_improvement_fraction < 0:
        raise SystemExit("route-improvement-fraction must be non-negative")
    rows: list[dict[str, object]] = []
    for input_dir in args.inputs:
        for path in sorted((input_dir / "replay").glob("seed_*_baihua*.json")):
            rows.append(
                analyze_record(
                    path,
                    energy_tail_fraction=args.energy_tail_fraction,
                    route_improvement_fraction=args.route_improvement_fraction,
                    ideal_shots=args.ideal_shots,
                )
            )
    summary = summarize(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "quantum_candidate_quality_subproblems.csv", rows)
    write_csv(args.outdir / "quantum_candidate_quality_seeds.csv", summary["seed_rows"])
    atomic_json(args.outdir / "quantum_candidate_quality_summary.json", summary)
    atomic_json(
        args.outdir / "frozen_confirmatory_protocol.json",
        {
            "schema_version": 1,
            "frozen_after_exploratory_batch": True,
            "candidate_definition": "exact bottom 10% proxy-QUBO energy rank within each subproblem",
            "energy_tail_fraction": args.energy_tail_fraction,
            "primary_statistical_unit": "independent_seed",
            "primary_comparator": "exact uniform distribution over the same bitstrings and QUBO",
            "decision_rule": "bootstrap CI95 lower bound of hardware-minus-uniform low-energy mass > 0 and exact paired sign-flip p < 0.05",
            "no_posthoc_threshold_changes": True,
            "downstream_route_metrics_part_of_primary_gate": False,
        },
    )
    low_energy = summary["tests"]["low_energy"]
    low_ci = low_energy["bootstrap_ci95"]
    atomic_json(
        args.outdir / "quantum_contribution_decision.json",
        {
            "threshold": "exact bottom 10% proxy-QUBO energy rank within each subproblem",
            "hardware_probability_mass": low_energy["mean_hardware_mass"],
            "uniform_probability_mass": low_energy["mean_uniform_mass"],
            "enrichment_ratio": float(low_energy["mean_hardware_mass"]) / float(low_energy["mean_uniform_mass"]),
            "hardware_minus_uniform": low_energy["mean_hardware_minus_uniform"],
            "bootstrap_ci95": low_ci,
            "exact_sign_flip_two_sided_pvalue": low_energy["exact_sign_flip_two_sided_pvalue"],
            "positive_seeds": low_energy["positive_seed_count"],
            "total_seeds": low_energy["n"],
            "existing_exploratory_batch_positive": bool(
                float(low_ci[0]) > 0
                and float(low_energy["exact_sign_flip_two_sided_pvalue"]) < 0.05
            ),
            "downstream_route_metrics_used_for_decision": False,
        },
    )
    write_report(args.outdir, rows, summary)
    write_figure(args.outdir, summary)
    print(json.dumps({"subproblems": len(rows), "seeds": len(summary["seed_rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

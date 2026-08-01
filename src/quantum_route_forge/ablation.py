from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Iterable

from .tabu_qaoa import TabuIterationRecord


def summarize_paired_ablation(
    records: Iterable[TabuIterationRecord],
) -> dict[str, object]:
    """Summarize paired quantum/classical proposals by actual qubit count.

    Each record compares the sampled quantum partition with a deterministic
    exact Max-Cut shadow proposal on the same vehicle pair, customers, sparse
    graph, current assignment, and Tabu state. The shadow proposal never
    changes the main search trajectory.
    """
    materialized = list(records)
    grouped: dict[int, list[TabuIterationRecord]] = defaultdict(list)
    for record in materialized:
        grouped[int(record.sub_k)].append(record)

    layers = {}
    for sub_k in sorted(grouped):
        layer_records = grouped[sub_k]
        quantum_improvements = [
            record.quantum_improvement for record in layer_records
        ]
        classical_improvements = [
            record.classical_improvement for record in layer_records
        ]
        paired_deltas = [
            record.paired_improvement_delta for record in layer_records
        ]
        layers[str(sub_k)] = {
            "sub_k": sub_k,
            "iterations": len(layer_records),
            "hardware_iterations": sum(
                record.sampler_source
                in {"quafu_hardware", "quafu_existing_task"}
                for record in layer_records
            ),
            "accepted_quantum_moves": sum(
                record.accepted for record in layer_records
            ),
            "mean_n_edges": fmean(
                record.n_edges for record in layer_records
            ),
            "mean_sub_cnot": fmean(
                record.sub_cnot for record in layer_records
            ),
            "mean_quantum_improvement": fmean(
                quantum_improvements
            ),
            "mean_classical_improvement": fmean(
                classical_improvements
            ),
            "mean_delta_b_minus_a": fmean(paired_deltas),
            "mean_quantum_cut": fmean(
                record.cut_proposed for record in layer_records
            ),
            "mean_classical_cut": fmean(
                record.classical_cut for record in layer_records
            ),
        }

    return {
        "comparison": "paired_counterfactual_shadow",
        "trajectory_owner": "quantum_or_configured_sampler",
        "classical_control": "exact_maxcut_shadow",
        "pairing_invariant": (
            "same iteration, vehicle pair, selected customers, sparse graph, "
            "assignment state, and Tabu state"
        ),
        "total_iterations": len(materialized),
        "layers": layers,
    }

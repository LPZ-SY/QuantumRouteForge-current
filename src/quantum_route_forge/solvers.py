from __future__ import annotations

from dataclasses import dataclass

import dimod


@dataclass(frozen=True)
class SolverRun:
    sample: dict[str, int]
    energy: float
    used_mode: str
    message: str


def solve_bqm_classical(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 120,
    num_sweeps: int = 40,
) -> SolverRun:
    """Solve BQM with local simulated annealing (fully local classical baseline)."""
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm,
        num_reads=max(20, int(num_reads)),
        num_sweeps=max(10, int(num_sweeps)),
    )
    best = sampleset.first
    return SolverRun(
        sample=dict(best.sample),
        energy=float(best.energy),
        used_mode="classical",
        message="Solved with dimod.SimulatedAnnealingSampler.",
    )

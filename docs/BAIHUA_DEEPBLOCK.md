# Baihua DeepBlock implementation protocol

## Scope

This workflow adds an 8-qubit, depth-1/2/3 QAOA candidate generator for local CVRP refinement. It does not replace the full CVRP objective, capacity checks, route optimization, acceptance rule, or the existing Shenglian width-scan workflow.

## Module boundaries

- `deepblock_builder.py`: selects one two-vehicle boundary pool and builds 1-3 interaction-aware overlapping blocks.
- `baihua_topology.py`: normalizes current calibration fields, validates manual mappings, and searches a fidelity-first calibrated physical path of width at most 8.
- `sparse_proxy_qubo.py`: computes linear reassignment effects and pair synergies, then keeps only directly supported physical interactions.
- `qaoa_depth_runner.py`: builds p=1/2/3 circuits, pre-trains parameters in a noiseless statevector, simulates counts, compiles for Baihua, audits the circuit, and guards hardware submission.
- `candidate_evaluator.py`: applies a common shots/candidate budget, repairs capacity, recomputes full route distance, and separates top-frequency-k from best-of-shots.
- `experiment_logger.py`: writes configs, calibration snapshots, counts, circuits, mappings, replay records, CSV data, and reports.
- `deepblock_solver.py`: coordinates forward/reverse scans and updates the objective immediately after accepted moves.

## Fairness contract

`random`, `sim`, and `baihua` receive the same `shots`, `candidate_k`, extreme-string filter, capacity repair, route scorer, and strict-improvement acceptance test. The `exact` arm enumerates the local block only to estimate refinement headroom and is not a sampling-budget competitor.

The raw ranked distribution is retained even when extreme strings are filtered. Only the first `candidate_k` eligible frequency-ranked strings can affect the pipeline. `best_of_shots` is diagnostic and never expands the acceptance budget.

## Hardware protection

Hardware submission is off by default. A task can be sent only when all of the following hold:

1. Current calibration or a user-supplied calibration snapshot yields a connected path with no uncalibrated edge.
2. The physical circuit passes mapping, SWAP, CNOT, and depth checks.
3. Both `--submit-hardware` and `--confirm-hardware-submit` are present.
4. A Quafu token is available.

Offline identity mappings are labeled `simulator_identity_not_hardware_calibration`; they are never presented as current hardware calibration and cannot pass the hardware mapping audit.

## Result interpretation

The independent seed is the statistical unit. Reports always include all seeds, the subset with exact local headroom, and the no-headroom proportion. A dry-run Baihua arm has no hardware counts and must not be interpreted as a hardware result. Neither local candidate quality nor a single hardware run establishes quantum speedup.

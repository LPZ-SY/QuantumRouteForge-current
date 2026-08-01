from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
import time
from typing import Mapping, Sequence

import numpy as np

from .cvrp_width_scan import qasm_gate_metrics
from .sparse_proxy_qubo import SparseProxyQUBO


@dataclass(frozen=True)
class QAOAParameters:
    depth: int
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    optimizer: str
    initial_value: float
    final_value: float
    evaluations: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompilationAudit:
    cnot_count: int
    swap_count: int
    depth: int
    two_qubit_gate_layers: int
    mapping_verified: bool
    uncalibrated_couplings: int
    passed: bool
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QAOARunResult:
    arm: str
    counts: dict[str, int]
    qasm: str
    physical_qasm: str
    parameters: QAOAParameters
    compilation: CompilationAudit
    shots: int
    seed: int
    task_id: str = ""
    backend: str = ""
    message: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "counts": dict(self.counts),
            "qasm": self.qasm,
            "physical_qasm": self.physical_qasm,
            "parameters": self.parameters.payload(),
            "compilation": self.compilation.payload(),
            "shots": self.shots,
            "seed": self.seed,
            "task_id": self.task_id,
            "backend": self.backend,
            "message": self.message,
        }


def _validate_parameters(
    depth: int,
    gamma: Sequence[float],
    beta: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    depth = int(depth)
    if depth not in {1, 2, 3}:
        raise ValueError("QAOA depth must be p=1, p=2, or p=3.")
    gamma_values = tuple(float(value) for value in gamma)
    beta_values = tuple(float(value) for value in beta)
    if len(gamma_values) != depth or len(beta_values) != depth:
        raise ValueError("gamma and beta must contain one value per QAOA layer.")
    return gamma_values, beta_values


def build_qaoa_qasm(
    proxy: SparseProxyQUBO,
    depth: int,
    gamma: Sequence[float],
    beta: Sequence[float],
) -> str:
    gamma_values, beta_values = _validate_parameters(depth, gamma, beta)
    width = proxy.width
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{width}];",
        f"creg c[{width}];",
    ]
    lines.extend(f"h q[{index}];" for index in range(width))

    # QUBO x=(1-Z)/2 conversion. Global phases are omitted.
    ising_linear = [-coefficient / 2.0 for coefficient in proxy.linear]
    ising_quadratic: list[tuple[int, int, float]] = []
    for row in proxy.kept_interactions:
        ising_linear[row.left] -= row.coefficient / 4.0
        ising_linear[row.right] -= row.coefficient / 4.0
        ising_quadratic.append((row.left, row.right, row.coefficient / 4.0))

    for layer, (layer_gamma, layer_beta) in enumerate(zip(gamma_values, beta_values), start=1):
        lines.append(f"// qaoa_layer_{layer}")
        for index, coefficient in enumerate(ising_linear):
            if abs(coefficient) > 1e-15:
                lines.append(f"rz({2.0 * layer_gamma * coefficient:.12g}) q[{index}];")
        for left, right, coefficient in ising_quadratic:
            lines.append(f"cx q[{left}],q[{right}];")
            lines.append(f"rz({2.0 * layer_gamma * coefficient:.12g}) q[{right}];")
            lines.append(f"cx q[{left}],q[{right}];")
        lines.extend(f"rx({2.0 * layer_beta:.12g}) q[{index}];" for index in range(width))
    lines.extend(f"measure q[{index}] -> c[{index}];" for index in range(width))
    return "\n".join(lines) + "\n"


def _statevector(
    proxy: SparseProxyQUBO,
    gamma: Sequence[float],
    beta: Sequence[float],
) -> np.ndarray:
    width = proxy.width
    size = 1 << width
    state = np.ones(size, dtype=np.complex128) / math.sqrt(size)
    energies = np.array(
        [proxy.energy(tuple((basis >> index) & 1 for index in range(width))) for basis in range(size)],
        dtype=np.float64,
    )
    for layer_gamma, layer_beta in zip(gamma, beta):
        state *= np.exp(-1j * float(layer_gamma) * energies)
        cosine = math.cos(float(layer_beta))
        minus_i_sine = -1j * math.sin(float(layer_beta))
        for qubit in range(width):
            step = 1 << qubit
            for base in range(0, size, step * 2):
                for offset in range(step):
                    zero = base + offset
                    one = zero + step
                    amp_zero, amp_one = state[zero], state[one]
                    state[zero] = cosine * amp_zero + minus_i_sine * amp_one
                    state[one] = minus_i_sine * amp_zero + cosine * amp_one
    return state


def expected_proxy_energy(
    proxy: SparseProxyQUBO,
    gamma: Sequence[float],
    beta: Sequence[float],
) -> float:
    state = _statevector(proxy, gamma, beta)
    probabilities = np.abs(state) ** 2
    energies = np.array(
        [proxy.energy(tuple((basis >> index) & 1 for index in range(proxy.width))) for basis in range(1 << proxy.width)],
        dtype=np.float64,
    )
    return float(np.dot(probabilities, energies))


def pretrain_parameters(
    proxy: SparseProxyQUBO,
    depth: int,
    initial_gamma: Sequence[float] | None = None,
    initial_beta: Sequence[float] | None = None,
    rounds: int = 2,
) -> QAOAParameters:
    depth = int(depth)
    gamma = list(initial_gamma or [0.8] * depth)
    beta = list(initial_beta or [0.4] * depth)
    _validate_parameters(depth, gamma, beta)
    initial_value = expected_proxy_energy(proxy, gamma, beta)
    best_value = initial_value
    evaluations = 1
    step = 0.40
    for _round in range(max(0, int(rounds))):
        for family in (gamma, beta):
            for index in range(depth):
                original = family[index]
                candidates = [original - step, original, original + step]
                scored = []
                for candidate in candidates:
                    family[index] = candidate
                    value = expected_proxy_energy(proxy, gamma, beta)
                    evaluations += 1
                    scored.append((value, candidate))
                best_value, chosen = min(scored, key=lambda item: (item[0], item[1]))
                family[index] = chosen
        step *= 0.5
    final_value = expected_proxy_energy(proxy, gamma, beta)
    evaluations += 1
    return QAOAParameters(
        depth=depth,
        gamma=tuple(gamma),
        beta=tuple(beta),
        optimizer="deterministic_coordinate_search_statevector",
        initial_value=float(initial_value),
        final_value=float(final_value),
        evaluations=evaluations,
    )


def simulate_counts(
    proxy: SparseProxyQUBO,
    parameters: QAOAParameters,
    shots: int,
    seed: int,
) -> dict[str, int]:
    shots = max(1, int(shots))
    state = _statevector(proxy, parameters.gamma, parameters.beta)
    probabilities = np.abs(state) ** 2
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(int(seed))
    sampled = rng.multinomial(shots, probabilities)
    return {
        format(index, f"0{proxy.width}b"): int(count)
        for index, count in enumerate(sampled)
        if count > 0
    }


def random_counts(width: int, shots: int, seed: int) -> dict[str, int]:
    width = int(width)
    shots = max(1, int(shots))
    rng = random.Random(int(seed))
    counts: dict[str, int] = {}
    for _ in range(shots):
        bitstring = format(rng.getrandbits(width), f"0{width}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def compilation_audit(
    qasm: str,
    *,
    swap_count: int | None = None,
    mapping_verified: bool = True,
    uncalibrated_couplings: int = 0,
    max_cnot: int | None = None,
    max_depth: int | None = None,
) -> CompilationAudit:
    metrics = qasm_gate_metrics(qasm)
    two_qubit_depths: dict[int, int] = {}
    two_qubit_layers = 0
    for raw_line in str(qasm or "").splitlines():
        line = raw_line.strip().lower()
        qubits = [int(value) for value in re.findall(r"q\[(\d+)\]", line)]
        if len(set(qubits)) != 2 or line.startswith(("measure", "barrier")):
            continue
        layer = max((two_qubit_depths.get(qubit, 0) for qubit in qubits), default=0) + 1
        for qubit in qubits:
            two_qubit_depths[qubit] = layer
        two_qubit_layers = max(two_qubit_layers, layer)
    detected_swaps = len(re.findall(r"(?mi)^\s*swap\s+", qasm))
    swaps = detected_swaps if swap_count is None else int(swap_count)
    reasons: list[str] = []
    if uncalibrated_couplings:
        reasons.append(f"uncalibrated_couplings={uncalibrated_couplings}")
    if swaps:
        reasons.append(f"swap_count={swaps}")
    if not mapping_verified:
        reasons.append("physical_mapping_not_verified")
    if max_cnot is not None and metrics["cnot_count"] > int(max_cnot):
        reasons.append(f"cnot_count={metrics['cnot_count']} exceeds {int(max_cnot)}")
    if max_depth is not None and metrics["depth"] > int(max_depth):
        reasons.append(f"depth={metrics['depth']} exceeds {int(max_depth)}")
    return CompilationAudit(
        cnot_count=metrics["cnot_count"],
        swap_count=swaps,
        depth=metrics["depth"],
        two_qubit_gate_layers=two_qubit_layers,
        mapping_verified=bool(mapping_verified),
        uncalibrated_couplings=int(uncalibrated_couplings),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def transpile_for_baihua(
    qasm: str,
    backend_name: str,
    target_qubits: Sequence[int],
) -> tuple[str, CompilationAudit]:
    """Compile locally with QuarkStudio and audit the physical circuit."""
    try:
        from quark.circuit import Backend, Transpiler
    except ImportError as exc:  # pragma: no cover - optional hardware runtime
        raise RuntimeError("QuarkStudio is required for Baihua physical compilation.") from exc
    backend = Backend(str(backend_name or "Baihua"))
    compiled = Transpiler(backend).run(
        qasm,
        target_qubits=[int(value) for value in target_qubits],
        optimize_level=1,
        niter=5,
    )
    physical_qasm = str(compiled.to_openqasm2)
    measured = sorted({int(value) for value in re.findall(r"measure\s+q\[(\d+)\]", physical_qasm)})
    mapping_verified = not measured or measured == sorted(int(value) for value in target_qubits)
    audit = compilation_audit(
        physical_qasm,
        mapping_verified=mapping_verified,
        uncalibrated_couplings=0,
    )
    return physical_qasm, audit


def run_simulator_arm(
    proxy: SparseProxyQUBO,
    depth: int,
    shots: int,
    seed: int,
    parameters: QAOAParameters | None = None,
) -> QAOARunResult:
    parameters = parameters or pretrain_parameters(proxy, depth)
    qasm = build_qaoa_qasm(proxy, depth, parameters.gamma, parameters.beta)
    counts = simulate_counts(proxy, parameters, shots=shots, seed=seed)
    audit = compilation_audit(qasm, swap_count=0, mapping_verified=True)
    return QAOARunResult(
        arm=f"sim_p{depth}",
        counts=counts,
        qasm=qasm,
        physical_qasm="",
        parameters=parameters,
        compilation=audit,
        shots=max(1, int(shots)),
        seed=int(seed),
        message="Noiseless statevector sampling with pre-trained parameters.",
    )


def run_baihua_arm(
    *,
    proxy: SparseProxyQUBO,
    parameters: QAOAParameters,
    physical_qasm: str,
    compilation: CompilationAudit,
    shots: int,
    seed: int,
    backend: str = "Baihua",
    api_token: str = "",
    submit_hardware: bool = False,
    confirm_hardware_submit: bool = False,
    wait: bool = True,
    poll_timeout_sec: int = 600,
    poll_interval_sec: float = 3.0,
) -> QAOARunResult:
    """Return a guarded dry-run or submit one precompiled Baihua circuit."""
    logical_qasm = build_qaoa_qasm(
        proxy,
        parameters.depth,
        parameters.gamma,
        parameters.beta,
    )
    if not compilation.passed:
        return QAOARunResult(
            arm=f"baihua_p{parameters.depth}",
            counts={},
            qasm=logical_qasm,
            physical_qasm=physical_qasm,
            parameters=parameters,
            compilation=compilation,
            shots=max(1, int(shots)),
            seed=int(seed),
            backend=backend,
            message="Hardware submission blocked by compilation audit: " + "; ".join(compilation.reasons),
        )
    if not submit_hardware:
        return QAOARunResult(
            arm=f"baihua_p{parameters.depth}",
            counts={},
            qasm=logical_qasm,
            physical_qasm=physical_qasm,
            parameters=parameters,
            compilation=compilation,
            shots=max(1, int(shots)),
            seed=int(seed),
            backend=backend,
            message="Dry-run only; no hardware task was submitted.",
        )
    if not confirm_hardware_submit:
        raise PermissionError(
            "Hardware submission requires both submit_hardware=True and explicit confirmation."
        )
    if not str(api_token or "").strip():
        raise ValueError("QUAFU_API_TOKEN is empty; Baihua submission was not attempted.")
    try:
        from quark import Task
    except ImportError as exc:  # pragma: no cover - optional hardware runtime
        raise RuntimeError("QuarkStudio is required for Baihua submission.") from exc

    manager = Task(str(api_token).strip())
    task_payload = {
        "chip": str(backend or "Baihua"),
        "name": f"QRF_Baihua_DeepBlock_p{parameters.depth}",
        "circuit": physical_qasm,
        "shots": max(1, int(shots)),
        "compile": False,
        "options": {
            "compiler": None,
            "correct": False,
            "open_dd": None,
            "target_qubits": [],
        },
    }
    task_id = str(manager.run(task_payload))
    counts: dict[str, int] = {}
    message = "Submitted precompiled circuit to Baihua; result polling skipped."
    if wait:
        deadline = time.monotonic() + max(1, int(poll_timeout_sec))
        status = "unknown"
        error_text = ""
        while True:
            raw_result = manager.result(task_id)
            counts = {}
            if isinstance(raw_result, Mapping):
                status = str(raw_result.get("status") or "unknown")
                error_text = str(raw_result.get("error") or "")
                raw_counts = raw_result.get("corrected") or raw_result.get("count") or {}
                if isinstance(raw_counts, Mapping):
                    for key, value in raw_counts.items():
                        try:
                            count = int(value)
                        except (TypeError, ValueError):
                            continue
                        if count > 0:
                            counts[str(key).replace(" ", "")] = count
            finished = status.strip().lower() in {
                "finished",
                "completed",
                "done",
                "success",
                "failed",
                "error",
                "cancelled",
            }
            if counts or finished or time.monotonic() >= deadline:
                break
            time.sleep(max(0.2, float(poll_interval_sec)))
        if counts:
            message = (
                f"Baihua task status={status}; collected {len(counts)} outcomes over "
                f"{sum(counts.values())} returned shots."
            )
        elif time.monotonic() >= deadline:
            message = f"Baihua task polling timed out with status={status}; counts are not available yet."
        else:
            message = f"Baihua task ended with status={status} but no counts; error={error_text or 'none'}."
    return QAOARunResult(
        arm=f"baihua_p{parameters.depth}",
        counts=counts,
        qasm=logical_qasm,
        physical_qasm=physical_qasm,
        parameters=parameters,
        compilation=compilation,
        shots=max(1, int(shots)),
        seed=int(seed),
        task_id=task_id,
        backend=str(backend or "Baihua"),
        message=message,
    )

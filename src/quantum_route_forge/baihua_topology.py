from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class QubitCalibration:
    qubit: int
    available: bool = True
    readout_fidelity: float | None = None
    t1: float | None = None
    t2: float | None = None


@dataclass(frozen=True)
class CouplerCalibration:
    left: int
    right: int
    fidelity: float
    directed: bool = False


@dataclass(frozen=True)
class PhysicalSubgraph:
    qubits: tuple[int, ...]
    couplers: tuple[CouplerCalibration, ...]
    calibration_time: str
    score: float
    source: str

    @property
    def width(self) -> int:
        return len(self.qubits)

    @property
    def minimum_coupling_fidelity(self) -> float:
        return min((edge.fidelity for edge in self.couplers), default=0.0)

    @property
    def average_coupling_fidelity(self) -> float:
        if not self.couplers:
            return 0.0
        return sum(edge.fidelity for edge in self.couplers) / len(self.couplers)

    @property
    def uncalibrated_couplings(self) -> int:
        return sum(edge.fidelity <= 0.0 for edge in self.couplers)

    @property
    def logical_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((index, index + 1) for index in range(max(0, self.width - 1)))

    def payload(self) -> dict[str, object]:
        return {
            "qubits": list(self.qubits),
            "couplers": [asdict(edge) for edge in self.couplers],
            "calibration_time": self.calibration_time,
            "score": self.score,
            "source": self.source,
            "width": self.width,
            "minimum_coupling_fidelity": self.minimum_coupling_fidelity,
            "average_coupling_fidelity": self.average_coupling_fidelity,
            "uncalibrated_couplings": self.uncalibrated_couplings,
            "logical_edges": [list(edge) for edge in self.logical_edges],
        }


def _as_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(row: Mapping[str, object], names: Sequence[str]) -> float | None:
    for name in names:
        value = _as_float(row.get(name))
        if value is not None:
            return value
    return None


def parse_qubit_calibrations(chip_info: Mapping[str, object]) -> dict[int, QubitCalibration]:
    raw = chip_info.get("qubits_info") or chip_info.get("qubit_info") or {}
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[int, QubitCalibration] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        raw_index = value.get("qubit_index", value.get("index", key))
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        status = str(value.get("status", value.get("state", ""))).strip().lower()
        available = status not in {"disabled", "offline", "bad", "unavailable", "0", "false"}
        parsed[index] = QubitCalibration(
            qubit=index,
            available=available,
            readout_fidelity=_first_float(
                value,
                ("readout_fidelity", "readoutFidelity", "fidelity", "measure_fidelity"),
            ),
            t1=_first_float(value, ("T1", "t1", "t_1")),
            t2=_first_float(value, ("T2", "t2", "t_2")),
        )
    return parsed


def parse_couplers(chip_info: Mapping[str, object]) -> list[CouplerCalibration]:
    raw = chip_info.get("couplers_info") or chip_info.get("coupler_info") or {}
    rows = raw.values() if isinstance(raw, Mapping) else raw if isinstance(raw, Sequence) else ()
    parsed: list[CouplerCalibration] = []
    for value in rows:
        if not isinstance(value, Mapping):
            continue
        pair = value.get("qubits_index") or value.get("qubits") or value.get("pair") or ()
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            continue
        try:
            left, right = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        fidelity = _first_float(
            value,
            ("fidelity", "cz_fidelity", "cnot_fidelity", "two_qubit_fidelity"),
        )
        parsed.append(
            CouplerCalibration(
                left=left,
                right=right,
                fidelity=float(fidelity or 0.0),
                directed=bool(value.get("directed", False)),
            )
        )
    return parsed


def calibration_snapshot(chip_info: Mapping[str, object]) -> dict[str, object]:
    """Return a normalized, non-invented view of the current backend data."""
    return {
        "backend": chip_info.get("name") or chip_info.get("chip_name") or "Baihua",
        "calibration_time": chip_info.get("calibration_time") or chip_info.get("calibrationTime"),
        "qubits": [asdict(row) for row in parse_qubit_calibrations(chip_info).values()],
        "couplers": [asdict(row) for row in parse_couplers(chip_info)],
    }


def _path_score(
    path: tuple[int, ...],
    lookup: Mapping[frozenset[int], CouplerCalibration],
    qubits: Mapping[int, QubitCalibration],
) -> float:
    edges = [lookup[frozenset((left, right))] for left, right in zip(path, path[1:])]
    min_fidelity = min((edge.fidelity for edge in edges), default=0.0)
    avg_fidelity = sum((edge.fidelity for edge in edges), 0.0) / max(1, len(edges))
    qrows = [qubits.get(index) for index in path]
    readouts = [row.readout_fidelity for row in qrows if row and row.readout_fidelity is not None]
    t1s = [row.t1 for row in qrows if row and row.t1 is not None]
    t2s = [row.t2 for row in qrows if row and row.t2 is not None]
    readout = sum(readouts) / len(readouts) if readouts else 0.0
    # T1/T2 are used as bounded tie breakers because vendors expose different units.
    coherence = 0.0
    if t1s or t2s:
        coherence = math.tanh((sum(t1s + t2s) / len(t1s + t2s)) / 100.0)
    return 0.50 * min_fidelity + 0.30 * avg_fidelity + 0.15 * readout + 0.05 * coherence


def select_baihua_subgraph(
    chip_info: Mapping[str, object],
    width: int = 8,
    manual_qubits: Sequence[int] | None = None,
    beam_width: int = 50_000,
) -> PhysicalSubgraph:
    """Choose a calibrated physical path, suitable for a zero-SWAP sparse QUBO."""
    width = int(width)
    if width < 1 or width > 8:
        raise ValueError("Baihua DeepBlock width must be between 1 and 8.")
    couplers = parse_couplers(chip_info)
    qubit_rows = parse_qubit_calibrations(chip_info)
    lookup: dict[frozenset[int], CouplerCalibration] = {}
    adjacency: dict[int, list[int]] = {}
    for edge in couplers:
        if edge.fidelity <= 0.0:
            continue
        if not qubit_rows.get(edge.left, QubitCalibration(edge.left)).available:
            continue
        if not qubit_rows.get(edge.right, QubitCalibration(edge.right)).available:
            continue
        key = frozenset((edge.left, edge.right))
        current = lookup.get(key)
        if current is None or edge.fidelity > current.fidelity:
            lookup[key] = edge
        adjacency.setdefault(edge.left, []).append(edge.right)
        adjacency.setdefault(edge.right, []).append(edge.left)
    for node in adjacency:
        adjacency[node] = sorted(set(adjacency[node]))

    calibration_time = str(
        chip_info.get("calibration_time") or chip_info.get("calibrationTime") or ""
    )
    if manual_qubits is not None:
        path = tuple(int(value) for value in manual_qubits)
        if len(path) != width or len(set(path)) != width:
            raise ValueError("Manual physical qubits must be unique and match the block width.")
        missing = [
            (left, right)
            for left, right in zip(path, path[1:])
            if frozenset((left, right)) not in lookup
        ]
        if missing:
            raise ValueError(f"Manual physical path contains uncalibrated couplings: {missing}")
        source = "manual"
    elif width == 1:
        available = sorted(index for index, row in qubit_rows.items() if row.available)
        if not available:
            available = sorted(adjacency)
        if not available:
            raise RuntimeError("Calibration snapshot contains no available Baihua qubit.")
        path = (max(available, key=lambda index: qubit_rows.get(index, QubitCalibration(index)).readout_fidelity or 0.0),)
        source = "automatic"
    else:
        states = [(node,) for node in sorted(adjacency)]
        for _ in range(1, width):
            candidates: list[tuple[int, ...]] = []
            for path_state in states:
                for neighbor in adjacency.get(path_state[-1], ()):  # simple calibrated paths
                    if neighbor not in path_state:
                        candidates.append(path_state + (neighbor,))
            if not candidates:
                raise RuntimeError(
                    f"No fully calibrated connected Baihua path of width {width} was found."
                )
            candidates.sort(
                key=lambda candidate: (_path_score(candidate, lookup, qubit_rows), candidate),
                reverse=True,
            )
            states = candidates[: max(1, int(beam_width))]
        path = max(states, key=lambda candidate: (_path_score(candidate, lookup, qubit_rows), candidate))
        source = "automatic"

    selected_edges = tuple(
        lookup[frozenset((left, right))] for left, right in zip(path, path[1:])
    )
    return PhysicalSubgraph(
        qubits=path,
        couplers=selected_edges,
        calibration_time=calibration_time,
        score=_path_score(path, lookup, qubit_rows),
        source=source,
    )


def offline_identity_subgraph(width: int) -> PhysicalSubgraph:
    """Explicitly marked simulator-only topology; never presented as calibration."""
    width = int(width)
    if width < 1 or width > 8:
        raise ValueError("width must be between 1 and 8")
    return PhysicalSubgraph(
        qubits=tuple(range(width)),
        couplers=tuple(
            CouplerCalibration(left=index, right=index + 1, fidelity=1.0)
            for index in range(max(0, width - 1))
        ),
        calibration_time="",
        score=1.0,
        source="simulator_identity_not_hardware_calibration",
    )

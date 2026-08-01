from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance
from quantum_route_forge.maxcut_qaoa import (
    build_qaoa_openqasm,
    build_sparse_maxcut,
)
from quantum_route_forge.quafu_bridge import submit_quafu_qasm_job


def test_quark_submission_uses_qsteed_and_normalizes_shots(monkeypatch):
    class FakeTask:
        URL = "https://example.invalid"
        last_payload = None

        def __init__(self, token):
            assert token == "test-token"

        def run(self, payload):
            FakeTask.last_payload = payload
            return 123456

        def result(self, task_id, timeout):
            assert task_id == "123456"
            assert timeout == 30
            return {
                "status": "Finished",
                "chip": "Dongling",
                "count": {"000111": 700, "111000": 324},
            }

    monkeypatch.setitem(
        sys.modules,
        "quark",
        types.SimpleNamespace(Task=FakeTask),
    )
    instance = generate_dispatch_instance(
        seed=5,
        num_customers=6,
        num_vehicles=2,
        vehicle_capacity=20,
    )
    problem = build_sparse_maxcut(instance.customers, max_edges=10)
    circuit = build_qaoa_openqasm(problem)
    result = submit_quafu_qasm_job(
        qasm=circuit.qasm,
        selected_customers=problem.customers,
        api_token="test-token",
        backend="Dongling",
        shots=100,
        wait=True,
        timeout_sec=30,
    )

    assert result.ok
    assert result.task_id == "123456"
    assert result.bitstring == "000111"
    assert result.counts == {"000111": 700, "111000": 324}
    assert FakeTask.last_payload["shots"] == 1024
    assert FakeTask.last_payload["options"]["compiler"] == "qsteed"
    assert FakeTask.last_payload["circuit"] == circuit.qasm.strip()

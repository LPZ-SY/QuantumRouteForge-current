from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization
from quantum_route_forge.quafu_bridge import QuafuJobResult


def _fake_quafu_job(instance):
    selected_ids = [c.customer_id for c in instance.customers[:6]]
    return QuafuJobResult(
        ok=True,
        message="Submitted to Quafu SQC composer API successfully.",
        task_id="fake-task-id",
        backend="Baihua",
        bitstring=None,
        selected_customer_ids=selected_ids,
        endpoint="https://quafu-sqc.baqis.ac.cn/",
    )


def test_manual_bitstring_forces_hybrid_mode():
    instance = generate_dispatch_instance(
        seed=2026,
        num_customers=18,
        num_vehicles=4,
        vehicle_capacity=28,
    )
    fake_job = _fake_quafu_job(instance)

    with patch("quantum_route_forge.pipeline.submit_quafu_qasm_job", return_value=fake_job):
        result = run_optimization(
            instance=instance,
            mode="quantum",
            quafu_token="dummy",
            quafu_wait=False,
            quafu_manual_bitstring="010101",
            tabu_iterations=1,
            routing_method="heuristic",
        )

    assert result.metadata.used_mode == "tabu_qaoa_manual_hybrid"
    assert result.metadata.quantum_bitstring == "010101"
    assert "Manual bitstring override applied." in result.metadata.message


def test_invalid_manual_bitstring_is_ignored():
    instance = generate_dispatch_instance(
        seed=2027,
        num_customers=18,
        num_vehicles=4,
        vehicle_capacity=28,
    )
    fake_job = _fake_quafu_job(instance)

    with patch("quantum_route_forge.pipeline.submit_quafu_qasm_job", return_value=fake_job):
        result = run_optimization(
            instance=instance,
            mode="quantum",
            quafu_token="dummy",
            quafu_wait=False,
            quafu_manual_bitstring="01AB10",
            tabu_iterations=1,
            routing_method="heuristic",
        )

    assert result.metadata.used_mode == "tabu_qaoa_quafu_submitted_local_fallback"
    assert "Manual bitstring ignored" in result.metadata.message

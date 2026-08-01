from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge.quafu_bridge import _extract_bitstring_from_counts, _extract_bitstring_from_payload
from quantum_route_forge import quafu_bridge


def test_extract_bitstring_from_counts_prefers_highest_weight():
    counts = {
        "0101": 18,
        "1111": 6,
        "0000": 10,
    }
    assert _extract_bitstring_from_counts(counts) == "0101"


def test_extract_bitstring_from_nested_serialized_payload():
    payload = {
        "status": "Completed",
        "result": {
            "res": '{"0011": 9, "1010": 3}',
        },
    }
    assert _extract_bitstring_from_payload(payload) == "0011"


def test_extract_bitstring_from_payload_returns_none_when_missing_counts():
    payload = {
        "status": "Submitted",
        "taskId": 123,
        "message": "still running",
    }
    assert _extract_bitstring_from_payload(payload) is None


def test_wait_polling_uses_task_detail_even_when_task_list_missing():
    original_item = quafu_bridge._query_sqc_task_item
    original_detail = quafu_bridge._query_sqc_task_detail
    original_sleep = quafu_bridge.time.sleep
    try:
        quafu_bridge._query_sqc_task_item = lambda **kwargs: None
        quafu_bridge._query_sqc_task_detail = lambda **kwargs: {"count": {"0101": 7, "1111": 2}}
        quafu_bridge.time.sleep = lambda *_args, **_kwargs: None

        bitstring, msg = quafu_bridge._wait_for_sqc_task_bitstring(
            task_id="fake",
            endpoint="https://quafu-sqc.baqis.ac.cn/",
            access_token="dummy",
            timeout_sec=5,
            proxy_url="",
            verify_ssl=True,
        )
    finally:
        quafu_bridge._query_sqc_task_item = original_item
        quafu_bridge._query_sqc_task_detail = original_detail
        quafu_bridge.time.sleep = original_sleep

    assert bitstring == "0101"
    assert "bitstring retrieved" in msg

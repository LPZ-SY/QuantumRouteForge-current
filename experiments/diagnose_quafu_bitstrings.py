from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge import generate_dispatch_instance, run_optimization  # noqa: E402
from quantum_route_forge.quafu_bridge import (  # noqa: E402
    fetch_quafu_task_bitstring,
    submit_quafu_partition_job,
)


try:
    from quantum_route_forge.quafu_bridge import (  # type: ignore[attr-defined] # noqa: E402
        _build_partition_qasm,
        _discover_sqc_chip,
        _extract_bitstring_from_payload,
        _http_json_request,
        _looks_like_jwt_access_token,
        _pick_customers_for_quantum,
    )
except Exception:
    _build_partition_qasm = None
    _discover_sqc_chip = None
    _extract_bitstring_from_payload = None
    _http_json_request = None
    _looks_like_jwt_access_token = None
    _pick_customers_for_quantum = None


ALLOWED_CLASSES = {
    "real_quantum_result_used",
    "quafu_bitstring_assignment_refine",
    "quafu_submitted_classical_refine",
    "classical_fallback",
    "token_missing",
    "failed",
    "unknown",
}


def _parse_bool(value: str) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_endpoint(base_url: str) -> str:
    text = (base_url or "").strip()
    if not text:
        return "https://quafu-sqc.baqis.ac.cn/"
    if not text.startswith(("http://", "https://")):
        text = f"https://{text}"
    if not text.endswith("/"):
        text = f"{text}/"
    return text


def _parse_task_id(data: Any) -> Optional[str]:
    if isinstance(data, (str, int)):
        val = str(data).strip()
        return val or None
    if isinstance(data, dict):
        for key in ("taskId", "task_id", "id"):
            if key in data and str(data.get(key)).strip():
                return str(data.get(key)).strip()
    return None


def _has_counts(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key in ("counts", "count", "probabilities", "logicalq_res", "res", "result", "data"):
            val = payload.get(key)
            if isinstance(val, dict):
                if any(re.fullmatch(r"[01]+", str(k).replace(" ", "")) for k in val.keys()):
                    return True
            if isinstance(val, (dict, list)) and _has_counts(val):
                return True
        for v in payload.values():
            if isinstance(v, (dict, list)) and _has_counts(v):
                return True
    if isinstance(payload, list):
        return any(_has_counts(item) for item in payload)
    return False


def _guess_counts_format(payload: Any) -> str:
    if _has_counts(payload):
        return "dict[str,number]"
    return "none"


def _extract_bitstring(payload: Any) -> Optional[str]:
    if _extract_bitstring_from_payload is not None:
        try:
            return _extract_bitstring_from_payload(payload)
        except Exception:
            pass

    if isinstance(payload, dict):
        for v in payload.values():
            bit = _extract_bitstring(v)
            if bit:
                return bit
    if isinstance(payload, list):
        for item in payload:
            bit = _extract_bitstring(item)
            if bit:
                return bit
    return None


def _inspect_codebase() -> dict[str, Any]:
    pipeline_text = _read_text(ROOT / "src" / "quantum_route_forge" / "pipeline.py")
    bridge_text = _read_text(ROOT / "src" / "quantum_route_forge" / "quafu_bridge.py")
    app_text = _read_text(ROOT / "app.py")
    cli_text = _read_text(ROOT / "run_cli.py")
    readme_text = _read_text(ROOT / "README.md")

    return {
        "pipeline_has_submit_call": "submit_quafu_partition_job(" in pipeline_text,
        "pipeline_has_result_query": "fetch_quafu_task_bitstring(" in pipeline_text,
        "pipeline_uses_bitstring_hints": "bitstring_to_vehicle_hints(" in pipeline_text,
        "pipeline_used_mode_quafu_submitted": "quafu_submitted_classical_refine" in pipeline_text,
        "pipeline_used_mode_quafu_hybrid": "quafu_quantum_hybrid" in pipeline_text,
        "bridge_can_wait_and_poll": "_wait_for_sqc_task_bitstring(" in bridge_text,
        "bridge_extracts_bitstring_from_counts": "_extract_bitstring_from_counts" in bridge_text,
        "app_shows_task_backend_endpoint": all(
            key in app_text for key in ["quantum_task_id", "quantum_backend", "quantum_endpoint"]
        ),
        "cli_outputs_quantum_fields": all(
            key in cli_text for key in ["quantum_task_id", "quantum_backend", "quantum_bitstring", "quantum_endpoint"]
        ),
        "readme_mentions_quafu_submission": "task_id" in readme_text and "Quafu" in readme_text,
    }


def _raw_submit_sqc(
    token: str,
    endpoint: str,
    customers: list,
    backend: str,
    shots: int,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
    max_qubits: int,
) -> tuple[dict[str, Any], Optional[str], Optional[str], list[int]]:
    if any(x is None for x in [_http_json_request, _build_partition_qasm, _discover_sqc_chip, _pick_customers_for_quantum]):
        return {
            "ok": False,
            "error": "raw_submit_unavailable",
            "message": "内部原始 HTTP 诊断函数不可用，已跳过原始提交响应抓取。",
        }, None, None, []

    selected = _pick_customers_for_quantum(customers, max_qubits=max_qubits)
    selected_ids = [c.customer_id for c in selected]
    chip = _discover_sqc_chip(
        endpoint=endpoint,
        access_token=token,
        preferred_chip=backend,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )
    qasm = _build_partition_qasm(selected)
    payload = {
        "chip": chip,
        "shots": max(100, int(shots)),
        "name": "QRF_diagnose_bitstring",
        "advanced": False,
        "compiler": "quarkcircuit",
        "targetQubits": [],
        "readoutErrorMiti": False,
        "dd": "none",
        "qasm": qasm,
    }
    headers = {
        "Cookie": f"access_token={token.strip()}",
        "Content-Type": "application/json",
    }
    status, reason, data, text = _http_json_request(
        endpoint=endpoint,
        method="POST",
        path="/api/task",
        headers=headers,
        payload=payload,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )

    submit_raw = {
        "ok": int(status) == 200,
        "endpoint": endpoint,
        "http_status": status,
        "reason": reason,
        "response_json": data,
        "response_text": text,
        "request_meta": {
            "chip": chip,
            "shots": max(100, int(shots)),
            "selected_customer_ids": selected_ids,
            "qasm_length": len(qasm),
        },
    }
    return submit_raw, _parse_task_id(data), chip, selected_ids


def _raw_query_sqc_task(
    token: str,
    endpoint: str,
    task_id: str,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
) -> dict[str, Any]:
    if _http_json_request is None:
        return {
            "ok": False,
            "error": "raw_query_unavailable",
            "message": "内部原始 HTTP 诊断函数不可用，已跳过原始结果抓取。",
        }

    headers = {"Cookie": f"access_token={token.strip()}"}

    detail_status, detail_reason, detail_data, detail_text = _http_json_request(
        endpoint=endpoint,
        method="GET",
        path=f"/api/task/{task_id}",
        headers=headers,
        payload=None,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )

    list_status, list_reason, list_data, list_text = _http_json_request(
        endpoint=endpoint,
        method="GET",
        path=(
            "/api/tasks?page=1&pageSize=1"
            '&sort={"submitTime":"descending"}'
            f'&filter={{"taskId":"{task_id}"}}'
        ),
        headers=headers,
        payload=None,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )

    payload = {
        "ok": detail_status == 200 or list_status == 200,
        "endpoint": endpoint,
        "task_id": task_id,
        "detail": {
            "http_status": detail_status,
            "reason": detail_reason,
            "response_json": detail_data,
            "response_text": detail_text,
        },
        "list": {
            "http_status": list_status,
            "reason": list_reason,
            "response_json": list_data,
            "response_text": list_text,
        },
    }
    return payload


def _classify(
    has_token: bool,
    submitted_successfully: bool,
    used_mode: str,
    has_bitstrings: bool,
    bitstring_used_in_assignment: bool,
    error_message: str,
) -> str:
    if not has_token:
        return "token_missing"
    if error_message:
        if "token" in error_message.lower():
            return "token_missing"
        if "fallback" in used_mode:
            return "classical_fallback"
    if used_mode == "quafu_submitted_classical_refine":
        return "quafu_submitted_classical_refine"
    if used_mode == "quafu_unavailable_classical_fallback":
        return "classical_fallback"
    if used_mode == "quafu_quantum_hybrid" and has_bitstrings and bitstring_used_in_assignment:
        return "quafu_bitstring_assignment_refine"
    if submitted_successfully and not has_bitstrings:
        return "quafu_submitted_classical_refine"
    if submitted_successfully and has_bitstrings and bitstring_used_in_assignment:
        return "quafu_bitstring_assignment_refine"
    if error_message:
        return "failed"
    return "unknown"


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    token = (args.quafu_token or "").strip() or os.getenv("QUAFU_API_TOKEN", "").strip()
    base_url = (args.quafu_base_url or "").strip() or os.getenv("QUAFU_BASE_URL", "").strip()
    backend = (args.quafu_backend or "").strip() or "Baihua"
    endpoint = _normalize_endpoint(base_url)
    proxy_url = (args.quafu_proxy_url or "").strip() or os.getenv("QUAFU_PROXY_URL", "").strip()

    has_token = bool(token)
    code_info = _inspect_codebase()

    summary: dict[str, Any] = {
        "timestamp": _now_iso(),
        "has_quafu_token": has_token,
        "submitted_successfully": False,
        "task_id": None,
        "backend": None,
        "endpoint": endpoint,
        "can_query_task_result": False,
        "has_measurement_counts": False,
        "has_bitstrings": False,
        "bitstring_format": "none",
        "counts_format": "none",
        "bitstring_used_in_assignment": False,
        "bitstring_used_in_route_refinement": False,
        "current_quantum_mode_classification": "unknown",
        "error_message": "",
        "requested_setup": {
            "customers": int(args.customers),
            "vehicles": int(args.vehicles),
            "seed": int(args.seed),
            "shots": int(args.shots),
            "backend": backend,
            "wait": bool(args.wait),
            "timeout_sec": int(args.timeout_sec),
            "verify_ssl": bool(args.verify_ssl),
            "proxy_url": proxy_url,
        },
        "code_inspection": code_info,
        "pipeline_run": {},
    }

    submit_raw_path = outdir / "raw_quafu_submit_response.json"
    result_raw_path = outdir / "raw_quafu_result_response.json"

    if not has_token:
        submit_raw = {
            "ok": False,
            "skipped": True,
            "reason": "QUAFU_API_TOKEN 未设置，跳过 Quafu 真机提交实测。",
        }
        _write_json(submit_raw_path, submit_raw)
        summary["current_quantum_mode_classification"] = "token_missing"
        return summary

    try:
        # 1) 构造小规模场景并做一次 pipeline 量子模式运行。
        capacity = max(8, int(args.capacity))
        instance = generate_dispatch_instance(
            seed=int(args.seed),
            num_customers=int(args.customers),
            num_vehicles=int(args.vehicles),
            vehicle_capacity=capacity,
        )

        run_result = run_optimization(
            instance=instance,
            mode="quantum",
            time_limit=10,
            num_reads=80,
            quafu_token=token,
            quafu_backend=backend,
            quafu_base_url=endpoint,
            quafu_shots=int(args.shots),
            quafu_wait=bool(args.wait),
            quafu_max_qubits=int(args.max_qubits),
            quafu_timeout_sec=int(args.timeout_sec),
            quafu_proxy_url=proxy_url,
            quafu_verify_ssl=bool(args.verify_ssl),
        )

        used_mode = run_result.metadata.used_mode
        quafu_task_id = run_result.metadata.quantum_task_id
        quafu_backend = run_result.metadata.quantum_backend
        quafu_endpoint = run_result.metadata.quantum_endpoint or endpoint
        quafu_bitstring = run_result.metadata.quantum_bitstring

        summary["pipeline_run"] = {
            "used_mode": used_mode,
            "message": run_result.metadata.message,
            "quantum_task_id": quafu_task_id,
            "quantum_backend": quafu_backend,
            "quantum_endpoint": quafu_endpoint,
            "quantum_bitstring": quafu_bitstring,
            "bqm_energy": run_result.metadata.energy,
        }

        summary["task_id"] = quafu_task_id
        summary["backend"] = quafu_backend or backend
        summary["endpoint"] = quafu_endpoint
        summary["has_bitstrings"] = bool(quafu_bitstring)
        if quafu_bitstring:
            summary["bitstring_format"] = f"binary_string_len_{len(quafu_bitstring)}"

        summary["bitstring_used_in_assignment"] = (
            used_mode == "quafu_quantum_hybrid" and bool(quafu_bitstring)
        )
        summary["bitstring_used_in_route_refinement"] = summary["bitstring_used_in_assignment"]
        summary["submitted_successfully"] = bool(quafu_task_id) or (
            "Submitted to Quafu" in run_result.metadata.message
        )

        # 2) 记录原始提交响应（仅在 JWT + SQC 路径可直接抓取原始 HTTP）。
        token_is_jwt = bool(_looks_like_jwt_access_token(token)) if _looks_like_jwt_access_token else False
        if token_is_jwt:
            submit_raw, raw_task_id, raw_chip, _selected_ids = _raw_submit_sqc(
                token=token,
                endpoint=endpoint,
                customers=instance.customers,
                backend=backend,
                shots=int(args.shots),
                timeout_sec=int(args.timeout_sec),
                proxy_url=proxy_url,
                verify_ssl=bool(args.verify_ssl),
                max_qubits=int(args.max_qubits),
            )
            _write_json(submit_raw_path, submit_raw)
            if not summary["task_id"] and raw_task_id:
                summary["task_id"] = raw_task_id
            if not summary["backend"] and raw_chip:
                summary["backend"] = raw_chip

            # 3) 有 task_id 时抓取原始结果响应。
            task_to_query = summary["task_id"]
            if task_to_query:
                raw_result = _raw_query_sqc_task(
                    token=token,
                    endpoint=endpoint,
                    task_id=str(task_to_query),
                    timeout_sec=int(args.timeout_sec),
                    proxy_url=proxy_url,
                    verify_ssl=bool(args.verify_ssl),
                )
                _write_json(result_raw_path, raw_result)
                summary["can_query_task_result"] = bool(raw_result.get("ok"))

                merged_payload = {
                    "detail": raw_result.get("detail", {}).get("response_json"),
                    "list": raw_result.get("list", {}).get("response_json"),
                }
                summary["has_measurement_counts"] = _has_counts(merged_payload)
                summary["counts_format"] = _guess_counts_format(merged_payload)
                bit = _extract_bitstring(merged_payload)
                if bit:
                    summary["has_bitstrings"] = True
                    summary["bitstring_format"] = f"binary_string_len_{len(bit)}"
        else:
            submit_job = submit_quafu_partition_job(
                customers=instance.customers,
                api_token=token,
                backend=backend,
                base_url=endpoint,
                shots=int(args.shots),
                wait=bool(args.wait),
                max_qubits=int(args.max_qubits),
                timeout_sec=int(args.timeout_sec),
                proxy_url=proxy_url,
                verify_ssl=bool(args.verify_ssl),
            )
            submit_raw = {
                "ok": submit_job.ok,
                "message": submit_job.message,
                "task_id": submit_job.task_id,
                "backend": submit_job.backend,
                "endpoint": submit_job.endpoint,
                "bitstring": submit_job.bitstring,
                "selected_customer_ids": submit_job.selected_customer_ids,
            }
            _write_json(submit_raw_path, submit_raw)

            if submit_job.task_id:
                result_job = fetch_quafu_task_bitstring(
                    task_id=submit_job.task_id,
                    api_token=token,
                    base_url=endpoint,
                    timeout_sec=int(args.timeout_sec),
                    proxy_url=proxy_url,
                    verify_ssl=bool(args.verify_ssl),
                )
                _write_json(
                    result_raw_path,
                    {
                        "ok": result_job.ok,
                        "message": result_job.message,
                        "task_id": result_job.task_id,
                        "backend": result_job.backend,
                        "endpoint": result_job.endpoint,
                        "bitstring": result_job.bitstring,
                    },
                )
                summary["can_query_task_result"] = True
                if result_job.bitstring:
                    summary["has_bitstrings"] = True
                    summary["bitstring_format"] = f"binary_string_len_{len(result_job.bitstring)}"

        summary["current_quantum_mode_classification"] = _classify(
            has_token=has_token,
            submitted_successfully=bool(summary["submitted_successfully"]),
            used_mode=summary.get("pipeline_run", {}).get("used_mode", ""),
            has_bitstrings=bool(summary["has_bitstrings"]),
            bitstring_used_in_assignment=bool(summary["bitstring_used_in_assignment"]),
            error_message=str(summary.get("error_message", "")),
        )

    except Exception as exc:
        summary["error_message"] = f"{type(exc).__name__}: {exc}"
        if summary.get("current_quantum_mode_classification") == "unknown":
            summary["current_quantum_mode_classification"] = _classify(
                has_token=has_token,
                submitted_successfully=bool(summary["submitted_successfully"]),
                used_mode=summary.get("pipeline_run", {}).get("used_mode", ""),
                has_bitstrings=bool(summary["has_bitstrings"]),
                bitstring_used_in_assignment=bool(summary["bitstring_used_in_assignment"]),
                error_message=summary["error_message"],
            )

    if summary["current_quantum_mode_classification"] not in ALLOWED_CLASSES:
        summary["current_quantum_mode_classification"] = "unknown"

    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    cls = summary.get("current_quantum_mode_classification", "unknown")
    has_token = summary.get("has_quafu_token", False)
    submitted = summary.get("submitted_successfully", False)
    has_bits = summary.get("has_bitstrings", False)
    task_id = summary.get("task_id") or "无"
    backend = summary.get("backend") or "无"
    endpoint = summary.get("endpoint") or "无"
    used_mode = summary.get("pipeline_run", {}).get("used_mode", "无")
    msg = summary.get("pipeline_run", {}).get("message", "")
    can_query = summary.get("can_query_task_result", False)

    lines = []
    lines.append("# Quafu measured bitstring 闭环诊断")
    lines.append("")
    lines.append("## 诊断结论（中文）")
    lines.append(f"- 当前分类：`{cls}`")
    lines.append(f"- 是否存在 QUAFU_API_TOKEN：`{has_token}`")
    lines.append(f"- 是否提交成功：`{submitted}`")
    lines.append(f"- task_id：`{task_id}`")
    lines.append(f"- backend：`{backend}`")
    lines.append(f"- endpoint：`{endpoint}`")
    lines.append(f"- pipeline used_mode：`{used_mode}`")
    lines.append(f"- 是否可查询任务结果：`{can_query}`")
    lines.append(f"- 是否检测到 bitstring：`{has_bits}`")
    lines.append(f"- bitstring 是否进入 assignment：`{summary.get('bitstring_used_in_assignment', False)}`")
    lines.append(
        f"- bitstring 是否进入 route refinement：`{summary.get('bitstring_used_in_route_refinement', False)}`"
    )

    if not has_token:
        lines.append("")
        lines.append("当前环境未设置 QUAFU_API_TOKEN，因此本次无法执行真实 Quafu 提交与结果查询，分类为 `token_missing`。")
    elif submitted and (not has_bits):
        lines.append("")
        lines.append("Quafu 任务提交成功，但当前流程尚未获得可用于优化闭环的 measured bitstring。")
    elif has_bits and (not summary.get("bitstring_used_in_assignment", False)):
        lines.append("")
        lines.append("当前可以获得 measured bitstring，但尚未用于 assignment refinement。")

    if msg:
        lines.append("")
        lines.append("## pipeline 消息")
        lines.append(msg)

    if summary.get("error_message"):
        lines.append("")
        lines.append("## 错误信息")
        lines.append(summary["error_message"])

    lines.append("")
    lines.append("## 代码级路径诊断")
    code = summary.get("code_inspection", {})
    for k in sorted(code.keys()):
        lines.append(f"- {k}: `{code.get(k)}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断 Quafu measured bitstring 是否进入优化闭环")
    parser.add_argument("--customers", type=int, default=6)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-qubits", type=int, default=6)
    parser.add_argument("--quafu-token", type=str, default="")
    parser.add_argument("--quafu-base-url", type=str, default="")
    parser.add_argument("--quafu-backend", type=str, default="Baihua")
    parser.add_argument("--quafu-proxy-url", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--wait", type=_parse_bool, default=True)
    parser.add_argument("--verify-ssl", type=_parse_bool, default=True)
    parser.add_argument("--outdir", type=str, default="results/quafu_diagnostics")
    args = parser.parse_args()

    summary = run_diagnostic(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(outdir / "diagnostic_summary.json", summary)
    (outdir / "quafu_bitstring_diagnostic.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

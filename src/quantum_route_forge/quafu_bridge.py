from __future__ import annotations

import ast
from dataclasses import dataclass
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from typing import Iterable, Optional
from urllib.parse import urlencode
from urllib.parse import urlparse

from .env import normalize_local_proxy_environment
from .geometry import euclidean
from .models import Customer


@dataclass(frozen=True)
class QuafuJobResult:
    ok: bool
    message: str
    task_id: Optional[str] = None
    backend: Optional[str] = None
    bitstring: Optional[str] = None
    selected_customer_ids: Optional[list[int]] = None
    endpoint: Optional[str] = None
    counts: Optional[dict[str, int]] = None


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url


def _candidate_base_urls(base_url: Optional[str]) -> list[str]:
    candidates: list[str] = []
    for candidate in [
        _normalize_base_url(base_url or ""),
        "https://quafu-sqc.baqis.ac.cn/",
        "https://quafu.baqis.ac.cn/",
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _dns_hint_for_base_url(base_url: str) -> str:
    host = urlparse(base_url).hostname or ""
    if not host:
        return "Endpoint URL has no valid hostname."

    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = sorted({item[4][0] for item in infos if item and item[4]})
    except Exception as exc:
        return f"DNS lookup failed for {host}: {type(exc).__name__}: {exc}"

    if not ips:
        return f"DNS lookup returned no IP for {host}."

    flagged: list[str] = []
    for ip in ips:
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if ip.startswith("198.18.") or ip_obj.is_reserved:
            flagged.append(ip)

    if flagged:
        return (
            f"{host} resolves to reserved/test IP {', '.join(flagged)}; "
            "this often means local proxy/TUN DNS interception."
        )
    return f"{host} resolved IPs: {', '.join(ips)}"


def _is_auth_error(error_message: str) -> bool:
    text = (error_message or "").lower()
    return (
        "invalid api_token" in text
        or ("api_token" in text and "not match" in text)
        or "invalid access token" in text
        or "token is invalid" in text
    )


def _is_sqc_endpoint(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").lower()
    return "quafu-sqc.baqis.ac.cn" in host


def _looks_like_jwt_access_token(token: str) -> bool:
    token = (token or "").strip()
    if not token:
        return False
    return re.match(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", token) is not None


def _http_json_request(
    endpoint: str,
    method: str,
    path: str,
    headers: dict,
    payload: Optional[dict],
    timeout_sec: int,
    verify_ssl: bool,
    proxy_url: str,
):
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid endpoint URL: {endpoint}")

    is_https = parsed.scheme.lower() == "https"
    target_host = parsed.hostname
    target_port = parsed.port or (443 if is_https else 80)
    base_path = parsed.path.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    request_path = f"{base_path}{path}" if base_path else path

    req_headers = dict(headers or {})
    body_bytes = b""
    if payload is not None:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req_headers.setdefault("Accept", "application/json, text/plain, */*")
    req_headers.setdefault("Content-Length", str(len(body_bytes)))

    context = None
    if is_https:
        context = ssl.create_default_context()
        if not verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

    proxy_url = (proxy_url or "").strip()
    if proxy_url.startswith("http://"):
        proxy_parsed = urlparse(proxy_url)
        proxy_host = proxy_parsed.hostname
        proxy_port = proxy_parsed.port or 80
        if not proxy_host:
            raise ValueError(f"Invalid proxy URL: {proxy_url}")
        if is_https:
            conn = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=timeout_sec, context=context)
            conn.set_tunnel(target_host, target_port)
        else:
            conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout_sec)
    else:
        if is_https:
            conn = http.client.HTTPSConnection(target_host, target_port, timeout=timeout_sec, context=context)
        else:
            conn = http.client.HTTPConnection(target_host, target_port, timeout=timeout_sec)

    try:
        conn.request(method.upper(), request_path, body=body_bytes, headers=req_headers)
        response = conn.getresponse()
        raw = response.read()
        text = raw.decode("utf-8", errors="replace")
        data = None
        if text.strip():
            try:
                data = json.loads(text)
            except Exception:
                data = text
        return response.status, response.reason, data, text
    finally:
        conn.close()


def _build_partition_qasm(selected: list[Customer]) -> str:
    qc = _build_partition_circuit(selected)
    qc.to_openqasm()
    return qc.openqasm


def _discover_sqc_chip(
    endpoint: str,
    access_token: str,
    preferred_chip: str,
    timeout_sec: int,
    verify_ssl: bool,
    proxy_url: str,
) -> str:
    headers = {
        "Cookie": f"access_token={access_token.strip()}",
    }
    status, _reason, data, _text = _http_json_request(
        endpoint=endpoint,
        method="GET",
        path="/api/availableChips",
        headers=headers,
        payload=None,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )
    if status != 200:
        if status == 401:
            raise RuntimeError(
                "availableChips failed: HTTP 401 (access_token invalid or expired for quafu-sqc)"
            )
        raise RuntimeError(f"availableChips failed: HTTP {status}")

    chips = []
    if isinstance(data, dict):
        available = data.get("available_chips")
        if isinstance(available, dict):
            chips = list(available.keys())
    if not chips:
        raise RuntimeError("No available chips returned by /api/availableChips")

    preferred_chip = (preferred_chip or "").strip()
    if preferred_chip:
        if preferred_chip in chips:
            return preferred_chip
        # Keep behavior resilient: fallback to a valid chip instead of submitting an invalid one.
        return chips[0]
    return chips[0]


def _submit_once_with_sqc(
    selected: list[Customer],
    access_token: str,
    backend: Optional[str],
    endpoint: str,
    shots: int,
    wait: bool,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
    qasm_override: str = "",
) -> QuafuJobResult:
    access_token = (access_token or "").strip()
    if not access_token:
        raise RuntimeError("SQC access token is empty.")
    requested_backend = (backend or "").strip()

    backend_name = _discover_sqc_chip(
        endpoint=endpoint,
        access_token=access_token,
        preferred_chip=requested_backend,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )

    qasm = (qasm_override or "").strip() or _build_partition_qasm(selected)
    payload = {
        "chip": backend_name,
        "shots": max(100, int(shots)),
        "name": "QRF_partition_seed",
        "advanced": False,
        "compiler": "quarkcircuit",
        "targetQubits": [],
        "readoutErrorMiti": False,
        "dd": "none",
        "qasm": qasm,
    }
    headers = {
        "Cookie": f"access_token={access_token}",
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
    if status != 200:
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("detail") or data.get("msg") or data)
        elif text:
            detail = text[:300]
        raise RuntimeError(f"SQC submit failed: HTTP {status} {reason} {detail}".strip())

    task_id = None
    if isinstance(data, (str, int)):
        task_id = str(data)
    elif isinstance(data, dict):
        task_id = str(data.get("taskId") or data.get("task_id") or data.get("id") or "")
    if not task_id:
        task_id = None

    bitstring = None
    backend_msg = ""
    if requested_backend and requested_backend != backend_name:
        backend_msg = (
            f" Requested backend '{requested_backend}' is unavailable; "
            f"fallback chip '{backend_name}' is used."
        )
    message = f"Submitted to Quafu SQC composer API successfully.{backend_msg}"
    if wait and task_id:
        bitstring, wait_msg = _wait_for_sqc_task_bitstring(
            task_id=task_id,
            endpoint=endpoint,
            access_token=access_token,
            timeout_sec=timeout_sec,
            proxy_url=proxy_url,
            verify_ssl=verify_ssl,
        )
        message = f"{message} {wait_msg}"
    elif wait and not task_id:
        message = f"{message} No task_id returned; skip result polling."

    return QuafuJobResult(
        ok=True,
        message=message,
        task_id=task_id,
        backend=backend_name,
        bitstring=bitstring,
        selected_customer_ids=[c.customer_id for c in selected],
        endpoint=endpoint,
    )


def _extract_bitstring_from_counts(counts: object) -> Optional[str]:
    if not isinstance(counts, dict) or not counts:
        return None

    cleaned: dict[str, float] = {}
    for key, value in counts.items():
        bitstring = str(key).replace(" ", "")
        if not re.fullmatch(r"[01]+", bitstring):
            continue
        try:
            cleaned[bitstring] = float(value)
        except (TypeError, ValueError):
            continue

    if not cleaned:
        return None
    return max(cleaned.items(), key=lambda kv: kv[1])[0]


SHOTS_GRANULARITY = 1024


def _normalize_shots(shots: int) -> int:
    try:
        value = int(shots)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return SHOTS_GRANULARITY
    remainder = value % SHOTS_GRANULARITY
    return (
        value
        if remainder == 0
        else value + SHOTS_GRANULARITY - remainder
    )


def _clean_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, int] = {}
    for key, value in raw.items():
        bits = str(key).replace(" ", "")
        if not re.fullmatch(r"[01]+", bits):
            continue
        try:
            cleaned[bits] = int(value)
        except (TypeError, ValueError):
            continue
    return cleaned


def _maybe_parse_serialized_object(value: object) -> object:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "{[":
        return value

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _extract_bitstring_from_payload(payload: object) -> Optional[str]:
    pending: list[object] = [payload]
    visited: set[int] = set()
    priority_fields = [
        "counts",
        "count",
        "res",
        "result",
        "probabilities",
        "logicalq_res",
        "data",
    ]

    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)

        parsed = _maybe_parse_serialized_object(current)
        if parsed is not current:
            pending.append(parsed)
            continue

        direct = _extract_bitstring_from_counts(current)
        if direct:
            return direct

        if isinstance(current, dict):
            for key in priority_fields:
                if key in current:
                    pending.append(current[key])
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)

    return None


def _query_sqc_task_item(
    task_id: str,
    endpoint: str,
    access_token: str,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
) -> Optional[dict]:
    headers = {
        "Cookie": f"access_token={access_token.strip()}",
    }
    params = urlencode(
        {
            "page": 1,
            "pageSize": 1,
            "sort": json.dumps({"submitTime": "descending"}),
            "filter": json.dumps({"taskId": str(task_id)}),
        }
    )
    status, _reason, data, _text = _http_json_request(
        endpoint=endpoint,
        method="GET",
        path=f"/api/tasks?{params}",
        headers=headers,
        payload=None,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )
    if status != 200 or not isinstance(data, dict):
        return None
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None
    item = items[0]
    if not isinstance(item, dict):
        return None
    return item


def _query_sqc_task_detail(
    task_id: str,
    endpoint: str,
    access_token: str,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
) -> Optional[dict]:
    headers = {
        "Cookie": f"access_token={access_token.strip()}",
    }
    status, _reason, data, _text = _http_json_request(
        endpoint=endpoint,
        method="GET",
        path=f"/api/task/{task_id}",
        headers=headers,
        payload=None,
        timeout_sec=timeout_sec,
        verify_ssl=verify_ssl,
        proxy_url=proxy_url,
    )
    if status != 200 or not isinstance(data, dict):
        return None
    return data


def _wait_for_sqc_task_bitstring(
    task_id: str,
    endpoint: str,
    access_token: str,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
) -> tuple[Optional[str], str]:
    max_wait_sec = max(5, int(timeout_sec))
    poll_interval_sec = 2.0
    deadline = time.monotonic() + max_wait_sec
    last_status = "Unknown"

    while time.monotonic() < deadline:
        item = _query_sqc_task_item(
            task_id=task_id,
            endpoint=endpoint,
            access_token=access_token,
            timeout_sec=timeout_sec,
            proxy_url=proxy_url,
            verify_ssl=verify_ssl,
        )
        detail = _query_sqc_task_detail(
            task_id=task_id,
            endpoint=endpoint,
            access_token=access_token,
            timeout_sec=timeout_sec,
            proxy_url=proxy_url,
            verify_ssl=verify_ssl,
        )

        status_text = "Unknown"
        if item:
            status_text = str(item.get("status") or status_text).strip() or "Unknown"
        elif detail:
            status_text = str(detail.get("status") or status_text).strip() or "Unknown"
        last_status = status_text

        bitstring = _extract_bitstring_from_payload(item) if item else None
        if not bitstring and detail:
            bitstring = _extract_bitstring_from_payload(detail)
        if bitstring:
            return bitstring, f"SQC task status={status_text}; bitstring retrieved."

        status_lower = status_text.lower()
        if status_lower in {"failed", "canceled", "cancelled", "error"}:
            return None, f"SQC task status={status_text}."
        if status_lower in {"completed", "finished", "done", "success", "succeeded"}:
            return None, f"SQC task status={status_text}; no bitstring in response."

        time.sleep(poll_interval_sec)

    return None, f"SQC task still pending after {max_wait_sec}s (last status={last_status})."


def _submit_once_with_sdk(
    selected: list[Customer],
    api_token: str,
    backend: Optional[str],
    endpoint: str,
    shots: int,
    wait: bool,
    timeout_sec: int,
    proxy_url: str,
    verify_ssl: bool,
    qasm_override: str = "",
) -> QuafuJobResult:
    from quafu import QuantumCircuit, Task, User
    from quafu.exceptions import ServerError
    from quafu.utils.client_wrapper import ClientWrapper

    qasm_override = (qasm_override or "").strip()
    if qasm_override:
        match = re.search(r"\bqreg\s+\w+\[(\d+)\]\s*;", qasm_override)
        if not match:
            raise ValueError("OpenQASM does not contain a valid qreg declaration.")
        qc = QuantumCircuit(int(match.group(1)))
        qc.from_openqasm(qasm_override)
    else:
        qc = _build_partition_circuit(selected)
    proxy_url = (proxy_url or "").strip()
    timeout_sec = max(5, int(timeout_sec))

    proxies = None
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}

    class _CompatResponse:
        def __init__(self, status_code: int, reason: str, body: bytes):
            self.status_code = int(status_code)
            self.reason = reason
            self._body = body or b""
            self.ok = 200 <= self.status_code < 300

        @property
        def text(self) -> str:
            return self._body.decode("utf-8", errors="replace")

        def json(self):
            return json.loads(self.text)

        def raise_for_status(self):
            if not self.ok:
                raise ServerError(
                    f"HTTP {self.status_code} {self.reason}: {self.text[:240]}"
                )

    def _http_client_post(url: str, headers: dict, data, json_payload, timeout: int, verify: bool):
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.hostname:
            raise ServerError(f"Invalid URL for Quafu request: {url}")

        is_https = parsed.scheme.lower() == "https"
        target_host = parsed.hostname
        target_port = parsed.port or (443 if is_https else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        body_bytes: bytes
        req_headers = dict(headers or {})

        if json_payload is not None:
            body_bytes = json.dumps(json_payload, ensure_ascii=False).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(data, bytes):
            body_bytes = data
        elif isinstance(data, dict):
            body_bytes = urlencode(data).encode("utf-8")
        elif data is None:
            body_bytes = b""
        else:
            body_bytes = str(data).encode("utf-8")

        context = None
        if is_https:
            context = ssl.create_default_context()
            if not verify:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

        if proxies and proxies.get("https", "").startswith("http://"):
            # HTTP proxy CONNECT tunnel for HTTPS endpoints.
            proxy_parsed = urlparse(proxies["https"])
            proxy_host = proxy_parsed.hostname
            proxy_port = proxy_parsed.port or 80
            if not proxy_host:
                raise ServerError(f"Invalid proxy URL: {proxies['https']}")
            if is_https:
                conn = http.client.HTTPSConnection(
                    proxy_host,
                    proxy_port,
                    timeout=timeout,
                    context=context,
                )
                conn.set_tunnel(target_host, target_port)
            else:
                conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout)
        else:
            if is_https:
                conn = http.client.HTTPSConnection(
                    target_host,
                    target_port,
                    timeout=timeout,
                    context=context,
                )
            else:
                conn = http.client.HTTPConnection(target_host, target_port, timeout=timeout)

        try:
            conn.request("POST", path, body=body_bytes, headers=req_headers)
            resp = conn.getresponse()
            payload = resp.read()
            return _CompatResponse(resp.status, resp.reason, payload)
        finally:
            conn.close()

    original_post = ClientWrapper.post

    def _patched_post(*args, **kwargs):
        try:
            url = kwargs.get("url") or (args[0] if args else "")
            headers = kwargs.get("headers", {})
            data = kwargs.get("data", None)
            json_payload = kwargs.get("json", None)
            timeout = int(kwargs.get("timeout", timeout_sec))
            verify = bool(kwargs.get("verify", verify_ssl))
            response = _http_client_post(
                url=url,
                headers=headers,
                data=data,
                json_payload=json_payload,
                timeout=timeout,
                verify=verify,
            )
            response.raise_for_status()
            return response
        except Exception as err:
            raise ServerError(
                f"Failed to communicate with quafu website via endpoint={endpoint}, err: {err}"
            ) from err

    ClientWrapper.post = staticmethod(_patched_post)
    try:
        User.url = endpoint
        user = User(api_token=api_token.strip())
        task = Task(user=user)

        backend_name = backend.strip() if backend else task.backend.name
        try:
            task.config(
                backend=backend_name,
                shots=max(100, int(shots)),
                compile=True,
                tomo=False,
                priority=2,
            )
        except Exception:
            backend_name = task.backend.name
            task.config(
                backend=backend_name,
                shots=max(100, int(shots)),
                compile=True,
                tomo=False,
                priority=2,
            )

        result = task.send(qc, name="QRF_partition_seed", wait=wait)
        bitstring = None
        counts = getattr(result, "counts", None)
        if counts:
            bitstring = max(counts.items(), key=lambda kv: kv[1])[0]

        return QuafuJobResult(
            ok=True,
            message="Submitted to Quafu successfully.",
            task_id=str(getattr(result, "taskid", "")) or None,
            backend=backend_name,
            bitstring=bitstring,
            selected_customer_ids=[c.customer_id for c in selected],
            endpoint=endpoint,
        )
    finally:
        ClientWrapper.post = original_post


def _pick_customers_for_quantum(customers: list[Customer], max_qubits: int) -> list[Customer]:
    # Prioritize high-demand points so quantum output influences the hardest customers.
    ranked = sorted(customers, key=lambda c: (-c.demand, c.customer_id))
    return ranked[: max(2, min(max_qubits, len(ranked)))]


def _build_partition_circuit(customers: list[Customer]):
    from quafu import QuantumCircuit  # lazy import

    n = len(customers)
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.h(i)

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(euclidean(customers[i].point, customers[j].point))
    max_dist = max(dists) if dists else 1.0

    gamma = 1.1
    beta = 0.8
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(customers[i].point, customers[j].point)
            proximity = 1.0 - min(1.0, dist / max_dist)
            weight = 0.25 + 0.75 * proximity
            qc.rzz(i, j, gamma * weight)

    for i in range(n):
        qc.rx(i, beta)

    qc.measure(list(range(n)), list(range(n)))
    return qc


def submit_quafu_partition_job(
    customers: list[Customer],
    api_token: str,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    shots: int = 1024,
    wait: bool = True,
    max_qubits: int = 8,
    timeout_sec: int = 25,
    proxy_url: str = "",
    verify_ssl: bool = True,
) -> QuafuJobResult:
    """Submit a small QAOA-style partition circuit to Quafu and return task metadata."""
    if not api_token or not api_token.strip():
        return QuafuJobResult(ok=False, message="Quafu token is empty.")

    selected = _pick_customers_for_quantum(
        customers,
        max_qubits=max_qubits,
    )
    return submit_quafu_qasm_job(
        qasm=_build_partition_qasm(selected),
        selected_customers=selected,
        api_token=api_token,
        backend=backend,
        base_url=base_url,
        shots=shots,
        wait=wait,
        timeout_sec=timeout_sec,
        proxy_url=proxy_url,
        verify_ssl=verify_ssl,
    )


def submit_quafu_qasm_job(
    qasm: str,
    selected_customers: Iterable[Customer],
    api_token: str,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    shots: int = 1024,
    wait: bool = True,
    timeout_sec: int = 25,
    proxy_url: str = "",
    verify_ssl: bool = True,
) -> QuafuJobResult:
    """Submit caller-built OpenQASM through quark.Task, with legacy fallbacks."""
    qasm = (qasm or "").strip()
    selected = list(selected_customers)
    if not qasm.startswith("OPENQASM 2.0;"):
        return QuafuJobResult(
            ok=False,
            message="Custom circuit must be OpenQASM 2.0.",
            selected_customer_ids=[c.customer_id for c in selected],
        )
    match = re.search(r"\bqreg\s+\w+\[(\d+)\]\s*;", qasm)
    if not match:
        return QuafuJobResult(
            ok=False,
            message="Custom OpenQASM does not contain a valid qreg declaration.",
            selected_customer_ids=[c.customer_id for c in selected],
        )
    num_qubits = int(match.group(1))
    if num_qubits < 2 or num_qubits > 8:
        return QuafuJobResult(
            ok=False,
            message="Custom QAOA circuit must use between 2 and 8 qubits.",
            selected_customer_ids=[c.customer_id for c in selected],
        )
    if num_qubits != len(selected):
        return QuafuJobResult(
            ok=False,
            message=(
                f"QASM uses {num_qubits} qubits but "
                f"{len(selected)} selected customers were provided."
            ),
            selected_customer_ids=[c.customer_id for c in selected],
        )
    if not api_token or not api_token.strip():
        return QuafuJobResult(
            ok=False,
            message="Quafu token is empty.",
            selected_customer_ids=[c.customer_id for c in selected],
        )

    attempts: list[str] = []
    token_is_jwt = _looks_like_jwt_access_token(api_token)

    # This is the platform-supported path used by the original project. QSteed
    # compilation happens in the cloud; qsteed is not a local runtime dependency.
    if not token_is_jwt:
        try:
            normalize_local_proxy_environment()
            from quark import Task

            chip = (backend or "").strip() or "Dongling"
            normalized_shots = _normalize_shots(shots)
            manager = Task(api_token.strip())
            task_payload = {
                "chip": chip,
                "name": "QRF_sparse_maxcut_qaoa",
                "circuit": qasm,
                "shots": normalized_shots,
                "options": {
                    "compiler": "qsteed",
                    "correct": False,
                    "open_dd": None,
                    "target_qubits": [],
                },
            }
            task_id = str(manager.run(task_payload))
            endpoint = str(getattr(Task, "URL", "") or "") or None
            shots_note = (
                ""
                if normalized_shots == int(shots)
                else f" shots {shots}->{normalized_shots}."
            )
            if not wait:
                return QuafuJobResult(
                    ok=True,
                    message=(
                        f"Submitted via quark.Task to chip={chip};"
                        f"{shots_note} result polling skipped."
                    ),
                    task_id=task_id,
                    backend=chip,
                    selected_customer_ids=[
                        customer.customer_id for customer in selected
                    ],
                    endpoint=endpoint,
                )

            raw_result = manager.result(
                task_id,
                timeout=max(5, int(timeout_sec)),
            )
            counts: dict[str, int] = {}
            status = "unknown"
            error_text = ""
            if isinstance(raw_result, dict):
                status = str(raw_result.get("status") or "unknown")
                error_text = str(raw_result.get("error") or "")
                counts = (
                    _clean_counts(raw_result.get("corrected"))
                    or _clean_counts(raw_result.get("count"))
                )
            bitstring = (
                max(counts.items(), key=lambda item: item[1])[0]
                if counts
                else None
            )
            result_note = (
                f"{len(counts)} outcomes over {sum(counts.values())} shots"
                if counts
                else "no counts returned yet"
            )
            if error_text:
                result_note += f"; error={error_text}"
            return QuafuJobResult(
                ok=True,
                message=(
                    f"Submitted via quark.Task to chip={chip};"
                    f"{shots_note} status={status}; {result_note}."
                ),
                task_id=task_id,
                backend=chip,
                bitstring=bitstring,
                selected_customer_ids=[
                    customer.customer_id for customer in selected
                ],
                endpoint=endpoint,
                counts=counts or None,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            attempts.append(f"quark.Task -> {type(exc).__name__}: {exc}")

    endpoints = _candidate_base_urls(base_url)
    if token_is_jwt:
        sqc_endpoints = [url for url in endpoints if _is_sqc_endpoint(url)]
        endpoints = sqc_endpoints or ["https://quafu-sqc.baqis.ac.cn/"]

    for endpoint in endpoints:
        dns_hint = _dns_hint_for_base_url(endpoint)
        try:
            if _is_sqc_endpoint(endpoint) or token_is_jwt:
                result = _submit_once_with_sqc(
                    selected=selected,
                    access_token=api_token,
                    backend=backend,
                    endpoint=endpoint,
                    shots=shots,
                    wait=wait,
                    timeout_sec=timeout_sec,
                    proxy_url=proxy_url,
                    verify_ssl=verify_ssl,
                    qasm_override=qasm,
                )
            else:
                result = _submit_once_with_sdk(
                    selected=selected,
                    api_token=api_token,
                    backend=backend,
                    endpoint=endpoint,
                    shots=shots,
                    wait=wait,
                    timeout_sec=timeout_sec,
                    proxy_url=proxy_url,
                    verify_ssl=verify_ssl,
                    qasm_override=qasm,
                )
            return QuafuJobResult(
                ok=result.ok,
                message=f"{result.message} | Endpoint={endpoint} | {dns_hint}",
                task_id=result.task_id,
                backend=result.backend,
                bitstring=result.bitstring,
                selected_customer_ids=[c.customer_id for c in selected],
                endpoint=endpoint,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(f"{endpoint} -> {error} | {dns_hint}")
            if _is_auth_error(error) and not token_is_jwt:
                break

    return QuafuJobResult(
        ok=False,
        message=(
            "Custom OpenQASM submission failed on all endpoints: "
            + " || ".join(attempts[-3:])
        ),
        backend=backend,
        selected_customer_ids=[c.customer_id for c in selected],
    )


def fetch_quafu_task_bitstring(
    task_id: str,
    api_token: str,
    base_url: Optional[str] = None,
    timeout_sec: int = 25,
    proxy_url: str = "",
    verify_ssl: bool = True,
) -> QuafuJobResult:
    task_id = str(task_id or "").strip()
    if not task_id:
        return QuafuJobResult(ok=False, message="Task id is empty.")
    if not api_token or not api_token.strip():
        return QuafuJobResult(ok=False, message="Quafu token is empty.")

    token_is_jwt = _looks_like_jwt_access_token(api_token)
    attempts: list[str] = []
    if not token_is_jwt:
        try:
            normalize_local_proxy_environment()
            from quark import Task

            manager = Task(api_token.strip())
            raw_result = manager.result(
                task_id,
                timeout=max(5, int(timeout_sec)),
            )
            counts: dict[str, int] = {}
            status = "unknown"
            chip = ""
            if isinstance(raw_result, dict):
                counts = (
                    _clean_counts(raw_result.get("corrected"))
                    or _clean_counts(raw_result.get("count"))
                )
                status = str(raw_result.get("status") or "unknown")
                chip = str(raw_result.get("chip") or "")
            if counts:
                return QuafuJobResult(
                    ok=True,
                    message=(
                        f"quark.Task result: status={status}; "
                        f"{len(counts)} outcomes over "
                        f"{sum(counts.values())} shots."
                    ),
                    task_id=task_id,
                    backend=chip or None,
                    bitstring=max(
                        counts.items(),
                        key=lambda item: item[1],
                    )[0],
                    endpoint=str(getattr(Task, "URL", "") or "") or None,
                    counts=counts,
                )
            attempts.append(
                f"quark.Task -> status={status}, no counts available"
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            attempts.append(
                f"quark.Task -> {type(exc).__name__}: {exc}"
            )

    endpoints = _candidate_base_urls(base_url)
    if token_is_jwt:
        sqc_endpoints = [u for u in endpoints if _is_sqc_endpoint(u)]
        endpoints = sqc_endpoints or ["https://quafu-sqc.baqis.ac.cn/"]

    for endpoint in endpoints:
        dns_hint = _dns_hint_for_base_url(endpoint)
        if not (_is_sqc_endpoint(endpoint) or token_is_jwt):
            attempts.append(f"{endpoint} -> task query currently supports SQC endpoints only")
            continue
        try:
            bitstring, msg = _wait_for_sqc_task_bitstring(
                task_id=task_id,
                endpoint=endpoint,
                access_token=api_token,
                timeout_sec=timeout_sec,
                proxy_url=proxy_url,
                verify_ssl=verify_ssl,
            )
            item = _query_sqc_task_item(
                task_id=task_id,
                endpoint=endpoint,
                access_token=api_token,
                timeout_sec=timeout_sec,
                proxy_url=proxy_url,
                verify_ssl=verify_ssl,
            )
            backend = None
            if item:
                backend = str(item.get("chipName") or item.get("chip") or "").strip() or None
            return QuafuJobResult(
                ok=bitstring is not None,
                message=f"{msg} | Endpoint={endpoint} | {dns_hint}",
                task_id=task_id,
                backend=backend,
                bitstring=bitstring,
                selected_customer_ids=None,
                endpoint=endpoint,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            err = f"{type(exc).__name__}: {exc}"
            attempts.append(f"{endpoint} -> {err} | {dns_hint}")

    summary = " || ".join(attempts[-3:])
    return QuafuJobResult(
        ok=False,
        message=f"Failed to query Quafu task bitstring: {summary}",
        task_id=task_id,
        backend=None,
        bitstring=None,
        selected_customer_ids=None,
        endpoint=None,
    )


def bitstring_to_vehicle_hints(
    selected_customers: Iterable[Customer],
    bitstring: str,
    num_vehicles: int,
) -> dict[int, int]:
    """Map qubit bitstring to preferred vehicle ids for first two vehicles."""
    if num_vehicles <= 1:
        return {}

    hints: dict[int, int] = {}
    ordered = list(selected_customers)
    for idx, customer in enumerate(ordered):
        # Quafu bitstring order can differ by backend mapping; this mirror mapping keeps deterministic behavior.
        bit_index = len(bitstring) - 1 - idx
        bit = "0"
        if 0 <= bit_index < len(bitstring):
            bit = bitstring[bit_index]
        hints[customer.customer_id] = 0 if bit == "0" else 1
    return hints

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import getproxies


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def read_env_file(path: Path | str | None = None) -> dict[str, str]:
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_value(
    name: str,
    default: str = "",
    *,
    path: Path | str | None = None,
) -> str:
    process_value = (os.environ.get(name) or "").strip()
    if process_value:
        return process_value
    file_value = (read_env_file(path).get(name) or "").strip()
    return file_value or default


def load_env_file(
    path: Path | str | None = None,
    *,
    override: bool = False,
) -> list[str]:
    applied: list[str] = []
    for key, value in read_env_file(path).items():
        if not override and (os.environ.get(key) or "").strip():
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def quafu_token(*, path: Path | str | None = None) -> str:
    from_environ = (os.environ.get("QUAFU_API_TOKEN") or "").strip()
    if from_environ and not any(
        ord(character) < 32 or ord(character) == 127
        for character in from_environ
    ):
        return from_environ
    from_file = (
        read_env_file(path).get("QUAFU_API_TOKEN") or ""
    ).strip()
    if any(
        ord(character) < 32 or ord(character) == 127
        for character in from_file
    ):
        return ""
    return from_file


def quafu_chip(
    default: str = "Dongling",
    *,
    path: Path | str | None = None,
) -> str:
    return env_value("QUAFU_CHIP", default, path=path)


def normalize_local_proxy_environment() -> bool:
    """Correct Windows proxy metadata for a local HTTP CONNECT proxy.

    Some Windows proxy tools register an HTTPS target as
    ``https://127.0.0.1:PORT`` even though the local listener speaks ordinary
    HTTP CONNECT. Requests then sends TLS to a plaintext proxy and receives a
    malformed response. Only loopback proxy addresses are normalized.
    """
    proxies = getproxies()
    changed = False
    for scheme in ("http", "https"):
        value = str(proxies.get(scheme) or "").strip()
        parsed = urlparse(value)
        if (
            parsed.scheme.lower() == "https"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            port = parsed.port or 443
            corrected = f"http://{parsed.hostname}:{port}"
            os.environ[f"{scheme.upper()}_PROXY"] = corrected
            os.environ[f"{scheme.lower()}_proxy"] = corrected
            changed = True
    return changed

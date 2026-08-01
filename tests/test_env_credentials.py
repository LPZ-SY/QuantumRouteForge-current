from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_route_forge.env import quafu_token


def test_invalid_process_token_falls_back_to_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("QUAFU_API_TOKEN=file-token-123\n", encoding="utf-8")
    monkeypatch.setenv("QUAFU_API_TOKEN", "broken\x08token")

    assert quafu_token(path=env_path) == "file-token-123"


def test_valid_process_token_has_priority(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("QUAFU_API_TOKEN=file-token\n", encoding="utf-8")
    monkeypatch.setenv("QUAFU_API_TOKEN", "process-token")

    assert quafu_token(path=env_path) == "process-token"

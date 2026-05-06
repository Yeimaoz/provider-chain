"""Test telemetry _log_usage."""
import json
from pathlib import Path


def test_log_usage_writes_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setattr("provider_chain.chain.USAGE_LOG", log_path)

    from provider_chain.chain import _log_usage
    _log_usage("ollama", "ner", "qwen2.5:7b", 100, 50, 234.5, success=True)

    assert log_path.exists()
    line = log_path.read_text().strip()
    record = json.loads(line)
    assert record["provider"] == "ollama"
    assert record["task"] == "ner"
    assert record["in_tokens"] == 100
    assert record["out_tokens"] == 50
    assert record["latency_ms"] == 234.5
    assert record["success"] is True


def test_log_usage_disabled_via_env(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setattr("provider_chain.chain.USAGE_LOG", log_path)
    monkeypatch.setenv("LCZ_LLM_CHAIN_DISABLE_TELEMETRY", "1")

    from provider_chain.chain import _log_usage
    _log_usage("ollama", "ner", "qwen2.5:7b", 100, 50, 234.5, True)

    assert not log_path.exists()


def test_log_usage_silent_on_fs_error(monkeypatch):
    """Should not raise when log path is unwritable."""
    monkeypatch.setattr("provider_chain.chain.USAGE_LOG", Path("/nonexistent/no/perm/usage.jsonl"))

    from provider_chain.chain import _log_usage
    # should not raise
    _log_usage("ollama", "ner", "model", 0, 0, 0, True)

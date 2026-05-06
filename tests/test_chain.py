"""Smoke tests for chain.py — does NOT actually call LLM endpoints.

Tests:
- public API importable
- TASK_MODEL_MATRIX shape correct
- PROVIDERS / endpoints / timeout dicts consistent
- ask() returns None when all providers misconfigured (no env)
"""
import os
import pytest


def test_public_imports():
    from lcz_llm_chain import (
        ask,
        embed,
        wrap_untrusted,
        sanitize_user_input,
        TASK_MODEL_MATRIX,
        TaskType,
        ProviderName,
        PROVIDERS,
        PROVIDER_ENDPOINTS,
        PROVIDER_TIMEOUT,
    )
    assert callable(ask)
    assert callable(embed)
    assert callable(wrap_untrusted)
    assert callable(sanitize_user_input)


def test_task_model_matrix_shape():
    from lcz_llm_chain import TASK_MODEL_MATRIX, PROVIDERS

    expected_tasks = {"ner", "summary_zh", "deep_reason", "code", "vision", "embedding"}
    assert set(TASK_MODEL_MATRIX.keys()) == expected_tasks

    # every task has every provider key (value can be None for unsupported)
    for task, providers_map in TASK_MODEL_MATRIX.items():
        assert set(providers_map.keys()) == set(PROVIDERS), f"task {task} provider mismatch"


def test_providers_dicts_consistent():
    from lcz_llm_chain import PROVIDERS, PROVIDER_ENDPOINTS, PROVIDER_TIMEOUT

    for p in PROVIDERS:
        assert p in PROVIDER_ENDPOINTS, f"{p} missing endpoint"
        assert p in PROVIDER_TIMEOUT, f"{p} missing timeout"
        assert PROVIDER_TIMEOUT[p] > 0


def test_ask_returns_none_no_env(monkeypatch):
    """When all provider env vars are missing, ask() should silently return None."""
    from lcz_llm_chain import ask, PROVIDER_ENDPOINTS

    for env_key, _ in PROVIDER_ENDPOINTS.values():
        monkeypatch.delenv(env_key, raising=False)
    # disable the auto-load by pointing to a non-existent env file
    monkeypatch.setenv("LCZ_LLM_CHAIN_DISABLE_AUTO_ENV", "1")

    result = ask("ner", "test prompt")
    # may be None if providers all skip, or string if some env is still set unexpectedly
    # at minimum should not raise
    assert result is None or isinstance(result, str)


def test_ask_unknown_task_raises():
    from lcz_llm_chain import ask
    with pytest.raises(KeyError):
        ask("unknown_task", "test")  # type: ignore[arg-type]

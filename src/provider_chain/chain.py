"""llm_chain.py — 4-tier task-aware LLM provider chain

Worldmonitor scripts/lib/llm-chain.cjs:18-49 啟發（Ollama → Groq → DeepSeek → OpenRouter）。

Provider fallback 順序：
  1. Ollama (local, free, fast — preferred)
  2. Groq  (cloud, free tier, fast inference)
  3. DeepSeek (cloud, cheap, strong reasoning)
  4. OpenRouter (cloud, universal fallback)

Task routing:
  ner         — entity extraction (short, fast)
  summary_zh  — Chinese summarisation
  deep_reason — chain-of-thought / reasoning
  code        — code generation / review
  vision      — image + text multimodal
  embedding   — vector embedding

Auto env load from python/order_server/.env (same pattern as lib/notify/discord.py).
Silent fail: all exceptions are caught; returns None when all providers fail.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

USAGE_LOG = Path.home() / ".provider-chain-usage.jsonl"


def _log_usage(
    provider: str,
    task: str,
    model: str,
    in_tokens: int,
    out_tokens: int,
    latency_ms: float,
    success: bool,
) -> None:
    """Append usage record to ~/.provider-chain-usage.jsonl. Silent fail.

    Disable via LCZ_LLM_CHAIN_DISABLE_TELEMETRY=1.
    """
    if os.environ.get("LCZ_LLM_CHAIN_DISABLE_TELEMETRY") == "1":
        return
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "task": task,
            "model": model,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never block primary call


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TaskType = Literal["ner", "summary_zh", "deep_reason", "code", "vision", "embedding"]
ProviderName = Literal["gemini", "ollama", "groq", "deepseek", "openrouter"]

# ---------------------------------------------------------------------------
# Task → Model matrix
# ---------------------------------------------------------------------------

TASK_MODEL_MATRIX: dict[TaskType, dict[ProviderName, Optional[str]]] = {
    "ner": {
        "gemini":     "gemini-2.0-flash",
        "ollama":     "qwen2.5:7b",
        "groq":       "llama-3.3-70b-versatile",
        "deepseek":   "deepseek-chat",
        "openrouter": "openai/gpt-oss-120b:free",  # v0.4.0: 切到 free tier reasoning model（中文 NER 質量驗過）
    },
    "summary_zh": {
        "gemini":     "gemini-2.0-flash",
        "ollama":     "glm4:9b",
        "groq":       "llama-3.3-70b-versatile",
        "deepseek":   "deepseek-chat",
        "openrouter": "openai/gpt-oss-120b:free",
    },
    "deep_reason": {
        "gemini":     "gemini-2.0-flash-thinking-exp",
        "ollama":     "deepseek-r1:14b",
        "groq":       None,                       # groq does not host reasoning models
        "deepseek":   "deepseek-reasoner",
        "openrouter": "openai/o1-mini",
    },
    "code": {
        "gemini":     "gemini-2.0-flash",
        "ollama":     "qwen2.5-coder:7b",
        "groq":       "llama-3.3-70b-versatile",
        "deepseek":   "deepseek-coder",
        "openrouter": "google/gemini-2.0-flash",
    },
    "vision": {
        "gemini":     "gemini-2.0-flash",
        "ollama":     "qwen2.5vl:7b",
        "groq":       None,
        "deepseek":   None,
        "openrouter": "google/gemini-2.0-flash",
    },
    "embedding": {
        "gemini":     "text-embedding-004",
        "ollama":     "bge-m3",
        "groq":       None,
        "deepseek":   None,
        "openrouter": "openai/text-embedding-3-small",
    },
}

# v0.3.0 (2026-05-04)：gemini native API 提到第一順位（user 已有 GEMINI_API_KEY，免轉 openrouter wrapper）。
PROVIDERS: list[ProviderName] = ["gemini", "openrouter", "groq", "deepseek", "ollama"]
# v0.2.0 (2026-05-04)：vendor-first chain。openrouter (gemini) > groq > deepseek > ollama (本地最後 fallback)。
# 動機：cloud vendor 品質高且 free tier 夠用，本地 ollama 留作離線/限速 fallback；舊版 ollama-first 會在桌機沒開時整條 fail。

# v0.2.0：暴露最後一次成功的 provider/model 給 caller（embed 字卡顯示來源用）。
# 不破舊 API：ask() 簽名不變，caller 自行讀 LAST_USED；single-thread 安全，多 thread 場景請用 ask_with_meta()。
LAST_USED: dict[str, str | None] = {"provider": None, "model": None, "task": None}

# Per-task provider order override. Falls back to PROVIDERS if task missing.
# Rationale (v0.4.0, 2026-05-06):
#   - ner / summary_zh / code: gemini → groq → openrouter (free) → ollama
#       Gemini 1500 RPD 第一順位（量最大）
#       Groq 100k TPD 第二（~111 篇/day）
#       OpenRouter free (gpt-oss-120b:free) 第三（~50-200 req/day per account）
#       Ollama 本機最後 fallback（debug 確認三個 cloud 全爆才用）
#   - vision: gemini > ollama qwen2.5vl (local) > openrouter gemini fallback
TASK_PROVIDERS: dict[TaskType, list[ProviderName]] = {
    "summary_zh":  ["gemini", "groq", "openrouter", "ollama"],
    "ner":         ["gemini", "groq", "openrouter", "ollama"],
    "vision":      ["gemini", "ollama", "openrouter"],
    "deep_reason": ["gemini", "ollama", "deepseek", "openrouter"],
    "code":        ["gemini", "groq", "openrouter", "ollama"],
}

# (env_key, base_url_or_None)
# For ollama, env_key holds the *base URL* itself; None means use default.
PROVIDER_ENDPOINTS: dict[ProviderName, tuple[str, Optional[str]]] = {
    "gemini":     ("GEMINI_API_KEY",     "https://generativelanguage.googleapis.com/v1beta"),
    "ollama":     ("OLLAMA_API_URL",     None),
    "groq":       ("GROQ_API_KEY",       "https://api.groq.com/openai/v1"),
    "deepseek":   ("DEEPSEEK_API_KEY",   "https://api.deepseek.com/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}

# Timeout per provider (seconds); vision gets 2× at call site
PROVIDER_TIMEOUT: dict[ProviderName, float] = {
    "gemini":     30.0,
    "ollama":     30.0,
    "groq":       15.0,
    "deepseek":   30.0,
    "openrouter": 30.0,
}

# ---------------------------------------------------------------------------
# Env loading (once per process)
# ---------------------------------------------------------------------------

_ENV_LOADED = False
_ENV_CACHE: dict[str, str] = {}

_ENV_KEYS = ("GEMINI_API_KEY", "OLLAMA_API_URL", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY")


def _load_env_once() -> None:
    """Load API keys / URLs into _ENV_CACHE (once per process).

    Priority:
      1. Process environment (always wins)
      2. File pointed to by ``LCZ_LLM_CHAIN_ENV_FILE`` env var (if set)
      3. ``./.env`` in current working directory (if exists)

    Set ``LCZ_LLM_CHAIN_DISABLE_AUTO_ENV=1`` to skip file loading entirely
    (recommended when consumer manages env via dotenv / systemd / etc).
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    # 1. Process environment takes priority
    for key in _ENV_KEYS:
        val = os.environ.get(key)
        if val:
            _ENV_CACHE[key] = val

    if os.environ.get("LCZ_LLM_CHAIN_DISABLE_AUTO_ENV") == "1":
        return

    # 2. Explicit env file via LCZ_LLM_CHAIN_ENV_FILE, else ./.env
    candidate_paths = []
    explicit = os.environ.get("LCZ_LLM_CHAIN_ENV_FILE")
    if explicit:
        candidate_paths.append(Path(explicit))
    else:
        candidate_paths.append(Path.cwd() / ".env")

    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k in _ENV_KEYS and k not in _ENV_CACHE:
                    _ENV_CACHE[k] = v.strip().strip('"').strip("'")
            break
        except Exception as exc:
            logger.debug(f"[provider_chain] .env read failed at {env_path}: {exc}")


def _get_env(key: str) -> Optional[str]:
    _load_env_once()
    return _ENV_CACHE.get(key) or None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_headers(provider: ProviderName, env_val: str) -> dict[str, str]:
    """Build HTTP headers for a given provider."""
    base = {"Content-Type": "application/json", "User-Agent": "provider-chain/0.1.0"}
    if provider == "ollama":
        # env_val is the base URL, not an API key
        # OLLAMA_API_KEY is optional; add if set
        api_key = os.environ.get("OLLAMA_API_KEY") or _ENV_CACHE.get("OLLAMA_API_KEY")
        if api_key:
            base["Authorization"] = f"Bearer {api_key}"
    elif provider == "openrouter":
        base["Authorization"] = f"Bearer {env_val}"
        base["HTTP-Referer"] = os.getenv("LLM_CHAIN_HTTP_REFERER", "https://github.com")
        base["X-Title"] = os.getenv("LLM_CHAIN_APP_TITLE", "provider-chain")
    else:
        # groq / deepseek — standard Bearer
        base["Authorization"] = f"Bearer {env_val}"
    return base


def _build_url(provider: ProviderName, env_val: str, endpoint: str) -> str:
    """Construct the full API URL for a provider + endpoint."""
    if provider == "ollama":
        base = env_val.rstrip("/") if env_val else "http://localhost:11434"
        return f"{base}/v1/{endpoint}"
    _, base_url = PROVIDER_ENDPOINTS[provider]
    assert base_url is not None
    return f"{base_url}/{endpoint}"


def _chat_once(
    provider: ProviderName,
    env_val: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    task: str = "",
) -> Optional[str]:
    """Single attempt. OpenAI-compat for groq/deepseek/openrouter/ollama.
    Gemini native API（不同 schema）走獨立 path。Returns text or None."""
    if provider == "gemini":
        return _chat_gemini_once(env_val, model, messages, max_tokens, temperature, timeout, task)

    url = _build_url(provider, env_val, "chat/completions")
    headers = _build_headers(provider, env_val)
    body: dict = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    # Ollama: disable internal reasoning preamble
    if provider == "ollama":
        body["think"] = False

    start = time.monotonic()
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        in_tok = (
            data.get("usage", {}).get("prompt_tokens", 0)
            or data.get("prompt_eval_count", 0)
        )
        out_tok = (
            data.get("usage", {}).get("completion_tokens", 0)
            or data.get("eval_count", 0)
        )
        latency_ms = (time.monotonic() - start) * 1000
        _log_usage(provider, task, model, in_tok, out_tok, latency_ms, success=True)
        return data["choices"][0]["message"]["content"].strip() or None
    except Exception:
        latency_ms = (time.monotonic() - start) * 1000
        _log_usage(provider, task, model, 0, 0, latency_ms, success=False)
        raise


def _chat_gemini_once(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    task: str = "",
) -> Optional[str]:
    """Gemini native API call (non-OpenAI schema).

    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...
    Body: {contents: [{role, parts: [{text}]}], generationConfig: {maxOutputTokens, temperature}}
    Response: {candidates: [{content: {parts: [{text}]}}], usageMetadata: {...}}

    Role mapping: openai 'system' → gemini systemInstruction; 'user' / 'assistant' (model) → contents.
    """
    base_url = PROVIDER_ENDPOINTS["gemini"][1]
    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    contents: list[dict] = []
    system_instruction: Optional[dict] = None
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        if role == "system":
            system_instruction = {"parts": [{"text": text}]}
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_instruction is not None:
        body["systemInstruction"] = system_instruction

    headers = {"Content-Type": "application/json", "User-Agent": "provider-chain/0.3.0"}
    start = time.monotonic()
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        latency_ms = (time.monotonic() - start) * 1000
        candidates = data.get("candidates") or []
        if not candidates:
            _log_usage("gemini", task, model, in_tok, out_tok, latency_ms, success=False)
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        _log_usage("gemini", task, model, in_tok, out_tok, latency_ms, success=True)
        return text or None
    except Exception:
        latency_ms = (time.monotonic() - start) * 1000
        _log_usage("gemini", task, model, 0, 0, latency_ms, success=False)
        raise


def _embed_once(
    provider: ProviderName,
    env_val: str,
    model: str,
    text: str,
    timeout: float,
) -> Optional[list[float]]:
    """Single attempt at /embeddings. Returns vector or None."""
    url = _build_url(provider, env_val, "embeddings")
    headers = _build_headers(provider, env_val)
    body = {"model": model, "input": text}
    start = time.monotonic()
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        in_tok = data.get("usage", {}).get("prompt_tokens", 0)
        out_tok = 0
        latency_ms = (time.monotonic() - start) * 1000
        _log_usage(provider, "embedding", model, in_tok, out_tok, latency_ms, success=True)
        return data["data"][0]["embedding"]
    except Exception:
        latency_ms = (time.monotonic() - start) * 1000
        _log_usage(provider, "embedding", model, 0, 0, latency_ms, success=False)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    task: TaskType,
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    image_path: Optional[str] = None,
) -> Optional[str]:
    """Task-aware fallback chain for text generation.

    Tries each provider in PROVIDERS order; skips if:
    - model is None for this task (provider does not support it), or
    - required env var is missing.

    Retry policy:
    - 5xx: retry once with 1s backoff (exponential: 1s, 2s).
    - 4xx: no retry, immediately fallback to next provider.
    - Timeout / network error: no retry, fallback.

    Args:
        task:        Task type for model routing.
        prompt:      User prompt.
        system:      Optional system prompt.
        max_tokens:  Max tokens in completion (default 1024).
        temperature: Sampling temperature (default 0.3).
        image_path:  Path to image file for vision tasks (optional).

    Returns:
        Generated text string, or None if all providers failed.
    """
    _load_env_once()

    messages = _build_messages(task, prompt, system=system, image_path=image_path)
    base_timeout = PROVIDER_TIMEOUT.copy()
    if task == "vision":
        base_timeout = {k: v * 2 for k, v in base_timeout.items()}

    provider_order = TASK_PROVIDERS.get(task, PROVIDERS)
    for provider in provider_order:
        model = TASK_MODEL_MATRIX[task][provider]
        if model is None:
            logger.debug(f"[llm_chain] skip {provider}: no model for task={task}")
            continue

        env_key, _ = PROVIDER_ENDPOINTS[provider]
        env_val = _get_env(env_key)
        if not env_val:
            logger.debug(f"[llm_chain] skip {provider}: {env_key} not set")
            continue

        timeout = base_timeout[provider]
        last_exc: Exception | None = None

        for attempt in range(2):  # 0 = first try, 1 = one retry on 5xx
            try:
                result = _chat_once(
                    provider, env_val, model, messages,
                    max_tokens, temperature, timeout,
                    task=task,
                )
                logger.debug(f"[llm_chain] {provider}/{model} OK (task={task})")
                LAST_USED["provider"] = provider
                LAST_USED["model"] = model
                LAST_USED["task"] = task
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500 and attempt == 0:
                    wait = 2 ** attempt  # 1s on first retry
                    logger.warning(
                        f"[llm_chain] {provider} 5xx ({status}), retry in {wait}s"
                    )
                    time.sleep(wait)
                    last_exc = exc
                    continue
                # 4xx or 5xx on retry → fallback
                logger.warning(f"[llm_chain] {provider} HTTP {status}, fallback")
                last_exc = exc
                break
            except Exception as exc:
                logger.warning(f"[llm_chain] {provider} error: {exc}, fallback")
                last_exc = exc
                break

        if last_exc:
            logger.debug(f"[llm_chain] {provider} gave up: {last_exc}")

    logger.warning(f"[llm_chain] all providers failed for task={task}")
    return None


def embed(text: str, *, max_tokens: int = 0) -> Optional[list[float]]:
    """Return embedding vector for text, or None if all providers fail.

    Only providers with a non-None embedding model are tried:
    - ollama (bge-m3)
    - openrouter (openai/text-embedding-3-small)

    Args:
        text:       Input text to embed.
        max_tokens: Unused (kept for API consistency); embeddings use all tokens.

    Returns:
        List of floats (embedding vector), or None.
    """
    _load_env_once()

    for provider in PROVIDERS:
        model = TASK_MODEL_MATRIX["embedding"][provider]
        if model is None:
            continue

        env_key, _ = PROVIDER_ENDPOINTS[provider]
        env_val = _get_env(env_key)
        if not env_val:
            logger.debug(f"[llm_chain] embed skip {provider}: {env_key} not set")
            continue

        timeout = PROVIDER_TIMEOUT[provider]
        try:
            result = _embed_once(provider, env_val, model, text, timeout)
            logger.debug(f"[llm_chain] embed {provider}/{model} OK")
            return result
        except Exception as exc:
            logger.warning(f"[llm_chain] embed {provider} error: {exc}, fallback")

    logger.warning("[llm_chain] embed: all providers failed")
    return None


# ---------------------------------------------------------------------------
# Internal: message building
# ---------------------------------------------------------------------------

def _build_messages(
    task: TaskType,
    prompt: str,
    *,
    system: str = "",
    image_path: Optional[str] = None,
) -> list[dict]:
    """Build the messages list for /chat/completions."""
    messages: list[dict] = []

    if system:
        messages.append({"role": "system", "content": system})

    if task == "vision" and image_path:
        # Multimodal: encode image as base64 data URL
        try:
            img_bytes = Path(image_path).read_bytes()
            ext = Path(image_path).suffix.lstrip(".").lower()
            mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/jpeg"
            b64 = base64.b64encode(img_bytes).decode()
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            })
        except Exception as exc:
            logger.warning(f"[llm_chain] vision image encode failed: {exc}, falling back to text-only")
            messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": prompt})

    return messages

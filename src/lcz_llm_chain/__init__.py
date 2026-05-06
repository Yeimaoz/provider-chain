"""lcz-llm-chain — task-aware LLM provider chain + prompt injection sanitize.

Multi-provider LLM chain for projects needing task-aware fallback.
Supports 5 providers (Gemini / Groq / DeepSeek / OpenRouter / Ollama) with
configurable per-task provider order and prompt injection defense.

Public API:
    from lcz_llm_chain import ask, embed, wrap_untrusted, sanitize_user_input
    from lcz_llm_chain import TASK_MODEL_MATRIX, TaskType, ProviderName, PROVIDERS
"""
from .chain import (
    ask,
    embed,
    TASK_MODEL_MATRIX,
    TaskType,
    ProviderName,
    PROVIDERS,
    PROVIDER_ENDPOINTS,
    PROVIDER_TIMEOUT,
)
from .sanitize import (
    wrap_untrusted,
    sanitize_user_input,
)

__all__ = [
    "ask",
    "embed",
    "wrap_untrusted",
    "sanitize_user_input",
    "TASK_MODEL_MATRIX",
    "TaskType",
    "ProviderName",
    "PROVIDERS",
    "PROVIDER_ENDPOINTS",
    "PROVIDER_TIMEOUT",
]

__version__ = "1.0.0"

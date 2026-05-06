# provider-chain

Task-aware LLM provider chain with N-tier fallback and prompt injection defense.

## Why

- Managing multiple LLM providers (Gemini / Groq / DeepSeek / OpenRouter / Ollama) in each project is tedious
- Each project was duplicating API call code with no fallback
- Switching providers required N-place changes
- Prompt injection defense was scattered or missing

This library unifies multi-provider fallback, task-aware routing, and prompt safety in one place.

## Install

```bash
pip install "git+https://github.com/Yeimaoz/provider-chain.git@v1.0.0"
```

## Quick Start

```python
from provider_chain import ask, wrap_untrusted

# task-routed call — tries providers in order until one succeeds
result = ask("ner", "TSMC announced 2nm process node")

# wrap external content to defend prompt injection
prompt = f"Summarize: {wrap_untrusted(user_provided_html)}"
result = ask("summary_zh", prompt)
```

## Configuration

Set whichever provider keys you have. The chain skips providers whose key is missing.

| Env var | Provider | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini | 1500 RPD free tier |
| `GROQ_API_KEY` | Groq | 100k TPD free tier |
| `DEEPSEEK_API_KEY` | DeepSeek | Pay-as-you-go |
| `OPENROUTER_API_KEY` | OpenRouter | 50 RPD per `:free` model |
| `OLLAMA_API_URL` | Ollama (local) | Default `http://localhost:11434` |
| `LLM_CHAIN_HTTP_REFERER` | OpenRouter identity | Default `https://github.com` |
| `LLM_CHAIN_APP_TITLE` | OpenRouter identity | Default `provider-chain` |

See `examples/env_setup.example` for a ready-to-copy template.

## Task Types

| Task | Description |
|---|---|
| `ner` | Named entity recognition / structured extraction |
| `summary_zh` | Chinese summarization |
| `deep_reason` | Chain-of-thought / complex reasoning |
| `code` | Code generation and review |
| `vision` | Image + text multimodal |
| `embedding` | Vector embedding |

```python
result = ask("deep_reason", "Explain why momentum works in small-cap stocks", max_tokens=1024)
```

## Provider Chain Order

Default fallback order (highest priority first):

1. **Gemini** (Google, 1500 RPD free)
2. **Groq** (cloud, free tier, fast inference)
3. **DeepSeek** (cloud, cheap, strong reasoning)
4. **OpenRouter** (cloud, universal fallback with `:free` models)
5. **Ollama** (local, zero cost, no network needed)

The chain automatically skips any provider whose API key is not set.

### Per-task provider matrix

| Task | Gemini | Groq | DeepSeek | OpenRouter | Ollama |
|---|---|---|---|---|---|
| `ner` | gemini-2.0-flash | llama-3.3-70b | deepseek-chat | gemini-2.0-flash:free | qwen2.5:7b |
| `summary_zh` | gemini-2.0-flash | llama-3.3-70b | deepseek-chat | gemini-2.0-flash:free | glm4:9b |
| `deep_reason` | gemini-2.0-flash | — | deepseek-reasoner | openai/o1-mini | deepseek-r1:14b |
| `code` | gemini-2.0-flash | llama-3.3-70b | deepseek-coder | gemini-2.0-flash:free | qwen2.5-coder:7b |
| `vision` | gemini-2.0-flash | — | — | gemini-2.0-flash:free | qwen2.5vl:7b |
| `embedding` | — | — | — | openai/text-embedding-3-small | bge-m3 |

## Prompt Injection Defense

`wrap_untrusted(content)` wraps third-party content (RSS / scraped HTML / user input) in
`<untrusted_content>...</untrusted_content>` tags and strips known injection patterns.

`sanitize_user_input(text, max_chars=4000)` strips control characters, truncates, and removes
known injection regex matches.

```python
from provider_chain import wrap_untrusted, sanitize_user_input

safe_prompt = f"Extract entities from: {wrap_untrusted(scraped_html)}"
clean_query = sanitize_user_input(user_form_input)
```

See `src/provider_chain/sanitize.py` for the `INJECTION_PATTERNS` regex list.

## Examples

The `examples/` directory contains ready-to-run scripts:

- `examples/ner_chinese.py` — Chinese NER with structured JSON output
- `examples/summary_zh.py` — Chinese summarization
- `examples/multi_provider.py` — Inspect which provider answered
- `examples/env_setup.example` — Copy-paste env template

## License

MIT

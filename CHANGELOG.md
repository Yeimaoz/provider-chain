# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-06

### Added
- Initial open source release.
- 5-provider chain (Gemini / Groq / DeepSeek / OpenRouter / Ollama).
- 6 task types: ner / summary_zh / deep_reason / code / vision / embedding.
- Per-task provider order via `TASK_PROVIDERS`.
- Prompt injection sanitize (`wrap_untrusted`).
- Configurable OpenRouter App identity via `LLM_CHAIN_HTTP_REFERER` / `LLM_CHAIN_APP_TITLE` env vars.

### History
Pre-public internal versions (private repo, not published):
- v0.1.0: basic 4-tier chain.
- v0.2.0: vendor-first chain (openrouter > groq > deepseek > ollama).
- v0.3.0: gemini native API at top, debug single-provider toggle.
- v0.4.0: 4-tier with openrouter free tier (gpt-oss-120b:free) for ner/summary_zh.

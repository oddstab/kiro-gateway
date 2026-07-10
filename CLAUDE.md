# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run server
python main.py                    # default 0.0.0.0:8000
python main.py --port 9000        # custom port

# Tests
pytest                            # all tests
pytest -v --tb=short              # verbose with short tracebacks
pytest tests/unit/test_config.py  # single file
pytest -k "test_grok"             # by name pattern

# Docker
docker-compose up -d
docker build -t kiro-gateway .
```

## Architecture

Kiro Gateway is a transparent proxy that converts OpenAI/Anthropic API requests into Kiro API (Amazon Q Developer) calls and streams back responses.

**Request flow:**

```
Client → Route (auth + validate) → ModelResolver → Converter → HttpClient → Kiro API
                                                                    ↓
Client ← StreamingParser (SSE) ← ThinkingParser ← AWS EventStream ←┘
```

Key modules:
- `routes_openai.py` / `routes_anthropic.py` — FastAPI endpoints, validates PROXY_API_KEY
- `model_resolver.py` — 4-layer resolution: normalize → alias → dynamic cache → pass-through
- `converters_core.py` — shared conversion logic; `converters_openai.py` / `converters_anthropic.py` per-API format translation to Kiro payload
- `http_client.py` — sends to Kiro API with retry (403→token refresh, 429/5xx→backoff)
- `streaming_core.py` — shared AWS event stream parser; per-API streaming converts to SSE
- `thinking_parser.py` — FSM that extracts `<thinking>` blocks into OpenAI `reasoning_content`
- `account_manager.py` — multi-account failover with circuit breaker
- `auth.py` — token lifecycle, supports JSON file / refresh_token / SQLite / AWS SSO OIDC
- `config.py` — all settings via env vars, MODEL_ALIASES, FALLBACK_MODELS

**Non-obvious patterns:**
- "Fake reasoning" (`FAKE_REASONING=true`) injects `<thinking_mode>` tags into request, then parses response `<thinking>` blocks back into OpenAI reasoning_content field
- Truncation recovery auto-injects synthetic messages when Kiro API truncates a response
- Payload guard (`AUTO_TRIM_PAYLOAD`) auto-trims message history if over 600KB
- Windows paths in .env are handled by `_get_raw_env_value()` which reads the file directly to avoid escape sequence issues

## Testing Conventions

- All tests are network-isolated (global fixture in `conftest.py` blocks real HTTP)
- Async tests use `pytest-asyncio`
- Check `tests/README.md` to find the right existing `test_*.py` file before creating new ones
- Tests must cover edge cases, error scenarios, and malformed inputs — not just happy path

## Code Style

- English for all code, comments, docstrings, variable names
- Google-style docstrings with Args/Returns/Raises
- Type hints mandatory on all function params and return values
- Logging with loguru (INFO=business logic, DEBUG=technical, ERROR=failures)
- Catch specific exceptions, never bare `except:`

## Git Workflow

- Never commit directly to main — always create a new branch and open a PR via `gh pr create`
- Commits should include tests for any logic changes

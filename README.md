# Routing.Run Backend

OpenRouter-compatible API gateway with multi-provider routing, plan-based access control, and Whop billing.

## Quick Start

```bash
uv sync
cp .env.example .env
# Configure .env with your database URLs and API keys
uv run uvicorn apps.api.main:app --reload
```

## Requirements

- Python 3.12+
- PostgreSQL 16+
- Redis 7+

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE)

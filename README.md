# Routing.Run Backend

OpenRouter-compatible API gateway with multi-provider routing, plan-based access control, and Whop billing integration.

## Features

- **Multi-Provider Routing**: MiniMax, OpenCode, Chutes, z.ai, OpenRouter (backup)
- **Plan-Based Access**: Free, Lite, Premium, Max tiers with different rate limits
- **OpenAI Compatible**: `/v1/chat/completions` API format
- **OAuth**: GitHub login
- **Whop Integration**: Subscription billing and webhook handling
- **Rate Limiting**: Redis-based sliding window rate limiter
- **Circuit Breaker**: Automatic failover between providers

## Quick Start

### Prerequisites

- Python 3.12+
- uv package manager
- PostgreSQL (Railway or self-hosted)
- Redis (Railway or self-hosted)

### Installation

```bash
# Install dependencies
cd routing-backend
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your database URLs and API keys

# Start infrastructure
docker compose -f infra/docker-compose.dev.yml up -d postgres redis

# Run database migrations
uv run alembic upgrade head

# Start API server
uv run uvicorn apps.api.main:app --reload
```

### Production Deployment

```bash
# Build and start with Docker
docker compose -f infra/docker-compose.yml build
docker compose -f infra/docker-compose.yml up -d
```

## Configuration

### Environment Variables

```bash
# Database (Railway PostgreSQL)
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/routing

# Redis (Railway Redis)
REDIS_URL=redis://host:6379/0

# JWT Secret (generate strong random string)
JWT_SECRET_KEY=your-super-secret-key

# Provider API Keys (from configs/provider.yaml)
MINIMAX_API_KEY=sk-cp-...
OPENCODE_API_KEY=sk-...
CHUTES_API_KEY=cpk_...
ZAI_API_KEY=288a...
OPENROUTER_API_KEY=sk-or-v1-...

# OAuth (GitHub Developer Console)
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
```

### Provider Configuration

Edit `configs/provider.yaml` to configure:
- Provider endpoints and API keys
- Model routing chains (primary + fallback providers)
- Rate limits and timeouts
- Circuit breaker settings

### Plan Configuration

Edit `configs/plans.yaml` to configure:
- Rate limits per plan tier
- Allowed models per tier
- Whop product IDs

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create account with email/password |
| POST | `/auth/login` | Login with email/password |
| GET | `/auth/oauth/{provider}` | OAuth redirect (github) |
| GET | `/auth/callback/{provider}` | OAuth callback |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/auth/me` | Get current user |

### Chat Completions (OpenAI Compatible)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Create chat completion |
| GET | `/v1/models` | List available models |

### User Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/user` | Get user profile |
| GET | `/v1/user/usage` | Get usage statistics |
| POST | `/v1/user/keys` | Create API key |
| GET | `/v1/user/keys` | List API keys |
| DELETE | `/v1/user/keys/{id}` | Revoke API key |

### Webhooks

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhooks/whop` | Whop subscription events |

## Available Models

### Lite Models (Free/Lite/Premium/Max)

| Model | Primary Provider | Fallback |
|-------|------------------|----------|
| glm-5 | z.ai | opencode, minimax |
| glm-5-turbo | z.ai | opencode |
| minimax-m2.7 | minimax | opencode, chutes |
| minimax-m2.5 | minimax | opencode |
| minimax-m2.1 | minimax | - |
| kimi-k2.5 | opencode | chutes |
| deepseek-v3.2 | chutes | openrouter |
| qwen3-coder-next | chutes | openrouter |
| qwen3-32b | chutes | openrouter |

### Premium Models (Premium/Max only)

| Model | Primary Provider | Fallback |
|-------|------------------|----------|
| claude-3.5-sonnet | opencode | openrouter |
| gpt-4o | openrouter | - |
| gemini-2.0-flash | openrouter | - |

## Rate Limits

| Plan | Lite Requests/Hour | Premium Requests/Hour |
|------|-------------------|----------------------|
| Free | 20 | 0 |
| Lite | 50 | 0 |
| Premium | 100 | 40 |
| Max | 150 | 60 |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Clients                          │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                      FastAPI API                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │   Auth   │  │   Chat   │  │   Middleware         │  │
│  │          │  │Complete  │  │ (Rate Limit, Auth)  │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    LLM Router                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │Circuit   │  │Provider  │  │ Health   │              │
│  │Breaker   │  │Fallback  │  │ Check    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐       ┌────▼────┐      ┌────▼────┐
│MiniMax│       │ OpenCode │      │ Chutes  │
└───────┘       └─────────┘      └─────────┘
```

## Tech Stack

- **API**: FastAPI 0.110+ / Uvicorn
- **Database**: PostgreSQL 16+ (SQLAlchemy 2.0 async)
- **Cache**: Redis 7+ (ioredis)
- **Workers**: ARQ (Redis-based task queue)
- **Auth**: JWT (python-jose) + Argon2 (passlib)
- **HTTP**: httpx (async, HTTP/2)
- **Container**: Docker / Docker Compose

## Project Structure

```
routing-backend/
├── apps/
│   ├── api/                 # FastAPI application
│   │   ├── api/v1/         # API endpoints
│   │   ├── core/           # Config, security, middleware
│   │   └── services/       # LLM, auth, usage services
│   └── worker/             # ARQ background tasks
├── packages/
│   ├── db/                 # SQLAlchemy models, session
│   ├── redis/              # Redis client, rate limiter
│   └── shared/             # Types, constants, exceptions
├── configs/
│   ├── provider.yaml       # Provider routing config
│   ├── plans.yaml          # Plan tier definitions
│   └── settings.yaml       # App settings
├── infra/
│   ├── docker-compose.yml  # Production compose
│   └── Dockerfile.*        # Container definitions
└── scripts/                # Dev/deploy scripts
```

## License

MIT

from typing import Any
from uuid import UUID

from packages.db.models import UsageLog
from packages.db.session import create_session_factory
from packages.redis.client import get_redis


async def process_usage_log(ctx: dict, job_id: str, data: dict[str, Any]):
    session_factory = create_session_factory()
    async with session_factory() as session:
        usage_log = UsageLog(
            id=UUID(data.get("id", job_id)),
            api_key_id=UUID(data["api_key_id"]),
            user_id=UUID(data["user_id"]),
            model=data["model"],
            provider=data["provider"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            latency_ms=data.get("latency_ms", 0),
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            request_id=data.get("request_id", job_id),
        )
        session.add(usage_log)
        await session.commit()


async def sync_user_subscription(ctx: dict, user_id: str):
    pass


async def cleanup_old_sessions(ctx: dict):
    pass


class AppSettings:
    def __init__(self):
        import os

        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/routing"
        )


async def startup(ctx: dict):
    import os

    settings = AppSettings()
    ctx["settings"] = settings
    os.environ["REDIS_URL"] = settings.redis_url
    ctx["redis"] = await get_redis()


async def shutdown(ctx: dict):
    pass


async def process_usage_log_job(ctx: dict, job_id: str, data: dict[str, Any]):
    session_factory = create_session_factory()
    async with session_factory() as session:
        usage_log = UsageLog(
            id=UUID(data.get("id", job_id)),
            api_key_id=UUID(data["api_key_id"]),
            user_id=UUID(data["user_id"]),
            model=data["model"],
            provider=data["provider"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            latency_ms=data.get("latency_ms", 0),
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            request_id=data.get("request_id", job_id),
        )
        session.add(usage_log)
        await session.commit()

    return {"status": "completed"}


WORKER_FUNCTIONS = [
    process_usage_log_job,
]

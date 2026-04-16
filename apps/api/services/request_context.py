import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from apps.api.core.security import hash_api_key
from apps.api.services.usage import UsageTracker
from packages.db.models import ApiKey, UsageLog
from packages.db.session import get_db_session
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")


def mark_timing(request: Request, stage: str, started_at: float) -> None:
    timings = getattr(request.state, "timings", None)
    if timings is None:
        timings = {}
        request.state.timings = timings
    timings[stage] = int((time.perf_counter() - started_at) * 1000)


def summarize_timings(request: Request) -> str:
    timings = getattr(request.state, "timings", {})
    if not timings:
        return ""
    return " | ".join(f"{stage}={duration}ms" for stage, duration in timings.items())


async def resolve_authenticated_user(request: Request, redis: Redis) -> tuple[str, str, str]:
    if hasattr(request.state, "user_id") and request.state.user_id:
        return request.state.user_id, getattr(request.state, "plan", "free"), ""

    api_key = getattr(request.state, "api_key", "") or request.headers.get("X-API-Key", "")
    if not api_key:
        raise AuthenticationError("Authentication required")

    key_hash = hash_api_key(api_key)
    cache_key = f"auth:{key_hash}"
    cached = await redis.get(cache_key)
    if cached:
        parts = cached.split(":")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]

    async with get_db_session() as session:
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.key_hash == key_hash, ApiKey.is_active)
            .options(joinedload(ApiKey.user))
        )
        api_key_obj = result.scalar_one_or_none()

        if not api_key_obj:
            raise AuthenticationError("Invalid API key")

        user = api_key_obj.user
        if user and user.upgraded_to_tier and user.upgraded_until:
            if user.upgraded_until > datetime.now(UTC):
                current_plan = user.upgraded_to_tier.value
            else:
                current_plan = user.plan_tier.value
        else:
            current_plan = user.plan_tier.value if user else api_key_obj.plan_tier.value

        cache_value = f"{api_key_obj.user_id}:{current_plan}:{api_key_obj.id}"
        await redis.set(cache_key, cache_value, ex=300)
        return str(api_key_obj.user_id), current_plan, str(api_key_obj.id)


async def persist_usage_log(
    *,
    user_id: str,
    api_key_id: str,
    model: str,
    provider: str,
    response_model: str | None,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    request_id: str,
    extra_metadata: dict[str, Any],
) -> None:
    import hashlib

    request_hash = hashlib.sha256(f"{user_id}:{request_id}".encode()).hexdigest()[:16]
    async with get_db_session() as session:
        usage_log = UsageLog(
            id=uuid4(),
            api_key_id=UUID(api_key_id) if api_key_id else None,
            user_id=UUID(user_id),
            model=model,
            provider=provider,
            prompt=None,
            response=None,
            response_model=response_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,
            latency_ms=latency_ms,
            status="success",
            request_id=request_id,
            request_hash=request_hash,
            extra_metadata=json.dumps(extra_metadata),
        )
        session.add(usage_log)
        await session.commit()


def schedule_usage_tracking(
    *,
    redis: Redis,
    user_id: str,
    api_key_id: str,
    model: str,
    provider: str,
    response_model: str | None,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    request_id: str,
    extra_metadata: dict[str, Any],
) -> None:
    async def _run() -> None:
        try:
            usage_tracker = UsageTracker(redis)
            await usage_tracker.track_request(
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            await persist_usage_log(
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                response_model=response_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                request_id=request_id,
                extra_metadata=extra_metadata,
            )
        except Exception as exc:
            logger.error(f"Async usage tracking failed: {exc} | component=usage")

    asyncio.create_task(_run())

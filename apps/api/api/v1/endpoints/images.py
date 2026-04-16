import asyncio
import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access
from apps.api.services.llm.providers import get_provider_for_model
from apps.api.services.request_context import (
    mark_timing,
    resolve_authenticated_user,
    schedule_usage_tracking,
    summarize_timings,
)
from apps.api.services.usage.tracker import UsageTracker
from apps.api.services.usage.request_manager import RequestManager
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")
router = APIRouter(prefix="/v1", tags=["images"])


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="route/minimax-image-1")
    prompt: str
    aspect_ratio: str = Field(default="1:1")
    response_format: str = Field(default="base64")
    size: str | None = None


class ImageResult(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    created: int
    data: list[ImageResult]
    provider: str
    model: str
    credits_charged: float | None = None


async def get_user_from_request(request: Request) -> tuple[str, str, str]:
    redis = await get_redis()
    return await resolve_authenticated_user(request, redis)


@router.post("/images/generations", response_model=ImageResponse)
async def generate_image(
    request: Request,
    body: ImageGenerationRequest,
):
    auth_started = time.perf_counter()
    user_id, plan, api_key_id = await get_user_from_request(request)
    mark_timing(request, "auth", auth_started)

    access_started = time.perf_counter()
    await check_model_access(plan, body.model)

    redis = await get_redis()
    mark_timing(request, "model_access", access_started)

    request_manager = RequestManager(redis)
    limit_started = time.perf_counter()
    await request_manager.check_and_increment(user_id, plan)
    mark_timing(request, "request_limit", limit_started)

    provider_config_data = {
        "minimax_image": {"provider_chain": [{"provider": "minimax_image", "model_id": "image-01"}]}
    }

    provider_chain = provider_config_data.get("minimax_image", {}).get("provider_chain", [])

    if not provider_chain:
        logger.warning(f"No provider chain for image model={body.model} | component=images")
        return {"error": "Model not found"}

    provider_entry = provider_chain[0]
    provider_name = provider_entry["provider"]
    model_id = provider_entry.get("model_id", "image-01")

    provider = get_provider_for_model(provider_name, model_id)

    try:
        provider_started = time.perf_counter()
        response = await provider.generate_image(
            model=model_id,
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            response_format=body.response_format,
            size=body.size,
        )
        mark_timing(request, "provider_call", provider_started)

        asyncio.create_task(
            _track_image_usage(
                redis=redis,
                user_id=user_id,
                api_key_id=api_key_id,
                model=body.model,
                provider=provider_name,
            )
        )

        logger.info(
            f"Image generated | user={user_id} | model={body.model} | "
            f"provider={provider_name} | timings={summarize_timings(request)} | component=images"
        )

        return ImageResponse(
            created=int(time.time()),
            data=[ImageResult(**img) for img in response.get("data", [])],
            provider=provider_name,
            model=body.model,
            credits_charged=None,
        )

    except Exception as e:
        logger.error(
            f"Image generation failed | user={user_id} | model={body.model} | error={e} | component=images"
        )
        return {"error": {"message": str(e), "type": "api_error"}}


async def _track_image_usage(
    *,
    redis,
    user_id: str,
    api_key_id: str,
    model: str,
    provider: str,
) -> None:
    try:
        usage_tracker = UsageTracker(redis)
        await usage_tracker.track_image_request(
            user_id=user_id,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            latency_ms=0,
        )
    except Exception as exc:
        logger.error(f"Async image usage tracking failed: {exc} | component=usage")

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access, check_rate_limit
from apps.api.core.security import hash_api_key, verify_access_token
from apps.api.services.llm.providers import get_provider_for_model
from apps.api.services.usage import CreditManager, UsageTracker
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

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
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = verify_access_token(token)
            user_id = payload.get("sub")
            plan = payload.get("plan", "free")
            api_key_id = ""
            return user_id, plan, api_key_id
        except Exception:
            pass

    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        async with get_db_session() as session:
            from sqlalchemy import select

            from packages.db.models import ApiKey

            key_hash = hash_api_key(api_key)
            result = await session.execute(
                select(ApiKey).where(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active,
                )
            )
            api_key_obj = result.scalar_one_or_none()

            if api_key_obj:
                return str(api_key_obj.user_id), api_key_obj.plan_tier.value, str(api_key_obj.id)
            else:
                raise AuthenticationError("Invalid API key")

    raise AuthenticationError("Authentication required")


@router.post("/images/generations", response_model=ImageResponse)
async def generate_image(
    request: Request,
    body: ImageGenerationRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    await check_model_access(plan, body.model)

    redis = await get_redis()
    api_key_header = request.headers.get("X-API-Key", "")
    key_hash = hash_api_key(api_key_header) if api_key_header else "default"
    await check_rate_limit(redis, plan, body.model, key_hash)

    provider_config_data = {
        "minimax_image": {"provider_chain": [{"provider": "minimax_image", "model_id": "image-01"}]}
    }

    from apps.api.core.config import get_provider_config

    provider_chain = provider_config_data.get("minimax_image", {}).get("provider_chain", [])

    if not provider_chain:
        return {"error": "Model not found"}

    provider_entry = provider_chain[0]
    provider_name = provider_entry["provider"]
    model_id = provider_entry.get("model_id", "image-01")

    provider = get_provider_for_model(provider_name, model_id)

    try:
        response = await provider.generate_image(
            model=model_id,
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            response_format=body.response_format,
            size=body.size,
        )

        credit_manager = CreditManager(redis)
        actual_cost = await credit_manager.deduct_credits_for_image(
            user_id=user_id,
            model=body.model,
        )

        usage_tracker = UsageTracker(redis)
        await usage_tracker.track_image_request(
            user_id=user_id,
            api_key_id=api_key_id,
            model=body.model,
            provider=provider_name,
            latency_ms=0,
        )

        return ImageResponse(
            created=int(time.time()),
            data=[ImageResult(**img) for img in response.get("data", [])],
            provider=provider_name,
            model=body.model,
            credits_charged=actual_cost,
        )

    except Exception as e:
        return {"error": {"message": str(e), "type": "api_error"}}

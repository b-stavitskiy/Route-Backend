import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access
from apps.api.core.security import hash_api_key, verify_access_token
from apps.api.services.llm.providers import get_provider_for_model
from apps.api.services.usage import UsageTracker
from apps.api.services.usage.request_manager import RequestManager
from packages.db.session import get_db_session
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
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = await verify_access_token(token)
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
            from sqlalchemy.orm import selectinload

            key_hash = hash_api_key(api_key)
            result = await session.execute(
                select(ApiKey)
                .where(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active,
                )
                .options(selectinload(ApiKey.user))
            )
            api_key_obj = result.scalar_one_or_none()

            if api_key_obj:
                user = api_key_obj.user
                if user and user.upgraded_to_tier and user.upgraded_until:
                    from datetime import UTC, datetime

                    if user.upgraded_until > datetime.now(UTC):
                        plan = user.upgraded_to_tier.value
                    else:
                        plan = user.plan_tier.value
                else:
                    plan = user.plan_tier.value if user else api_key_obj.plan_tier.value
                return str(api_key_obj.user_id), plan, str(api_key_obj.id)
            else:
                raise AuthenticationError("Invalid API key")

    raise AuthenticationError("Authentication required")


@router.post("/images/generations", response_model=ImageResponse)
async def generate_image(
    request: Request,
    body: ImageGenerationRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    logger.info(
        f"Image request | user={user_id} | plan={plan} | model={body.model} | component=images"
    )

    await check_model_access(plan, body.model)

    redis = await get_redis()

    request_manager = RequestManager(redis)
    await request_manager.check_and_increment(user_id, plan)

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

    logger.info(
        f"Calling image provider | provider={provider_name} | model={model_id} | component=images"
    )

    provider = get_provider_for_model(provider_name, model_id)

    try:
        response = await provider.generate_image(
            model=model_id,
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            response_format=body.response_format,
            size=body.size,
        )

        usage_tracker = UsageTracker(redis)
        await usage_tracker.track_image_request(
            user_id=user_id,
            api_key_id=api_key_id,
            model=body.model,
            provider=provider_name,
            latency_ms=0,
        )

        logger.info(
            f"Image generated | user={user_id} | model={body.model} | "
            f"provider={provider_name} | component=images"
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

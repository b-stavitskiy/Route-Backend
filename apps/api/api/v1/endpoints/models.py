from fastapi import APIRouter, Request
from pydantic import BaseModel
from apps.api.core.config import get_provider_config
from apps.api.core.security import verify_access_token
from apps.api.services.llm import LLMRouter
from packages.redis.client import get_redis

router = APIRouter(prefix="/v1", tags=["models"])


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str
    tier: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


async def get_user_plan(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = verify_access_token(token)
            return payload.get("plan", "free")
        except Exception:
            pass

    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        from apps.api.core.security import hash_api_key
        from packages.db.models import ApiKey
        from packages.db.session import get_db_session
        from sqlalchemy import select

        key_hash = hash_api_key(api_key)
        async with get_db_session() as session:
            result = await session.execute(
                select(ApiKey).where(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active,
                )
            )
            api_key_obj = result.scalar_one_or_none()
            if api_key_obj:
                return api_key_obj.plan_tier.value

    return "free"


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
):
    user_plan = await get_user_plan(request)

    redis = await get_redis()
    router_instance = LLMRouter(redis)
    models = await router_instance.list_available_models(user_plan)

    return ModelListResponse(data=[ModelObject(**m) for m in models])


@router.get("/models/{model:path}")
async def get_model(
    request: Request,
    model: str,
):
    user_plan = await get_user_plan(request)

    provider_config = get_provider_config()
    model_config = provider_config.get_model_config(model)

    if not model_config:
        return {"error": "Model not found"}

    allowed = provider_config.is_model_allowed(model, user_plan)

    chain = model_config.get("provider_chain", [])
    primary_provider = chain[0]["provider"] if chain else None

    return {
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": primary_provider,
        "tier": "lite"
        if provider_config._config.get("providers", {}).get("models", {}).get("lite", {}).get(model)
        else "premium",
        "allowed": allowed,
        "providers": chain,
    }

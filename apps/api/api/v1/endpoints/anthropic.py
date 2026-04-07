import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access, check_rate_limit
from apps.api.core.security import hash_api_key, verify_access_token
from apps.api.services.llm import LLMRouter
from apps.api.services.usage import CreditManager, UsageTracker
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")
router = APIRouter(prefix="/v1", tags=["anthropic"])


class AnthropicMessage(BaseModel):
    role: str
    content: str


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str | None = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    system: str | None = None
    stop_sequences: list[str] | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


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


def convert_to_openai_format(messages: list[dict], system: str | None) -> list[dict]:
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        result.append({"role": msg["role"], "content": msg["content"]})
    return result


@router.post("/messages")
async def create_message(
    request: Request,
    body: AnthropicMessagesRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    logger.info(f"Anthropic message request | user={user_id} | plan={plan} | model={body.model}")

    await check_model_access(plan, body.model)

    redis = await get_redis()
    api_key_header = request.headers.get("X-API-Key", "")
    key_hash = hash_api_key(api_key_header) if api_key_header else "default"
    await check_rate_limit(redis, plan, body.model, key_hash)

    credit_manager = CreditManager(redis)
    try:
        estimated_cost = await credit_manager.check_credits_for_request(
            user_id=user_id,
            model=body.model,
            max_tokens=body.max_tokens,
        )
        logger.info(f"Credit check passed | estimated_cost={estimated_cost} | component=credits")
    except Exception as e:
        logger.error(f"Credit check failed: {e} | component=credits")
        raise e

    router_instance = LLMRouter(redis)

    messages = []
    for m in body.messages:
        msg = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)
    messages = convert_to_openai_format(messages, body.system)

    logger.info(f"Routing Anthropic request to model | model={body.model} | component=router")

    response = await router_instance.route_chat_complete(
        model=body.model,
        messages=messages,
        user_plan=plan,
        temperature=body.temperature or 0.7,
        max_tokens=body.max_tokens,
        stream=False,
        top_p=body.top_p,
    )

    usage_tracker = UsageTracker(redis)
    input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)
    await usage_tracker.track_request(
        user_id=user_id,
        api_key_id=api_key_id,
        model=body.model,
        provider=response.get("provider", "unknown"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=response.get("latency_ms", 0),
    )

    actual_cost = await credit_manager.deduct_credits(
        user_id=user_id,
        model=body.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    logger.info(
        f"Anthropic request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"cost={actual_cost}"
    )

    return {
        "id": f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ).model_dump(),
        "credits_charged": actual_cost,
    }

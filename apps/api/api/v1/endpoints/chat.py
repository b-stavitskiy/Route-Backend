import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access, check_rate_limit
from apps.api.core.security import hash_api_key, verify_access_token
from apps.api.services.llm import LLMRouter
from apps.api.services.usage import CreditManager, UsageTracker
from packages.db.models import UsageLog
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")

router = APIRouter(prefix="/v1", tags=["chat"])


class MessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class Message(BaseModel):
    role: str
    content: str | list[MessageContent]


class Tool(BaseModel):
    type: str = "function"
    function: dict


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


async def get_user_from_request(request: Request) -> tuple[str, str, str]:
    api_key = request.headers.get("X-API-Key", "")

    if hasattr(request.state, "api_key") and request.state.api_key:
        api_key = request.state.api_key

    import logging

    logger = logging.getLogger("routing.run.api")
    logger.info(f"get_user_from_request - api_key: {api_key[:30] if api_key else None}")

    if api_key:
        async with get_db_session() as session:
            from sqlalchemy import select

            from packages.db.models import ApiKey, User

            key_hash = hash_api_key(api_key)
            logger.info(f"get_user_from_request - key_hash: {key_hash}")
            result = await session.execute(
                select(ApiKey).where(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active,
                )
            )
            api_key_obj = result.scalar_one_or_none()
            logger.info(f"get_user_from_request - api_key_obj found: {api_key_obj is not None}")

            if api_key_obj:
                user_result = await session.execute(
                    select(User).where(User.id == api_key_obj.user_id)
                )
                user = user_result.scalar_one_or_none()
                current_plan = user.plan_tier.value if user else api_key_obj.plan_tier.value
                return str(api_key_obj.user_id), current_plan, str(api_key_obj.id)
            else:
                raise AuthenticationError("Invalid API key")

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

    raise AuthenticationError("Authentication required")


async def stream_generator(
    router_instance: LLMRouter,
    credit_manager: CreditManager,
    usage_tracker: UsageTracker,
    model: str,
    messages: list[dict[str, Any]],
    user_plan: str,
    user_id: str,
    api_key_id: str,
    temperature: float,
    max_tokens: int | None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    **kwargs,
) -> AsyncGenerator[bytes, None]:
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    input_tokens = 0
    output_tokens = 0
    provider = "unknown"
    latency_ms = 0
    start_time = time.time()

    try:
        async for chunk in router_instance.route_chat_complete_stream(
            model=model,
            messages=messages,
            user_plan=user_plan,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        ):
            if not chunk or not isinstance(chunk, dict):
                continue
            data = chunk.get("data", {})
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            if isinstance(data, dict):
                if data.get("event") == "error":
                    error_msg = data.get("data", "Unknown error")
                    error_payload = json.dumps(
                        {"error": {"message": error_msg, "type": "api_error"}}
                    )
                    yield f"data: {error_payload}\n\n".encode()
                    break

                if "usage" in data:
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                if "provider" in data:
                    provider = data.get("provider", "unknown")

                delta = data.get("delta", data.get("content", ""))
                if delta:
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()
            elif isinstance(data, str) and data.startswith("{"):
                yield f"data: {data}\n\n".encode()

        yield b"data: [DONE]\n\n"
        latency_ms = int((time.time() - start_time) * 1000)

    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "api_error",
                "code": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n".encode()

    total_tokens = input_tokens + output_tokens
    if total_tokens > 0:
        try:
            actual_cost = await credit_manager.deduct_credits(
                user_id=user_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            await usage_tracker.track_request(
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

            from uuid import UUID, uuid4
            import hashlib

            request_hash = hashlib.sha256(f"{user_id}:{request_id}".encode()).hexdigest()[:16]

            async with get_db_session() as session:
                usage_log = UsageLog(
                    id=uuid4(),
                    api_key_id=UUID(api_key_id) if api_key_id and api_key_id != "" else None,
                    user_id=UUID(user_id),
                    model=model,
                    provider=provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=actual_cost,
                    latency_ms=latency_ms,
                    status="success",
                    request_id=request_id,
                    request_hash=request_hash,
                )
                session.add(usage_log)
                await session.commit()

            logger.info(
                f"Stream completed | model={model} | provider={provider} | "
                f"input_tokens={input_tokens} | output_tokens={output_tokens} | "
                f"latency_ms={latency_ms} | cost={actual_cost} | component=chat"
            )
        except Exception as e:
            logger.error(f"Failed to track stream usage: {e} | component=chat")


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    logger.info(
        f"Chat request | user={user_id} | plan={plan} | model={body.model} | stream={body.stream} | component=chat"
    )

    logger.info(f"User authenticated | user_id={user_id} | plan={plan} | component=auth")

    await check_model_access(plan, body.model)
    logger.info(f"Model access granted | model={body.model} | plan={plan} | component=auth")

    redis = await get_redis()
    api_key_header = request.headers.get("X-API-Key", "")
    key_hash = hash_api_key(api_key_header) if api_key_header else "default"
    await check_rate_limit(redis, plan, body.model, key_hash)
    logger.info(f"Rate limit check passed | model={body.model} | component=ratelimit")

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

    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    logger.info(
        f"Routing request to model | model={body.model} | messages_count={len(messages)} | component=router"
    )

    if body.stream:
        usage_tracker = UsageTracker(redis)
        return StreamingResponse(
            stream_generator(
                router_instance=router_instance,
                credit_manager=credit_manager,
                usage_tracker=usage_tracker,
                model=body.model,
                messages=messages,
                user_plan=plan,
                user_id=user_id,
                api_key_id=api_key_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                top_p=body.top_p,
                frequency_penalty=body.frequency_penalty,
                presence_penalty=body.presence_penalty,
                stop=body.stop,
                tools=[t.model_dump() for t in body.tools] if body.tools else None,
                tool_choice=body.tool_choice,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": f"chatcmpl-{int(time.time() * 1000)}",
            },
        )

    response = await router_instance.route_chat_complete(
        model=body.model,
        messages=messages,
        user_plan=plan,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        stream=False,
        top_p=body.top_p,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        stop=body.stop,
        tools=[t.model_dump() for t in body.tools] if body.tools else None,
        tool_choice=body.tool_choice,
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

    from uuid import UUID, uuid4
    import hashlib

    request_id = response.get("id", str(uuid4()))
    request_hash = hashlib.sha256(f"{user_id}:{request_id}".encode()).hexdigest()[:16]

    async with get_db_session() as session:
        usage_log = UsageLog(
            id=uuid4(),
            api_key_id=UUID(api_key_id) if api_key_id and api_key_id != "" else None,
            user_id=UUID(user_id),
            model=body.model,
            provider=response.get("provider", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=actual_cost,
            latency_ms=response.get("latency_ms", 0),
            status="success",
            request_id=request_id,
            request_hash=request_hash,
        )
        session.add(usage_log)
        await session.commit()

    response["credits_charged"] = actual_cost

    logger.info(
        f"Request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"cost={actual_cost} | component=chat"
    )
    logger.info(f"Response body: {response}")

    return response

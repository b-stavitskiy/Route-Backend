import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from apps.api.core.plans import get_user_effective_plan_name
from apps.api.core.rate_limiter import check_model_access
from apps.api.core.security import get_access_token_from_request, hash_api_key, verify_access_token
from apps.api.services.llm import LLMRouter
from apps.api.services.llm.anthropic_adapter import (
    AnthropicStreamState,
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    format_anthropic_sse,
    openai_response_to_anthropic,
    openai_stream_chunk_to_anthropic_events,
)
from apps.api.services.llm.router import get_model_token_budget, truncate_messages
from apps.api.services.usage import CreditManager, UsageTracker
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")
router = APIRouter(prefix="/v1", tags=["anthropic"])


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None


class AnthropicContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str | None = None


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    system: str | list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


async def get_user_from_request(request: Request) -> tuple[str, str, str]:
    api_key = request.headers.get("X-API-Key", "")
    if hasattr(request.state, "api_key") and request.state.api_key:
        api_key = request.state.api_key

    if api_key:
        async with get_db_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            from packages.db.models import ApiKey

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
                plan = get_user_effective_plan_name(user) if user else api_key_obj.plan_tier.value
                return str(api_key_obj.user_id), plan, str(api_key_obj.id)
            raise AuthenticationError("Invalid API key")

    token = get_access_token_from_request(request)

    if token:
        try:
            payload = await verify_access_token(token)
            user_id = payload.get("sub")
            plan = payload.get("plan", "free")
            api_key_id = ""
            return user_id, plan, api_key_id
        except Exception:
            pass

    raise AuthenticationError("Authentication required")


def extract_text_from_content(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)
    return str(content)


def convert_to_openai_format(messages: list[dict], system: str | list[dict] | None) -> list[dict]:
    return anthropic_messages_to_openai(messages, system)


def _anthropic_error_event(message: str) -> bytes:
    return format_anthropic_sse(
        "error",
        {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        },
    )


async def anthropic_stream_generator(
    router_instance: LLMRouter,
    usage_tracker: UsageTracker,
    credit_manager: CreditManager,
    model: str,
    messages: list[dict[str, Any]],
    user_plan: str,
    user_id: str,
    api_key_id: str,
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    **kwargs,
):
    state = AnthropicStreamState(model=model)
    provider = "unknown"
    start_time = time.time()
    latency_ms = 0

    yield format_anthropic_sse(*state.message_start_event())

    try:
        async for chunk in router_instance.route_chat_complete_stream(
            model=model,
            messages=messages,
            user_plan=user_plan,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            user_id=user_id,
            stream_request_id=state.msg_id,
            **kwargs,
        ):
            if not isinstance(chunk, dict):
                continue
            if chunk.get("event") == "error":
                yield _anthropic_error_event(str(chunk.get("data", "Unknown provider error")))
                return
            provider = chunk.get("provider") or provider
            data = chunk.get("data") or {}
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, dict):
                continue
            if data.get("event") == "error":
                yield _anthropic_error_event(str(data.get("data", "Unknown provider error")))
                return

            for event in openai_stream_chunk_to_anthropic_events(data, state):
                yield format_anthropic_sse(*event)

        for event in state.final_events():
            yield format_anthropic_sse(*event)
        latency_ms = int((time.time() - start_time) * 1000)

    except Exception as e:
        yield _anthropic_error_event(str(e))
        return

    input_tokens = state.usage.get("input_tokens", 0)
    output_tokens = state.usage.get("output_tokens", 0)

    if input_tokens or output_tokens:
        try:
            await usage_tracker.track_request(
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            await credit_manager.deduct_credits(
                user_id=user_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to track Anthropic stream usage: {e} | component=anthropic")


@router.post("/messages")
async def create_message(
    request: Request,
    body: AnthropicMessagesRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    logger.info(f"Anthropic message request | user={user_id} | plan={plan} | model={body.model}")

    await check_model_access(plan, body.model)

    redis = await get_redis()
    router_instance = LLMRouter(redis)

    messages = [m.model_dump(exclude_none=True) for m in body.messages]
    messages = anthropic_messages_to_openai(messages, body.system)

    model_config = router_instance.provider_config.get_model_config(body.model, plan)
    context_size, output_tokens_estimate, available_for_input = get_model_token_budget(
        model_config,
        body.max_tokens,
    )
    provider_max_tokens = output_tokens_estimate

    original_msg_count = len(messages)
    messages = truncate_messages(messages, max_tokens=available_for_input)
    if len(messages) < original_msg_count:
        logger.info(
            f"Truncated Anthropic messages from {original_msg_count} to {len(messages)} "
            f"(context_size={context_size}, output_tokens_estimate={output_tokens_estimate}, "
            f"available_for_input={available_for_input}) | component=anthropic"
        )

    credit_manager = CreditManager(redis)
    try:
        estimated_cost = await credit_manager.check_credits_for_request(
            user_id=user_id,
            model=body.model,
            max_tokens=provider_max_tokens,
        )
        logger.info(f"Credit check passed | estimated_cost={estimated_cost} | component=credits")
    except Exception as e:
        logger.error(f"Credit check failed: {e} | component=credits")
        raise e

    tools = anthropic_tools_to_openai(body.tools)
    tool_choice = anthropic_tool_choice_to_openai(body.tool_choice)

    logger.info(f"Routing Anthropic request to model | model={body.model} | component=router")

    if body.stream:
        usage_tracker = UsageTracker(redis)
        return StreamingResponse(
            anthropic_stream_generator(
                router_instance=router_instance,
                usage_tracker=usage_tracker,
                credit_manager=credit_manager,
                model=body.model,
                messages=messages,
                user_plan=plan,
                user_id=user_id,
                api_key_id=api_key_id,
                temperature=body.temperature if body.temperature is not None else 0.7,
                max_tokens=provider_max_tokens,
                top_p=body.top_p,
                stop=body.stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                thinking=body.thinking,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": f"msg_{int(time.time() * 1000)}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    response = await router_instance.route_chat_complete(
        model=body.model,
        messages=messages,
        user_plan=plan,
        temperature=body.temperature if body.temperature is not None else 0.7,
        max_tokens=provider_max_tokens,
        stream=False,
        top_p=body.top_p,
        stop=body.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        thinking=body.thinking,
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

    anthropic_response = openai_response_to_anthropic(response, body.model)

    logger.info(
        f"Anthropic request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"cost={actual_cost}"
    )

    return anthropic_response

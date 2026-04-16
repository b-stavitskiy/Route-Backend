import asyncio
import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access
from apps.api.services.llm import LLMRouter
from apps.api.services.request_context import (
    mark_timing,
    resolve_authenticated_user,
    schedule_usage_tracking,
    summarize_timings,
)
from apps.api.services.usage import CreditManager
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")
router = APIRouter(prefix="/v1", tags=["anthropic"])


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]
    tool_call_id: str | None = None


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str | None = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    system: str | list[dict] | None = None
    stop_sequences: list[str] | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


async def get_user_from_request(request: Request) -> tuple[str, str, str]:
    redis = await get_redis()
    return await resolve_authenticated_user(request, redis)


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
    result = []
    if system:
        system_text = extract_text_from_content(system)
        result.append({"role": "system", "content": system_text})
    for msg in messages:
        content = msg.get("content", "")
        content_text = extract_text_from_content(content)
        result.append({"role": msg["role"], "content": content_text})
    return result


@router.post("/messages")
async def create_message(
    request: Request,
    body: AnthropicMessagesRequest,
):
    auth_started = time.perf_counter()
    user_id, plan, api_key_id = await get_user_from_request(request)
    mark_timing(request, "auth", auth_started)

    access_started = time.perf_counter()
    await check_model_access(plan, body.model)

    redis = await get_redis()
    mark_timing(request, "model_access", access_started)

    credit_manager = CreditManager(redis)
    credit_started = time.perf_counter()
    try:
        estimated_cost = await credit_manager.check_credits_for_request(
            user_id=user_id,
            model=body.model,
            max_tokens=body.max_tokens,
        )
    except Exception as e:
        logger.error(f"Credit check failed: {e} | component=credits")
        raise e
    mark_timing(request, "credit_check", credit_started)

    request_manager_start = time.perf_counter()
    from apps.api.services.usage.request_manager import RequestManager

    request_manager = RequestManager(redis)
    await request_manager.check_and_increment(user_id, plan)
    mark_timing(request, "request_limit", request_manager_start)

    router_instance = LLMRouter(redis)

    messages = []
    for m in body.messages:
        msg = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)
    messages = convert_to_openai_format(messages, body.system)

    provider_started = time.perf_counter()
    response = await router_instance.route_chat_complete(
        model=body.model,
        messages=messages,
        user_plan=plan,
        temperature=body.temperature or 0.7,
        max_tokens=body.max_tokens,
        stream=False,
        top_p=body.top_p,
    )
    mark_timing(request, "provider_call", provider_started)

    input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)

    actual_cost = await credit_manager.deduct_credits(
        user_id=user_id,
        model=body.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    request_id = response.get("id", f"msg_{int(time.time() * 1000)}")
    schedule_usage_tracking(
        redis=redis,
        user_id=user_id,
        api_key_id=api_key_id,
        model=body.model,
        provider=response.get("provider", "unknown"),
        response_model=response.get("model"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=response.get("latency_ms", 0),
        request_id=request_id,
        extra_metadata={
            "temperature": body.temperature,
            "max_tokens": body.max_tokens,
            "top_p": body.top_p,
        },
    )

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    logger.info(
        f"Anthropic request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"cost={actual_cost} | timings={summarize_timings(request)}"
    )

    return {
        "id": request_id,
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

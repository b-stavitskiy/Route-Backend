import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.api.core.rate_limiter import check_model_access
from apps.api.services.llm import LLMRouter
from apps.api.services.llm.router import truncate_messages
from apps.api.services.request_context import (
    mark_timing,
    resolve_authenticated_user,
    schedule_usage_tracking,
    summarize_timings,
)
from apps.api.services.llm.transforms import map_finish_reason, store_streaming_tool_calls
from apps.api.services.usage.request_manager import RequestManager
from packages.redis.client import get_redis

logger = logging.getLogger("routing.run.api")

router = APIRouter(prefix="/v1", tags=["chat"])


class MessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class Message(BaseModel):
    role: str
    content: str | list[MessageContent]
    tool_call_id: str | None = None


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
    redis = await get_redis()
    return await resolve_authenticated_user(request, redis)


async def stream_generator(
    router_instance: LLMRouter,
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
    streaming_tool_calls: list[dict[str, Any]] = []
    redis = await get_redis()

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
            stream_request_id=request_id,
            **kwargs,
        ):
            if not chunk:
                continue
            if isinstance(chunk, str):
                yield f"data: {chunk}\n\n".encode()
                continue
            if not isinstance(chunk, dict):
                continue
            data = chunk.get("data") or {}
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    yield f"data: {data}\n\n".encode()
                    continue

            if not isinstance(data, dict):
                continue

            if data.get("event") == "error":
                error_msg = data.get("data", "Unknown error")
                error_payload = json.dumps({"error": {"message": error_msg, "type": "api_error"}})
                yield f"data: {error_payload}\n\n".encode()
                break

            if "usage" in data:
                usage = data.get("usage") or {}
                input_tokens = usage.get("prompt_tokens") or 0
                output_tokens = usage.get("completion_tokens") or 0

            if "provider" in data:
                provider = data.get("provider") or "unknown"

            delta = None
            finish_reason = None
            tool_calls = None
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "delta" in choice:
                    delta = choice["delta"].get("content", "")
                    tool_calls = choice["delta"].get("tool_calls")
                elif "message" in choice:
                    delta = choice["message"].get("content", "")
                    tool_calls = choice["message"].get("tool_calls")
                finish_reason = map_finish_reason(choice.get("finish_reason"))

            if tool_calls is not None:
                for tc in tool_calls:
                    tc_copy = {
                        "id": tc.get("id"),
                        "function": tc.get("function"),
                        "index": tc.get("index"),
                    }
                    streaming_tool_calls.append(tc_copy)
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()
            elif delta is not None:
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()

        if streaming_tool_calls:
            await store_streaming_tool_calls(redis, user_id, request_id, streaming_tool_calls)

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
            schedule_usage_tracking(
                redis=redis,
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                response_model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                request_id=request_id,
                extra_metadata={"temperature": temperature, "max_tokens": max_tokens},
            )

            logger.info(
                f"Stream completed | model={model} | provider={provider} | "
                f"input_tokens={input_tokens} | output_tokens={output_tokens} | "
                f"latency_ms={latency_ms} | component=chat"
            )
        except Exception as e:
            logger.error(f"Failed to track stream usage: {e} | component=chat")
    else:
        logger.warning(
            f"Stream completed with zero tokens (usage not tracked) | model={model} | "
            f"provider={provider} | chunks_yielded_estimate=unknown | latency_ms={latency_ms} | "
            f"component=chat"
        )


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
):
    auth_started = time.perf_counter()
    user_id, plan, api_key_id = await get_user_from_request(request)
    mark_timing(request, "auth", auth_started)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f"Chat request | user={user_id} | plan={plan} | model={body.model} | "
            f"stream={body.stream} | component=chat"
        )

    access_started = time.perf_counter()
    await check_model_access(plan, body.model)
    redis = await get_redis()
    mark_timing(request, "model_access", access_started)

    request_manager = RequestManager(redis)
    limit_started = time.perf_counter()
    await request_manager.check_and_increment(user_id, plan)
    mark_timing(request, "request_limit", limit_started)

    router_instance = LLMRouter(redis)

    prep_started = time.perf_counter()
    messages = []
    for m in body.messages:
        content = m.content
        if isinstance(content, list):
            content = [c.model_dump() if hasattr(c, "model_dump") else c for c in content]
        msg = {"role": m.role, "content": content}
        if m.tool_call_id is not None:
            if m.role == "tool" and not m.tool_call_id:
                raise HTTPException(
                    status_code=400, detail="tool_call_id cannot be empty for tool results"
                )
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)

    tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
    if tool_result_msgs:
        for msg in tool_result_msgs:
            logger.info(
                f"Tool result received: tool_call_id={msg.get('tool_call_id')} | "
                f"content_preview={str(msg.get('content', ''))[:100]} | component=chat"
            )

    logger.info(
        f"Routing request to model | model={body.model} | "
        f"messages_count={len(messages)} | component=router"
    )

    model_config = router_instance.provider_config.get_model_config(body.model, plan)
    context_size = model_config.get("context_size", 80000) if model_config else 80000
    output_tokens_estimate = body.max_tokens or 4096
    available_for_input = max(1000, context_size - output_tokens_estimate - 5000)

    original_msg_count = len(messages)
    messages = truncate_messages(messages, max_messages=50, max_tokens=available_for_input)
    if len(messages) < original_msg_count:
        logger.info(
            f"Truncated messages from {original_msg_count} to {len(messages)} "
            f"(context_size={context_size}, output_tokens_estimate={output_tokens_estimate}, available_for_input={available_for_input}) | component=router"
        )
    mark_timing(request, "request_prep", prep_started)

    max_tokens = body.max_tokens
    if max_tokens is None or max_tokens > 32768:
        max_tokens = 32768

    if body.stream:
        return StreamingResponse(
            stream_generator(
                router_instance=router_instance,
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
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    provider_started = time.perf_counter()
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
    mark_timing(request, "provider_call", provider_started)

    input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)
    request_id = response.get("id", f"chatcmpl-{int(time.time() * 1000)}")
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
            "frequency_penalty": body.frequency_penalty,
            "presence_penalty": body.presence_penalty,
        },
    )

    logger.info(
        f"Request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"timings={summarize_timings(request)} | component=chat"
    )

    return response

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis

from apps.api.core.config import get_provider_config
from apps.api.services.llm.base import (
    AnthropicCompatProvider,
    OpenAICompatProvider,
)
from apps.api.services.llm.providers import get_provider_for_model
from packages.redis.client import RedisCache
from packages.shared.exceptions import (
    InvalidModelError,
    ProviderError,
    ProviderTimeoutError,
)

logger = logging.getLogger("routing.run.router")


def sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if isinstance(content, str) and (
                        "\n" in content or "\r" in content or "\t" in content
                    ):
                        choice["message"]["content"] = (
                            content.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
                        )
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if isinstance(content, str) and (
                        "\n" in content or "\r" in content or "\t" in content
                    ):
                        choice["delta"]["content"] = (
                            content.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
                        )
    except Exception:
        pass
    return response


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        half_open_requests: int = 1,
        reset_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.half_open_requests = half_open_requests
        self.reset_timeout = reset_timeout

        self._failure_count: dict[str, int] = {}
        self._circuit_open: dict[str, bool] = {}
        self._last_failure_time: dict[str, float] = {}
        self._half_open_count: dict[str, int] = {}

    def is_open(self, provider: str) -> bool:
        if self._circuit_open.get(provider, False):
            if time.time() - self._last_failure_time.get(provider, 0) > self.reset_timeout:
                self._half_open_count[provider] = 0
                self._circuit_open[provider] = False
                return False
            return True
        return False

    def record_failure(self, provider: str):
        self._failure_count[provider] = self._failure_count.get(provider, 0) + 1
        self._last_failure_time[provider] = time.time()

        if self._failure_count[provider] >= self.failure_threshold:
            self._circuit_open[provider] = True

    def record_success(self, provider: str):
        self._failure_count[provider] = 0
        self._circuit_open[provider] = False
        self._half_open_count[provider] = 0

    def can_attempt(self, provider: str) -> bool:
        if self.is_open(provider):
            if self._half_open_count.get(provider, 0) < self.half_open_requests:
                self._half_open_count[provider] = self._half_open_count.get(provider, 0) + 1
                return True
            return False
        return True


class LLMRouter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)
        self.provider_config = get_provider_config()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            half_open_requests=1,
            reset_timeout=60,
        )
        self._provider_clients: dict[str, OpenAICompatProvider | AnthropicCompatProvider] = {}

    async def get_provider_client(
        self,
        provider_name: str,
        model_id: str,
    ) -> OpenAICompatProvider | AnthropicCompatProvider:
        key = f"{provider_name}:{model_id}"
        if key not in self._provider_clients:
            self._provider_clients[key] = get_provider_for_model(provider_name, model_id)
        return self._provider_clients[key]

    async def route_chat_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        user_plan: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        request_id = f"chatcmpl-{uuid4().hex[:8]}"

        provider_chain = self.provider_config.get_provider_chain(model, user_plan)
        if not provider_chain:
            logger.warning(f"No provider chain found for model={model} plan={user_plan}")
            raise InvalidModelError(model)

        routing_config = self.provider_config.get_routing_config()
        retry_count = routing_config.get("retry_count", 2)

        has_tool_results = any(m.get("role") == "tool" for m in messages)
        if has_tool_results:
            logger.info(
                f"Tool results detected in request, routing to first provider only | component=router"
            )
            provider_chain = [provider_chain[0]] if provider_chain else []
            retry_count = 0

        logger.info(
            f"Routing chat request | model={model} | plan={user_plan} | request_id={request_id} | "
            f"providers={len(provider_chain)} | max_retries={retry_count}",
        )

        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            for provider_entry in provider_chain:
                provider_name = provider_entry["provider"]
                model_id = provider_entry.get("model_id", model)

                if self.circuit_breaker.is_open(provider_name):
                    logger.debug(
                        f"Circuit breaker open, skipping provider={provider_name}",
                    )
                    continue

                try:
                    client = await self.get_provider_client(provider_name, model_id)

                    logger.info(
                        f"Calling provider | provider={provider_name} | "
                        f"model={model_id} | attempt={attempt + 1}",
                    )

                    start_time = time.time()
                    response = await client.chat_complete(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        **kwargs,
                    )
                    latency_ms = int((time.time() - start_time) * 1000)

                    self.circuit_breaker.record_success(provider_name)
                    await self._update_provider_health(provider_name, latency_ms, False)

                    response["id"] = response.get("id", request_id)
                    response["provider"] = provider_name
                    response["model"] = model
                    response["latency_ms"] = latency_ms

                    logger.info(
                        f"Provider response success | provider={provider_name} | "
                        f"model={model_id} | latency_ms={latency_ms}",
                    )

                    return sanitize_response(response)

                except ProviderTimeoutError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Provider timeout | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

                except ProviderError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Provider error | provider={provider_name} | model={model_id} | error={e}",
                    )
                    continue

                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    logger.error(
                        f"Unexpected error | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

            if attempt < retry_count:
                delay_ms = routing_config.get("retry_delay_ms", 200)
                logger.info(
                    f"Retrying request | attempt={attempt + 1} | delay_ms={delay_ms}",
                )
                await asyncio.sleep(delay_ms / 1000)

        logger.error(
            f"All providers failed | model={model} | last_error={last_error}",
        )

        if last_error:
            raise last_error

        raise ProviderError(
            message="No providers available for this model",
            provider="all",
        )

    async def route_chat_complete_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        user_plan: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = f"chatcmpl-{uuid4().hex[:8]}"

        provider_chain = self.provider_config.get_provider_chain(model, user_plan)
        if not provider_chain:
            logger.warning(f"No provider chain found for model={model} plan={user_plan}")
            raise InvalidModelError(model)

        routing_config = self.provider_config.get_routing_config()
        retry_count = routing_config.get("retry_count", 2)

        has_tool_results = any(m.get("role") == "tool" for m in messages)
        if has_tool_results:
            logger.info(
                f"Tool results detected in request, routing to first provider only | component=router"
            )
            provider_chain = [provider_chain[0]] if provider_chain else []
            retry_count = 0

        logger.info(
            f"Routing stream request | model={model} | plan={user_plan} | "
            f"request_id={request_id} | providers={len(provider_chain)} | "
            f"max_retries={retry_count}",
        )

        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            for provider_entry in provider_chain:
                provider_name = provider_entry["provider"]
                model_id = provider_entry.get("model_id", model)

                if self.circuit_breaker.is_open(provider_name):
                    logger.debug(
                        f"Circuit breaker open, skipping provider={provider_name}",
                    )
                    continue

                try:
                    client = await self.get_provider_client(provider_name, model_id)

                    logger.info(
                        f"Calling stream provider | provider={provider_name} | "
                        f"model={model_id} | attempt={attempt + 1}",
                    )

                    start_time = time.time()
                    stream_gen = client.chat_complete_stream(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    first_chunk = True
                    async for chunk in stream_gen:
                        if first_chunk:
                            latency_ms = int((time.time() - start_time) * 1000)
                            self.circuit_breaker.record_success(provider_name)
                            await self._update_provider_health(provider_name, latency_ms, False)
                            first_chunk = False
                            logger.info(
                                f"Stream started | provider={provider_name} | "
                                f"model={model_id} | first_chunk_latency_ms={latency_ms}",
                            )

                        chunk["request_id"] = request_id
                        chunk["provider"] = provider_name
                        yield chunk

                    logger.info(
                        f"Stream completed | provider={provider_name} | model={model_id}",
                    )
                    return

                except ProviderTimeoutError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Stream provider timeout | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

                except ProviderError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Stream provider error | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    logger.error(
                        f"Stream unexpected error | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

            if attempt < retry_count:
                delay_ms = routing_config.get("retry_delay_ms", 200)
                logger.info(
                    f"Retrying stream request | attempt={attempt + 1} | delay_ms={delay_ms}",
                )
                await asyncio.sleep(delay_ms / 1000)

        logger.error(
            f"All stream providers failed | model={model} | last_error={last_error}",
        )

        if last_error:
            raise last_error

        raise ProviderError(
            message="No providers available for this model",
            provider="all",
        )

    async def _update_provider_health(
        self,
        provider: str,
        latency_ms: int,
        is_failure: bool,
    ):
        health_key = f"provider:{provider}:health"
        latency_key = f"provider:{provider}:latency"

        await self.cache.set(health_key, "healthy" if not is_failure else "degraded", ex=30)

        if latency_ms > 0:
            await self.cache.set(latency_key, str(latency_ms), ex=60)

    async def list_available_models(self, user_plan: str) -> list[dict[str, Any]]:
        all_models = []

        provider_config = self.provider_config._config.get("providers", {}).get("models", {})

        allowed_models = self.provider_config.get_allowed_models(user_plan)

        for tier, models in provider_config.items():
            if tier not in ["free", "pro"]:
                continue

            for model_name, model_config in models.items():
                if allowed_models == "all" or model_name in allowed_models:
                    chain = model_config.get("provider_chain", [])
                    primary_provider = chain[0]["provider"] if chain else None

                    all_models.append(
                        {
                            "id": model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": primary_provider or "unknown",
                            "tier": tier,
                        }
                    )

        return all_models

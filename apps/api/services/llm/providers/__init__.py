import os
from typing import Any

import httpx

from apps.api.services.llm.base import AnthropicCompatProvider, OpenAICompatProvider


class MiniMaxProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="minimax",
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url="https://api.minimax.io/v1",
            timeout=30,
            max_connections=100,
        )


class OpenRouterProvider(OpenAICompatProvider):
    def __init__(self, api_key_name: str = "OPENROUTER_FREE_API_KEY"):
        super().__init__(
            name="openrouter",
            api_key=os.environ.get(api_key_name, ""),
            base_url="https://openrouter.ai/api/v1",
            timeout=30,
            max_connections=50,
        )


class OpenRouterXiaomiProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="openrouter_xiaomi",
            api_key=os.environ.get("OPENROUTER_XIAOMI_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            timeout=30,
            max_connections=50,
        )


class OpenRouterDeepSeekProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="openrouter_deepseek",
            api_key=os.environ.get("OPENROUTER_DEEPSEEK_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            timeout=30,
            max_connections=50,
        )


class OpenRouterGrokProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="openrouter_grok",
            api_key=os.environ.get("OPENROUTER_GROK_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            timeout=30,
            max_connections=50,
        )


class OpenCodeChatProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="opencode",
            api_key=os.environ.get("OPENCODE_API_KEY", ""),
            base_url="https://opencode.ai/zen/go/v1",
            timeout=30,
            max_connections=100,
        )


class OpenCodeMessagesProvider(AnthropicCompatProvider):
    def __init__(self):
        super().__init__(
            name="opencode",
            api_key=os.environ.get("OPENCODE_API_KEY", ""),
            base_url="https://opencode.ai/zen/go/v1",
            timeout=30,
            max_connections=100,
        )


class ChutesProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="chutes",
            api_key=os.environ.get("CHUTES_API_KEY", ""),
            base_url="https://llm.chutes.ai/v1",
            timeout=30,
            max_connections=100,
        )


class ZAIProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="zai",
            api_key=os.environ.get("ZAI_API_KEY", ""),
            base_url="https://api.z.ai/api/paas/v4",
            timeout=30,
            max_connections=100,
        )


class CrofProvider(OpenAICompatProvider):
    def __init__(self):
        super().__init__(
            name="crof",
            api_key=os.environ.get("CROF_API_KEY", ""),
            base_url="https://crof.ai/v1",
            timeout=30,
            max_connections=100,
        )


class MiniMaxImageProvider:
    def __init__(self):
        self.name = "minimax_image"
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.io/v1"
        self.timeout = 30
        self.max_connections = 100
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    timeout=self.timeout,
                    connect=1.0,
                ),
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=20,
                ),
                http2=False,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate_image(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str = "1:1",
        response_format: str = "base64",
        **kwargs,
    ) -> dict[str, Any]:
        client = await self.get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "response_format": response_format,
        }

        if kwargs.get("size"):
            payload["size"] = kwargs["size"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await client.post(
                "/image_generation",
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()
                image_data = result.get("data", {})
                image_urls = image_data.get("image_urls", [])
                image_b64 = image_data.get("image_base64", [])
                return {
                    "data": [
                        {"url": url, "b64_json": None, "revised_prompt": None} for url in image_urls
                    ]
                    + [{"url": None, "b64_json": b64, "revised_prompt": None} for b64 in image_b64]
                }
            elif response.status_code == 429:
                raise Exception("Rate limited")
            else:
                error_data = response.json() if response.content else {}
                raise Exception(error_data.get("error", {}).get("message", "Unknown error"))
        except httpx.TimeoutException:
            raise Exception(f"Request to {self.name} timed out")
        except Exception as e:
            raise Exception(str(e))

    async def health_check(self) -> bool:
        try:
            client = await self.get_client()
            response = await client.get(
                "/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return response.status_code == 200
        except Exception:
            return False


PROVIDER_CLASSES = {
    "minimax": MiniMaxProvider,
    "minimax_image": MiniMaxImageProvider,
    "openrouter": OpenRouterProvider,
    "openrouter_xiaomi": OpenRouterXiaomiProvider,
    "openrouter_deepseek": OpenRouterDeepSeekProvider,
    "openrouter_grok": OpenRouterGrokProvider,
    "opencode_chat": OpenCodeChatProvider,
    "opencode_messages": OpenCodeMessagesProvider,
    "chutes": ChutesProvider,
    "zai": ZAIProvider,
    "crof": CrofProvider,
}

_provider_cache: dict[
    str, OpenAICompatProvider | AnthropicCompatProvider | MiniMaxImageProvider
] = {}


def get_provider(name: str) -> OpenAICompatProvider | AnthropicCompatProvider:
    if name in _provider_cache:
        return _provider_cache[name]
    provider_class = PROVIDER_CLASSES.get(name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {name}")
    provider_instance = provider_class()
    _provider_cache[name] = provider_instance
    return provider_instance


def get_provider_for_model(
    provider_name: str, model_id: str
) -> OpenAICompatProvider | AnthropicCompatProvider | MiniMaxImageProvider:
    cache_key = f"{provider_name}:{model_id}"
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    provider_instance: OpenAICompatProvider | AnthropicCompatProvider | MiniMaxImageProvider
    if provider_name == "opencode":
        if model_id in ["minimax-m2.7", "minimax-m2.5"]:
            provider_instance = OpenCodeMessagesProvider()
        else:
            provider_instance = OpenCodeChatProvider()
    elif provider_name == "openrouter_xiaomi":
        provider_instance = OpenRouterXiaomiProvider()
    elif provider_name == "openrouter_deepseek":
        provider_instance = OpenRouterDeepSeekProvider()
    elif provider_name == "openrouter_grok":
        provider_instance = OpenRouterGrokProvider()
    elif provider_name == "minimax_image":
        provider_instance = MiniMaxImageProvider()
    elif provider_name == "openrouter":
        if "xiaomi" in model_id.lower():
            provider_instance = OpenRouterXiaomiProvider()
        elif "deepseek" in model_id.lower():
            provider_instance = OpenRouterDeepSeekProvider()
        elif "grok" in model_id.lower():
            provider_instance = OpenRouterGrokProvider()
        else:
            provider_instance = OpenRouterProvider()
    else:
        provider_instance = get_provider(provider_name)

    _provider_cache[cache_key] = provider_instance
    return provider_instance

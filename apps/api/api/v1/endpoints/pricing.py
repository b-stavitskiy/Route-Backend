from fastapi import APIRouter
from pydantic import BaseModel

from apps.api.core.config import get_provider_config

router = APIRouter(prefix="/v1", tags=["pricing"])

MODEL_DISPLAY_NAMES = {
    "route/nemotron-3-super-120b": "Nemotron-3 Super 120B",
    "route/trinity-large-preview": "Trinity Large Preview",
    "route/glm-4.5-air": "GLM-4.5 Air",
    "route/nemotron-3-nano-30b": "Nemotron-3 Nano 30B",
    "route/minimax-m2.5": "MiniMax M2.5",
    "route/minimax-m2.7": "MiniMax M2.7",
    "route/minimax-m2.5-highspeed": "MiniMax M2.5 Highspeed",
    "route/minimax-m2.7-highspeed": "MiniMax M2.7 Highspeed",
    "route/kimi-k2.5": "Kimi K2.5",
    "route/qwen3-coder": "Qwen3 Coder",
    "route/qwen3-coder-next": "Qwen3 Coder Next",
    "route/qwen3-32b": "Qwen3 32B",
    "route/qwen3-next-80b": "Qwen3 Next 80B",
    "route/gpt-oss-120b": "GPT-OSS 120B",
    "route/hermes-3-llama-3.1-405b": "Hermes-3 Llama 3.1 405B",
    "route/llama-3.2-3b-instruct": "Llama 3.2 3B Instruct",
    "route/gemma-3-27b-it": "Gemma-3 27B Instruct",
    "route/glm-5": "GLM-5",
    "route/glm-5-turbo": "GLM-5 Turbo",
    "route/deepseek-v3.2": "DeepSeek V3.2",
    "route/qwen3.5-9b": "Qwen3.5 9B",
    "route/qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "route/qwen3.5-plus": "Qwen3.5 Plus",
    "route/qwen3.6-plus": "Qwen3.6 Plus",
    "route/gemma-4-31b-it": "Gemma-4 31B IT",
    "route/deepseek-v3.2-speciale": "DeepSeek V3.2 Speciale",
    "route/deepseek-r1": "DeepSeek R1",
    "route/grok-4-fast": "Grok-4 Fast",
    "route/grok-4.20-beta": "Grok-4.20 Beta",
    "route/grok-4.20-multi-agent-beta": "Grok-4.20 Multi-Agent Beta",
    "route/mimo-v2-omni": "Xiaomi MiMo V2 Omni",
    "route/mimo-v2-pro": "Xiaomi MiMo V2 Pro",
    "route/mimo-v2-flash": "Xiaomi MiMo V2 Flash",
    "route/minimax-image-1": "MiniMax Image Generation",
}


class ModelPricing(BaseModel):
    model: str
    display_name: str
    tier: str
    input_per_million: float
    output_per_million: float


class PricingResponse(BaseModel):
    models: list[ModelPricing]


@router.get("/pricing", response_model=PricingResponse)
async def get_pricing():
    provider_config = get_provider_config()
    model_pricing = provider_config._config.get("providers", {}).get("model_pricing", {})

    models = []
    for tier, tier_models in model_pricing.items():
        for model_name, pricing in tier_models.items():
            models.append(
                ModelPricing(
                    model=model_name,
                    display_name=MODEL_DISPLAY_NAMES.get(
                        model_name, model_name.replace("route/", "").replace("-", " ").title()
                    ),
                    tier=tier,
                    input_per_million=pricing.get("input_per_million", 0.0),
                    output_per_million=pricing.get("output_per_million", 0.0),
                )
            )

    return PricingResponse(models=models)

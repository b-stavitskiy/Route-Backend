from apps.api.core.config import get_provider_config


class CreditService:
    def __init__(self):
        self.provider_config = get_provider_config()

    def calculate_request_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        pricing = self.provider_config.get_model_pricing(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_million", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output_per_million", 0)

        return round(input_cost + output_cost, 6)

    def get_model_pricing(self, model: str) -> dict | None:
        return self.provider_config.get_model_pricing(model)


def get_credit_service() -> CreditService:
    return CreditService()

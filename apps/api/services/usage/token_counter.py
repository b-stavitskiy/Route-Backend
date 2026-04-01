import tiktoken
import yaml
from pathlib import Path


class TokenCounter:
    def __init__(self):
        self._encodings = {}
        self._config = self._load_config()
        self._encoding_map = self._build_encoding_map()

    def _load_config(self) -> dict:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "provider.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _build_encoding_map(self) -> dict[str, str]:
        encoding_map: dict[str, str] = {}
        for encoding_name, models in self._config.get("token_encodings", {}).items():
            for model in models:
                encoding_map[model.lower()] = encoding_name
        return encoding_map

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        encoding_name = "cl100k_base"
        model_lower = model.lower()

        for model_prefix, enc_name in self._encoding_map.items():
            if model_prefix in model_lower:
                encoding_name = enc_name
                break

        if encoding_name not in self._encodings:
            self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)

        return self._encodings[encoding_name]

    def count_messages_tokens(
        self,
        messages: list[dict],
        model: str,
    ) -> int:
        encoding = self._get_encoding(model)
        tokens = 0

        for message in messages:
            tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    tokens += len(encoding.encode(value))
            tokens += 3

        tokens += 3
        return tokens

    def count_text_tokens(self, text: str, model: str) -> int:
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        input_cost_per_1k = 0.0001
        output_cost_per_1k = 0.0003

        if "gpt-4o" in model.lower():
            input_cost_per_1k = 0.0025
            output_cost_per_1k = 0.01
        elif "gpt-4-turbo" in model.lower():
            input_cost_per_1k = 0.01
            output_cost_per_1k = 0.03
        elif "claude" in model.lower():
            input_cost_per_1k = 0.003
            output_cost_per_1k = 0.015

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k


token_counter = TokenCounter()

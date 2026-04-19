import apps.api.services.llm.providers as providers


def reset_opencode_state() -> None:
    providers._opencode_api_keys = []
    providers._opencode_key_index = 0
    providers._opencode_chat_providers = []
    providers._opencode_message_providers = []
    providers._opencode_provider_index = 0


def test_opencode_uses_primary_key_when_secondary_missing(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "primary-key")
    monkeypatch.delenv("OPENCODE_API_KEY_2", raising=False)
    reset_opencode_state()

    first = providers._get_next_opencode_key()
    second = providers._get_next_opencode_key()

    assert first == "primary-key"
    assert second == "primary-key"


def test_opencode_ignores_blank_secondary_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "primary-key")
    monkeypatch.setenv("OPENCODE_API_KEY_2", "   ")
    reset_opencode_state()

    provider_one = providers._get_opencode_chat_provider()
    provider_two = providers._get_opencode_chat_provider()

    assert provider_one is provider_two
    assert provider_one.api_key == "primary-key"


def test_opencode_round_robins_when_both_keys_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "primary-key")
    monkeypatch.setenv("OPENCODE_API_KEY_2", "secondary-key")
    reset_opencode_state()

    first = providers._get_next_opencode_key()
    second = providers._get_next_opencode_key()
    third = providers._get_next_opencode_key()

    assert [first, second, third] == ["primary-key", "secondary-key", "primary-key"]

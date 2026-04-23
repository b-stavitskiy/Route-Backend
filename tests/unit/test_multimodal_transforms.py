from apps.api.api.v1.endpoints.anthropic import convert_to_openai_format
from apps.api.services.llm.transforms import transform_anthropic_messages


def test_transform_anthropic_messages_converts_openai_image_blocks() -> None:
    messages, _ = transform_anthropic_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    },
                ],
            }
        ]
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "ZmFrZQ==",
                    },
                },
            ],
        }
    ]


def test_convert_to_openai_format_preserves_anthropic_images() -> None:
    messages = convert_to_openai_format(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is shown?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc123",
                        },
                    },
                ],
            }
        ],
        system=None,
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is shown?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc123"},
                },
            ],
        }
    ]

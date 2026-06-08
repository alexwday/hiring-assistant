from utilities.config import ConversationConfig
from utilities.conversation import normalize_messages


def test_normalize_messages_filters_and_trims():
    config = ConversationConfig(
        include_system_messages=False,
        allowed_roles=("user", "assistant"),
        max_history_length=2,
    )
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    normalized = normalize_messages(messages, config)

    assert normalized == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


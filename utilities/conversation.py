"""Conversation setup and validation helpers."""

from __future__ import annotations

from typing import Any

from utilities.config import ConversationConfig
from utilities.prompt_loader import PromptDefinition

VALID_ROLES = {"system", "user", "assistant", "tool"}


def normalize_messages(
    conversation_input: Any,
    config: ConversationConfig,
) -> list[dict[str, str]]:
    """Validate, filter, and trim incoming conversation messages."""
    if isinstance(conversation_input, dict):
        messages = conversation_input.get("messages")
    else:
        messages = conversation_input

    if messages is None:
        return []
    if not isinstance(messages, list):
        raise ValueError("Conversation messages must be a list")

    normalized: list[dict[str, str]] = []
    allowed = set(config.allowed_roles)
    for index, message in enumerate(messages):
        normalized_message = _normalize_message(message, index)
        role = normalized_message["role"]
        if role == "system" and not config.include_system_messages:
            continue
        if role not in allowed:
            continue
        normalized.append(normalized_message)

    if len(normalized) > config.max_history_length:
        normalized = normalized[-config.max_history_length :]
    return normalized


def messages_from_prompt(
    prompt: PromptDefinition,
    conversation_history: Any = None,
    config: ConversationConfig | None = None,
) -> list[dict[str, str]]:
    """Build chat messages from optional history plus the prompt."""
    if conversation_history is None:
        return prompt.messages()
    if config is None:
        raise ValueError("config is required when conversation_history is provided")

    messages = normalize_messages(conversation_history, config)
    messages.append({"role": "system", "content": prompt.system_prompt})
    messages.append({"role": "user", "content": prompt.user_prompt})
    return messages


def _normalize_message(message: Any, index: int) -> dict[str, str]:
    """Validate and normalize one chat message."""
    if not isinstance(message, dict):
        raise ValueError(f"Message at index {index} must be a mapping")

    role = message.get("role")
    content = message.get("content")
    if role not in VALID_ROLES:
        raise ValueError(f"Message at index {index} has invalid role {role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Message at index {index} requires non-empty string content")
    return {"role": role, "content": content}


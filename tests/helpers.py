"""Shared test helpers for framework configuration objects."""

from connections.oauth_connector import OAuthConfig
from utilities.config import (
    AppConfig,
    AuthConfig,
    ConversationConfig,
    DatabaseConfig,
    LLMConfig,
    LLMModelConfig,
    PromptConfig,
    RuntimeConfig,
    SSLConfig,
)


def build_model_config(
    model: str = "test-model",
    max_tokens: int = 100,
) -> LLMModelConfig:
    """Build a small LLM profile for unit tests."""
    return LLMModelConfig(
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=30,
        max_retries=0,
        reasoning_effort=None,
    )


def build_app_config(
    small_model: str = "test-model",
    large_model: str = "large-model",
) -> AppConfig:
    """Build a complete application config for unit tests."""
    return AppConfig(
        auth=AuthConfig(
            mode="local",
            api_key="test-key",
            oauth=OAuthConfig(token_endpoint="", client_id="", client_secret=""),
        ),
        ssl=SSLConfig(verify=False),
        database=DatabaseConfig(
            host="127.0.0.1",
            port=5432,
            database="postgres",
            user="postgres",
            password="",
            schema="public",
        ),
        llm=LLMConfig(
            base_url="https://example.test/v1",
            small=build_model_config(small_model, 200),
            large=build_model_config(large_model, 500),
        ),
        prompt=build_prompt_config(),
        conversation=build_conversation_config(),
        runtime=RuntimeConfig(log_level="INFO", output_logs=False),
    )


def build_prompt_config() -> PromptConfig:
    """Build the default prompt selector config for unit tests."""
    return PromptConfig(
        model="base_llm_framework",
        layer="default",
        default_prompt="example",
    )


def build_conversation_config() -> ConversationConfig:
    """Build the default conversation config for unit tests."""
    return ConversationConfig(
        include_system_messages=True,
        allowed_roles=("system", "user", "assistant"),
        max_history_length=20,
    )

"""OpenAI-compatible LLM connector with local API-key or OAuth auth."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import httpx
from openai import OpenAI

from connections.oauth_connector import OAuthClient
from utilities.config import AppConfig, load_config
from utilities.ssl_setup import SSLSetupResult

logger = logging.getLogger(__name__)

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    """Return whether chat params should use reasoning-model conventions."""
    lowered = model.lower()
    return lowered.startswith(REASONING_MODEL_PREFIXES)


def _prepare_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Clone tool definitions so callers can safely reuse prompt data."""
    return copy.deepcopy(tools or [])


class LLMClient:
    """OpenAI-compatible client wrapper with swappable authentication."""

    def __init__(
        self,
        config: AppConfig | None = None,
        ssl_setup: SSLSetupResult | None = None,
    ):
        """Create the connector without making a live LLM request."""
        self.config = config or load_config()
        self.ssl_setup = ssl_setup
        self.oauth_client: OAuthClient | None = None
        self._openai_client: OpenAI | None = None
        self._oauth_token = ""
        self._http_client: httpx.Client | None = None

        if self.config.auth_mode == "oauth":
            verify = self._ssl_verify_value()
            self.oauth_client = OAuthClient(self.config.oauth, verify=verify)

    def call(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make one chat completion call and return the SDK response as a dict."""
        settings = settings or {}
        client = self.get_client()
        kwargs = self._build_chat_kwargs(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            settings=settings,
        )

        logger.info(
            "Calling LLM: model=%s messages=%s tools=%s",
            kwargs["model"],
            len(messages),
            len(kwargs.get("tools", [])),
        )
        response = client.chat.completions.create(**kwargs)
        return response.model_dump()

    def get_client(self) -> OpenAI:
        """Return an OpenAI SDK client configured for the active auth mode."""
        if self.config.auth_mode == "local":
            if not self.config.api_key:
                raise ValueError("OPENAI_API_KEY or API_KEY is required for AUTH_MODE=local")
            if self._openai_client is None:
                self._openai_client = self._new_openai_client(self.config.api_key)
            return self._openai_client

        if self.oauth_client is None:
            raise ValueError("OAuth client was not initialized")
        token = self.oauth_client.get_token()
        if self._openai_client is not None and token == self._oauth_token:
            return self._openai_client

        self.close()
        self._openai_client = self._new_openai_client(token)
        self._oauth_token = token
        return self._openai_client

    def close(self) -> None:
        """Close the underlying HTTP client when one is active."""
        if self._http_client is not None:
            self._http_client.close()
        self._http_client = None
        self._openai_client = None

    def _new_openai_client(self, api_key: str) -> OpenAI:
        """Create a configured OpenAI SDK client."""
        timeout_seconds = max(
            self.config.llm.small.timeout_seconds,
            self.config.llm.large.timeout_seconds,
        )
        max_retries = max(
            self.config.llm.small.max_retries,
            self.config.llm.large.max_retries,
        )
        self._http_client = httpx.Client(
            verify=self._ssl_verify_value(),
            timeout=httpx.Timeout(timeout_seconds),
        )
        return OpenAI(
            api_key=api_key,
            base_url=self.config.llm.base_url,
            http_client=self._http_client,
            max_retries=max_retries,
        )

    def _ssl_verify_value(self) -> bool | str:
        """Return the verification argument for httpx/requests."""
        if self.ssl_setup is not None:
            return self.ssl_setup.verify_value
        if not self.config.ssl.verify:
            return False
        return True

    def _build_chat_kwargs(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Build OpenAI chat-completions request params."""
        model_size = str(settings.get("model_size") or "small")
        profile = self.config.llm.get_profile(model_size)
        model = str(settings.get("model") or profile.model)
        max_tokens = int(settings.get("max_tokens") or profile.max_tokens)
        temperature = settings.get("temperature")
        reasoning_effort = settings.get(
            "reasoning_effort",
            profile.reasoning_effort,
        )
        reasoning_model = _is_reasoning_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = float(temperature)

        prepared_tools = _prepare_tools(tools)
        if prepared_tools:
            kwargs["tools"] = prepared_tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        passthrough_keys = {
            "frequency_penalty",
            "logit_bias",
            "metadata",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "top_p",
        }
        for key in passthrough_keys:
            if key in settings:
                kwargs[key] = settings[key]
        return kwargs


def extract_message_text(response: dict[str, Any]) -> str:
    """Return the first assistant text content from a chat response."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def extract_tool_arguments(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parsed function-call arguments from the first assistant message."""
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    parsed: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": raw_arguments}
        parsed.append(
            {
                "id": tool_call.get("id"),
                "name": function.get("name"),
                "arguments": arguments,
            }
        )
    return parsed

"""Tests for OpenAI-compatible chat request construction."""

from types import SimpleNamespace

from connections.llm_connector import LLMClient
from tests.helpers import build_app_config


def build_fake_openai_client(**_kwargs):
    """Build a fake OpenAI SDK client that echoes completion kwargs."""

    def create(**kwargs):
        """Return an SDK-like response object."""

        def model_dump():
            """Return a dict matching the SDK response method."""
            return {"payload": kwargs}

        return SimpleNamespace(model_dump=model_dump)

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def test_llm_client_builds_chat_completion_kwargs(monkeypatch):
    """It sends standard chat completion params for non-reasoning models."""
    monkeypatch.setattr("connections.llm_connector.OpenAI", build_fake_openai_client)

    client = LLMClient(config=build_app_config())
    response = client.call(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "answer",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
        settings={"max_tokens": 50},
    )

    payload = response["payload"]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 50
    assert "temperature" not in payload
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "answer"


def test_llm_client_uses_reasoning_params_for_o_series(monkeypatch):
    """It uses max_completion_tokens for o-series reasoning models."""
    monkeypatch.setattr("connections.llm_connector.OpenAI", build_fake_openai_client)

    client = LLMClient(config=build_app_config(small_model="o3-mini"))
    response = client.call(
        messages=[{"role": "user", "content": "hello"}],
        settings={"reasoning_effort": "low"},
    )

    payload = response["payload"]
    assert payload["max_completion_tokens"] == 200
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_llm_client_uses_reasoning_params_for_gpt_5_dot_model(monkeypatch):
    """It treats GPT-5 models as reasoning models."""
    monkeypatch.setattr("connections.llm_connector.OpenAI", build_fake_openai_client)

    client = LLMClient(config=build_app_config(small_model="gpt-5.4-mini"))
    response = client.call(
        messages=[{"role": "user", "content": "hello"}],
        settings={"max_tokens": 75},
    )

    payload = response["payload"]
    assert payload["model"] == "gpt-5.4-mini"
    assert payload["max_completion_tokens"] == 75
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_llm_client_uses_large_profile_when_prompt_selects_large(monkeypatch):
    """It resolves the large model profile when prompt metadata requests it."""
    monkeypatch.setattr("connections.llm_connector.OpenAI", build_fake_openai_client)

    client = LLMClient(config=build_app_config())
    response = client.call(
        messages=[{"role": "user", "content": "hello"}],
        settings={"model_size": "large"},
    )

    payload = response["payload"]
    assert payload["model"] == "large-model"
    assert payload["max_tokens"] == 500

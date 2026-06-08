from connections.llm_connector import LLMClient
from connections.oauth_connector import OAuthConfig
from utilities.config import (
    AppConfig,
    ConversationConfig,
    DatabaseConfig,
    LLMConfig,
    LLMModelConfig,
    PromptConfig,
    SSLConfig,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return {"payload": self.payload}


class FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(kwargs)


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def _model_config(model="test-model", max_tokens=200):
    return LLMModelConfig(
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=30,
        max_retries=0,
        reasoning_effort=None,
    )


def _config(small_model="test-model", large_model="large-model"):
    return AppConfig(
        auth_mode="local",
        api_key="test-key",
        oauth=OAuthConfig(token_endpoint="", client_id="", client_secret=""),
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
            small=_model_config(small_model, 200),
            large=_model_config(large_model, 500),
        ),
        prompt=PromptConfig(
            model="base_llm_framework",
            layer="default",
            default_prompt="example",
        ),
        conversation=ConversationConfig(
            include_system_messages=True,
            allowed_roles=("system", "user", "assistant"),
            max_history_length=20,
        ),
        log_level="INFO",
        output_logs=False,
    )


def test_llm_client_builds_chat_completion_kwargs(monkeypatch):
    monkeypatch.setattr("connections.llm_connector.OpenAI", FakeClient)

    client = LLMClient(config=_config())
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
    monkeypatch.setattr("connections.llm_connector.OpenAI", FakeClient)

    client = LLMClient(config=_config(small_model="o3-mini"))
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
    monkeypatch.setattr("connections.llm_connector.OpenAI", FakeClient)

    client = LLMClient(config=_config(small_model="gpt-5.4-mini"))
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
    monkeypatch.setattr("connections.llm_connector.OpenAI", FakeClient)

    client = LLMClient(config=_config())
    response = client.call(
        messages=[{"role": "user", "content": "hello"}],
        settings={"model_size": "large"},
    )

    payload = response["payload"]
    assert payload["model"] == "large-model"
    assert payload["max_tokens"] == 500

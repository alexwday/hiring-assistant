import pytest

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
from utilities.prompt_loader import ensure_prompts_table, validate_prompt_row


class FakeCursor:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchone(self):
        return [self.result]


class FakeConnection:
    def __init__(self, result):
        self.cursor_obj = FakeCursor(result)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self, *_, **__):
        return self.cursor_obj


def _config():
    model = LLMModelConfig(
        model="test-model",
        max_tokens=100,
        timeout_seconds=30,
        max_retries=0,
        reasoning_effort=None,
    )
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
            small=model,
            large=model,
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


def test_validate_prompt_row_reads_metadata_and_tool_definition():
    prompt = validate_prompt_row(
        {
            "model": "base_llm_framework",
            "layer": "default",
            "name": "classify",
            "description": "Classify text.",
            "comments": (
                '{"model_size": "large", "settings": {"max_tokens": 400}, '
                '"tool_choice": "auto"}'
            ),
            "system_prompt": "You classify text about {topic}.",
            "user_prompt": "Classify this: {text}",
            "tool_definition": {
                "type": "function",
                "function": {
                    "name": "classify",
                    "description": "Classify text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                    },
                },
            },
            "uses_global": [],
            "version": "1.0.0",
        }
    ).render({"topic": "banks", "text": "Revenue rose."})

    assert prompt.name == "classify"
    assert prompt.model_size == "large"
    assert prompt.tools is not None
    assert prompt.tools[0]["function"]["name"] == "classify"
    assert prompt.tool_choice == "auto"
    assert prompt.settings == {"max_tokens": 400}
    assert prompt.llm_settings()["model_size"] == "large"
    assert prompt.messages()[0]["content"] == "You classify text about banks."


def test_prompt_render_reports_missing_variable():
    prompt = validate_prompt_row(
        {
            "model": "base_llm_framework",
            "layer": "default",
            "name": "example",
            "comments": "{}",
            "system_prompt": "Hello {name}",
            "user_prompt": "Go",
            "tool_definition": None,
            "uses_global": [],
            "version": "1.0.0",
        }
    )

    with pytest.raises(ValueError, match="Missing prompt variable 'name'"):
        prompt.render({})


def test_ensure_prompts_table_checks_existing_table_without_ddl(monkeypatch):
    fake_connection = FakeConnection("public.prompts")
    monkeypatch.setattr(
        "utilities.prompt_loader.connection_scope",
        lambda *_args, **_kwargs: fake_connection,
    )

    ensure_prompts_table(config=_config())

    query, params = fake_connection.cursor_obj.queries[0]
    assert "to_regclass" in query
    assert "CREATE TABLE" not in query
    assert params == ("public.prompts",)


def test_ensure_prompts_table_reports_missing_table(monkeypatch):
    fake_connection = FakeConnection(None)
    monkeypatch.setattr(
        "utilities.prompt_loader.connection_scope",
        lambda *_args, **_kwargs: fake_connection,
    )

    with pytest.raises(FileNotFoundError, match="prompts table not found"):
        ensure_prompts_table(config=_config())

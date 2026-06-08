"""Tests for PostgreSQL prompt row validation and table checks."""

import pytest

from tests.helpers import build_app_config
from utilities.prompt_loader import ensure_prompts_table, validate_prompt_row


class FakeCursor:
    """Minimal context-manager cursor for prompt table checks."""

    def __init__(self, result: str | None):
        """Store a single fetch result."""
        self.result = result
        self.queries = []

    def __enter__(self):
        """Return this cursor for context-manager usage."""
        return self

    def __exit__(self, *_):
        """Do not suppress context-manager exceptions."""
        return False

    def execute(self, query, params=None) -> None:
        """Record executed SQL and parameters."""
        self.queries.append((query, params))

    def fetchone(self) -> list[str | None]:
        """Return one database-like row."""
        return [self.result]


class FakeConnection:
    """Minimal context-manager connection for prompt table checks."""

    def __init__(self, result: str | None):
        """Create one fake cursor for inspection."""
        self.cursor_obj = FakeCursor(result)

    def __enter__(self):
        """Return this connection for context-manager usage."""
        return self

    def __exit__(self, *_):
        """Do not suppress context-manager exceptions."""
        return False

    def cursor(self, *_, **__):
        """Return the fake cursor."""
        return self.cursor_obj

    def commit(self) -> None:
        """Provide a no-op commit method for DB-helper compatibility."""


def test_validate_prompt_row_reads_metadata_and_tool_definition():
    """It reads prompt metadata and normalizes tool definitions."""
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
    """It raises a clear error when a template variable is missing."""
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
    """It checks table existence without attempting schema DDL."""
    fake_connection = FakeConnection("public.prompts")
    monkeypatch.setattr(
        "utilities.prompt_loader.connection_scope",
        lambda *_args, **_kwargs: fake_connection,
    )

    ensure_prompts_table(config=build_app_config())

    query, params = fake_connection.cursor_obj.queries[0]
    assert "to_regclass" in query
    assert "CREATE TABLE" not in query
    assert params == ("public.prompts",)


def test_ensure_prompts_table_reports_missing_table(monkeypatch):
    """It reports a missing prompts table before seeding."""
    fake_connection = FakeConnection(None)
    monkeypatch.setattr(
        "utilities.prompt_loader.connection_scope",
        lambda *_args, **_kwargs: fake_connection,
    )

    with pytest.raises(FileNotFoundError, match="prompts table not found"):
        ensure_prompts_table(config=build_app_config())

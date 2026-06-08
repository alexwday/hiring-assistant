"""PostgreSQL-backed prompt loading from the prompts table."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from psycopg2.extras import Json, RealDictCursor

from connections.postgres_connector import connection_scope, fetch_one
from utilities.config import AppConfig, load_config

MODEL_SIZES = {"small", "large"}
PROMPTS_TABLE = "prompts"


@dataclass(frozen=True)
class PromptIdentity:
    """Stable prompt table identity fields."""

    name: str
    model: str
    layer: str


@dataclass(frozen=True)
class PromptMetadata:
    """Descriptive prompt table metadata."""

    description: str = ""
    version: str = ""


@dataclass(frozen=True)
class PromptRuntime:
    """Prompt-level LLM runtime settings."""

    model_size: str = "small"
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    settings: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptDefinition:
    """Validated prompt table content."""

    identity: PromptIdentity
    system_prompt: str
    user_prompt: str
    metadata: PromptMetadata
    runtime: PromptRuntime

    @property
    def name(self) -> str:
        """Return the prompt name."""
        return self.identity.name

    @property
    def model(self) -> str:
        """Return the prompt model namespace."""
        return self.identity.model

    @property
    def layer(self) -> str:
        """Return the prompt layer."""
        return self.identity.layer

    @property
    def description(self) -> str:
        """Return the prompt description."""
        return self.metadata.description

    @property
    def version(self) -> str:
        """Return the prompt version."""
        return self.metadata.version

    @property
    def model_size(self) -> str:
        """Return the selected LLM model profile size."""
        return self.runtime.model_size

    @property
    def tools(self) -> list[dict[str, Any]] | None:
        """Return normalized OpenAI tool definitions."""
        return self.runtime.tools

    @property
    def tool_choice(self) -> str | dict[str, Any] | None:
        """Return the prompt tool choice, if configured."""
        return self.runtime.tool_choice

    @property
    def settings(self) -> dict[str, Any] | None:
        """Return prompt-level LLM request settings."""
        return self.runtime.settings

    def messages(self) -> list[dict[str, str]]:
        """Return system and user messages for the chat API."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def llm_settings(self) -> dict[str, Any]:
        """Return LLM request settings with the selected model profile."""
        return {**(self.runtime.settings or {}), "model_size": self.runtime.model_size}

    def render(self, variables: dict[str, Any]) -> "PromptDefinition":
        """Format prompt templates with supplied variables."""
        return replace(
            self,
            system_prompt=_format_template(
                self.system_prompt, variables, "system_prompt"
            ),
            user_prompt=_format_template(self.user_prompt, variables, "user_prompt"),
        )


def load_prompt(
    prompt_ref: str,
    config: AppConfig | None = None,
    layer: str | None = None,
    prompt_model: str | None = None,
) -> PromptDefinition:
    """Load and validate the latest prompt row from PostgreSQL."""
    config = config or load_config()
    row = get_latest_prompt_row(
        model=prompt_model or config.prompt.model,
        layer=layer or config.prompt.layer,
        name=prompt_ref,
        config=config,
    )
    if row is None:
        raise FileNotFoundError(
            "Prompt not found in PostgreSQL prompts table: "
            f"model={prompt_model or config.prompt.model!r}, "
            f"layer={layer or config.prompt.layer!r}, name={prompt_ref!r}"
        )
    return validate_prompt_row(row, config=config)


def get_latest_prompt_row(
    model: str,
    layer: str,
    name: str,
    config: AppConfig | None = None,
) -> dict[str, Any] | None:
    """Return the latest prompt table row for model/layer/name."""
    return fetch_one(
        """
        SELECT
            id,
            model,
            layer,
            name,
            description,
            comments,
            system_prompt,
            user_prompt,
            tool_definition,
            uses_global,
            version,
            created_at,
            updated_at
        FROM prompts
        WHERE model = %s
          AND layer = %s
          AND name = %s
        ORDER BY updated_at DESC NULLS LAST, id DESC
        LIMIT 1
        """,
        (model, layer, name),
        config=config,
    )


def validate_prompt_row(
    row: dict[str, Any],
    config: AppConfig | None = None,
) -> PromptDefinition:
    """Validate and normalize one prompt table row."""
    metadata = _metadata_from_comments(row.get("comments"))
    uses_global = _normalize_uses_global(row.get("uses_global"))
    system_prompt = _required_str(row, "system_prompt")

    if uses_global:
        global_prompts = _load_global_prompts(
            uses_global=uses_global,
            model=_required_str(row, "model"),
            config=config,
        )
        if global_prompts:
            system_prompt = "\n\n---\n\n".join([*global_prompts, system_prompt])

    return PromptDefinition(
        identity=PromptIdentity(
            name=_required_str(row, "name"),
            model=_required_str(row, "model"),
            layer=_required_str(row, "layer"),
        ),
        metadata=PromptMetadata(
            description=_optional_str(row.get("description")),
            version=_optional_str(row.get("version")),
        ),
        runtime=PromptRuntime(
            model_size=_model_size(metadata.get("model_size")),
            tools=_normalize_tools(row.get("tool_definition")),
            tool_choice=metadata.get("tool_choice"),
            settings=_normalize_settings(metadata.get("settings")),
        ),
        system_prompt=system_prompt,
        user_prompt=_required_str(row, "user_prompt"),
    )


def ensure_prompts_table(config: AppConfig | None = None) -> None:
    """Verify the prompts table exists without issuing schema DDL."""
    config = config or load_config()
    table_ref = f"{config.database.schema}.{PROMPTS_TABLE}"
    with connection_scope(config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT to_regclass(%s)
                """,
                (table_ref,),
            )
            exists = cursor.fetchone()[0] is not None
    if not exists:
        raise FileNotFoundError(
            f"PostgreSQL prompts table not found: {table_ref}. "
            "Create it with an owner/migration account before running the seed script."
        )


def upsert_prompt(
    prompt: dict[str, Any],
    config: AppConfig | None = None,
) -> None:
    """Insert or update one prompt row identified by model/layer/name/version."""
    ensure_prompts_table(config)
    with connection_scope(config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            selector = (
                prompt["model"],
                prompt["layer"],
                prompt["name"],
                prompt["version"],
            )
            cursor.execute(
                """
                SELECT id
                FROM prompts
                WHERE model = %s
                  AND layer = %s
                  AND name = %s
                  AND version = %s
                ORDER BY updated_at DESC NULLS LAST, id DESC
                LIMIT 1
                """,
                selector,
            )
            existing = cursor.fetchone()
            params = _prompt_write_params(prompt)
            if existing:
                cursor.execute(
                    """
                    UPDATE prompts
                    SET description = %(description)s,
                        comments = %(comments)s,
                        system_prompt = %(system_prompt)s,
                        user_prompt = %(user_prompt)s,
                        tool_definition = %(tool_definition)s,
                        uses_global = %(uses_global)s,
                        updated_at = NOW()
                    WHERE id = %(id)s
                    """,
                    {**params, "id": existing["id"]},
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO prompts (
                        model,
                        layer,
                        name,
                        description,
                        comments,
                        system_prompt,
                        user_prompt,
                        tool_definition,
                        uses_global,
                        version,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        %(model)s,
                        %(layer)s,
                        %(name)s,
                        %(description)s,
                        %(comments)s,
                        %(system_prompt)s,
                        %(user_prompt)s,
                        %(tool_definition)s,
                        %(uses_global)s,
                        %(version)s,
                        NOW(),
                        NOW()
                    )
                    """,
                    params,
                )
        conn.commit()


def _load_global_prompts(
    uses_global: list[str],
    model: str,
    config: AppConfig | None,
) -> list[str]:
    """Load global prompt fragments from the prompts table."""
    prompt_parts: list[str] = []
    for global_name in uses_global:
        global_row = get_latest_prompt_row(
            model=model,
            layer="global",
            name=global_name,
            config=config,
        )
        if global_row and global_row.get("system_prompt"):
            prompt_parts.append(str(global_row["system_prompt"]).strip())
    return prompt_parts


def _prompt_write_params(prompt: dict[str, Any]) -> dict[str, Any]:
    """Normalize prompt write params for psycopg2."""
    metadata = {
        "model_size": prompt.get("model_size", "small"),
        "settings": prompt.get("settings") or {},
        "tool_choice": prompt.get("tool_choice"),
    }
    tool_definition = prompt.get("tool_definition")
    return {
        "model": prompt["model"],
        "layer": prompt["layer"],
        "name": prompt["name"],
        "description": prompt.get("description"),
        "comments": json.dumps(metadata, sort_keys=True),
        "system_prompt": prompt["system_prompt"],
        "user_prompt": prompt["user_prompt"],
        "tool_definition": (
            Json(tool_definition) if tool_definition is not None else None
        ),
        "uses_global": prompt.get("uses_global") or [],
        "version": prompt.get("version", "1.0.0"),
    }


def _metadata_from_comments(comments: Any) -> dict[str, Any]:
    """Parse optional JSON metadata stored in the prompts.comments column."""
    if comments is None:
        return {}
    if isinstance(comments, dict):
        return comments
    if isinstance(comments, str):
        value = comments.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_settings(value: Any) -> dict[str, Any]:
    """Return optional prompt-level LLM request settings."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Prompt metadata settings must be a mapping")
    return value


def _normalize_tools(raw_tool_definition: Any) -> list[dict[str, Any]]:
    """Normalize prompt table tool_definition into an OpenAI tools list."""
    if raw_tool_definition is None:
        return []
    if isinstance(raw_tool_definition, str):
        raw_tool_definition = json.loads(raw_tool_definition)
    if isinstance(raw_tool_definition, dict):
        tools = [raw_tool_definition]
    elif isinstance(raw_tool_definition, list):
        tools = raw_tool_definition
    else:
        raise ValueError("tool_definition must be a mapping or list")

    formatted_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Tool at index {index} must be a mapping")
        if tool.get("type") != "function":
            raise ValueError(f"Tool at index {index} must have type='function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Tool at index {index} requires function mapping")
        if not function.get("name"):
            raise ValueError(f"Tool at index {index} requires function.name")
        if not isinstance(function.get("parameters"), dict):
            raise ValueError(f"Tool at index {index} requires function.parameters")
        formatted_tools.append(tool)
    return formatted_tools


def _normalize_uses_global(value: Any) -> list[str]:
    """Normalize uses_global from Postgres array or JSON/text value."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return _normalize_uses_global(parsed)
    raise ValueError("uses_global must be a list, tuple, string, or null")


def _model_size(value: Any) -> str:
    """Return a validated model profile name."""
    if value is None:
        value = "small"
    if not isinstance(value, str):
        raise ValueError("model_size must be a string")
    value = value.strip().lower()
    if value not in MODEL_SIZES:
        allowed = ", ".join(sorted(MODEL_SIZES))
        raise ValueError(f"model_size must be one of: {allowed}; got {value!r}")
    return value


def _required_str(row: dict[str, Any], field: str) -> str:
    """Return a required non-empty string field."""
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt row requires non-empty {field}")
    return value.strip()


def _optional_str(value: Any) -> str:
    """Return an optional string field."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def _format_template(template: str, variables: dict[str, Any], field: str) -> str:
    """Format one prompt template with a clear missing-variable error."""
    try:
        return template.format(**variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Missing prompt variable {missing!r} for {field}") from exc

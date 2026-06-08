"""Environment and runtime configuration for the base LLM framework."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from connections.oauth_connector import OAuthConfig

AUTH_MODES = {"local", "oauth"}
MODEL_SIZES = {"small", "large"}
TRUTHY = {"1", "true", "yes", "y", "on"}
FALSY = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class SSLConfig:
    """SSL configuration settings."""

    verify: bool


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection settings."""

    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str


@dataclass(frozen=True)
class LLMModelConfig:
    """OpenAI-compatible settings for one model profile."""

    model: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    reasoning_effort: str | None


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible chat model profiles."""

    base_url: str
    small: LLMModelConfig
    large: LLMModelConfig

    def get_profile(self, model_size: str) -> LLMModelConfig:
        """Return the requested model profile."""
        value = model_size.strip().lower()
        if value == "small":
            return self.small
        if value == "large":
            return self.large
        allowed = ", ".join(sorted(MODEL_SIZES))
        raise ValueError(f"model_size must be one of: {allowed}; got {model_size!r}")


@dataclass(frozen=True)
class PromptConfig:
    """Prompt loading settings."""

    model: str
    layer: str
    default_prompt: str


@dataclass(frozen=True)
class ConversationConfig:
    """Conversation normalization settings."""

    include_system_messages: bool
    allowed_roles: tuple[str, ...]
    max_history_length: int


@dataclass(frozen=True)
class AppConfig:
    """Full application configuration."""

    auth_mode: str
    api_key: str
    oauth: OAuthConfig
    ssl: SSLConfig
    database: DatabaseConfig
    llm: LLMConfig
    prompt: PromptConfig
    conversation: ConversationConfig
    log_level: str
    output_logs: bool


def load_config(
    env_path: str | Path | None = None,
    override: bool = False,
) -> AppConfig:
    """Load ``.env`` values and return typed runtime configuration."""
    path = Path(env_path) if env_path is not None else Path(".env")
    if path.exists():
        load_env_file(path, override=override)

    return AppConfig(
        auth_mode=_auth_mode(),
        api_key=_env_any(["OPENAI_API_KEY", "API_KEY"], ""),
        oauth=OAuthConfig(
            token_endpoint=_env_any(["OAUTH_TOKEN_ENDPOINT", "OAUTH_ENDPOINT"], ""),
            client_id=_env("OAUTH_CLIENT_ID", ""),
            client_secret=_env("OAUTH_CLIENT_SECRET", ""),
            grant_type=_env("OAUTH_GRANT_TYPE", "client_credentials"),
            scope=_env("OAUTH_SCOPE", ""),
            max_retries=_int("OAUTH_MAX_RETRIES", "3", minimum=1),
            retry_delay_seconds=_float("OAUTH_RETRY_DELAY_SECONDS", "1.0", minimum=0.0),
            timeout_seconds=_float("OAUTH_TIMEOUT_SECONDS", "30.0", minimum=0.0),
        ),
        ssl=SSLConfig(
            verify=_bool("SSL_VERIFY", False),
        ),
        database=DatabaseConfig(
            host=_env_any(["DB_HOST", "POSTGRES_HOST"], ""),
            port=_int_any(["DB_PORT", "POSTGRES_PORT"], "5432", 1),
            database=_env_any(["DB_NAME", "POSTGRES_DATABASE"], ""),
            user=_env_any(["DB_USER", "POSTGRES_USER"], ""),
            password=_env_any(["DB_PASSWORD", "POSTGRES_PASSWORD"], ""),
            schema=_env_any(["DB_SCHEMA", "POSTGRES_SCHEMA"], "public"),
        ),
        llm=LLMConfig(
            base_url=_env_any(
                ["LLM_BASE_URL", "LLM_DEFAULT_URL"],
                "https://api.openai.com/v1",
            ).rstrip("/"),
            small=_load_llm_profile("SMALL", default_model="gpt-5.4"),
            large=_load_llm_profile("LARGE", default_model="gpt-5.4"),
        ),
        prompt=PromptConfig(
            model="base_llm_framework",
            layer="default",
            default_prompt="example",
        ),
        conversation=ConversationConfig(
            include_system_messages=_bool("INCLUDE_SYSTEM_MESSAGES", True),
            allowed_roles=tuple(
                role.strip()
                for role in _env("ALLOWED_ROLES", "system,user,assistant").split(",")
                if role.strip()
            ),
            max_history_length=_int("MAX_HISTORY_LENGTH", "20", minimum=1),
        ),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        output_logs=_bool("OUTPUT_LOGS", False),
    )


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load dotenv-style key/value pairs into ``os.environ``."""
    values = read_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse dotenv-style key/value pairs without mutating the environment."""
    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _clean_env_value(raw_value)
    return values


def _clean_env_value(raw_value: str) -> str:
    """Normalize one raw dotenv value while preserving quoted content."""
    value = raw_value.strip()
    if not value:
        return ""

    quote = value[0] if value[0] in {"'", '"'} else ""
    if quote:
        end_index = _find_closing_quote(value, quote)
        if end_index is not None:
            return value[1:end_index]

    return _strip_inline_comment(value).strip()


def _find_closing_quote(value: str, quote: str) -> int | None:
    """Return the closing quote index, ignoring escaped quote characters."""
    escaped = False
    for index, char in enumerate(value[1:], start=1):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return index
    return None


def _strip_inline_comment(value: str) -> str:
    """Remove unquoted inline comments from a dotenv value."""
    in_single = False
    in_double = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            previous = value[index - 1] if index else ""
            if not previous or previous.isspace():
                return value[:index]
    return value


def _auth_mode() -> str:
    """Return normalized auth mode."""
    raw = (_env("AUTH_MODE", "") or _env("LLM_AUTH_MODE", "local")).strip().lower()
    aliases = {
        "api_key": "local",
        "apikey": "local",
        "default": "local",
    }
    value = aliases.get(raw, raw)
    if value not in AUTH_MODES:
        allowed = ", ".join(sorted(AUTH_MODES))
        raise ValueError(f"AUTH_MODE must be one of: {allowed}; got {raw!r}")
    return value


def _load_llm_profile(size: str, default_model: str) -> LLMModelConfig:
    """Load one LLM profile from suffixed environment variables."""
    legacy_suffixes = [""] if size == "SMALL" else []
    model_names = [f"LLM_MODEL_{size}", *(f"LLM_MODEL{s}" for s in legacy_suffixes)]
    max_tokens_names = [
        f"LLM_MAX_TOKENS_{size}",
        *(f"LLM_MAX_TOKENS{s}" for s in legacy_suffixes),
    ]
    timeout_names = [
        f"LLM_TIMEOUT_SECONDS_{size}",
        f"LLM_TIMEOUT_{size}",
        *(f"LLM_TIMEOUT_SECONDS{s}" for s in legacy_suffixes),
        *(f"LLM_TIMEOUT{s}" for s in legacy_suffixes),
    ]
    max_retries_names = [
        f"LLM_MAX_RETRIES_{size}",
        *(f"LLM_MAX_RETRIES{s}" for s in legacy_suffixes),
    ]
    reasoning_effort_names = [
        f"LLM_REASONING_EFFORT_{size}",
        *(f"LLM_REASONING_EFFORT{s}" for s in legacy_suffixes),
    ]

    return LLMModelConfig(
        model=_env_any(model_names, default_model),
        max_tokens=_int_any(max_tokens_names, "2048", 1),
        timeout_seconds=_float_any(timeout_names, "60.0", minimum=0.0),
        max_retries=_int_any(max_retries_names, "2", 0),
        reasoning_effort=_env_any(reasoning_effort_names, "") or None,
    )


def _env(name: str, default: str = "", required: bool = False) -> str:
    """Return a stripped environment value."""
    value = os.getenv(name, default).strip()
    if required and not value:
        raise ValueError(f"{name} is required")
    return value


def _env_any(names: list[str], default: str = "") -> str:
    """Return the first non-empty environment value from ``names``."""
    for name in names:
        value = _env(name, "")
        if value:
            return value
    return default


def _bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = _env(name, str(default).lower()).lower()
    if raw in TRUTHY:
        return True
    if raw in FALSY:
        return False
    raise ValueError(f"{name} must be true or false; got {raw!r}")


def _int(name: str, default: str, minimum: int | None = None) -> int:
    """Parse an integer environment variable."""
    raw = _env(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _int_any(names: list[str], default: str, minimum: int | None = None) -> int:
    """Parse the first non-empty integer environment value from ``names``."""
    for name in names:
        if _env(name, ""):
            return _int(name, default, minimum)
    return _parse_int_value(names[0], default, minimum)


def _float(name: str, default: str, minimum: float | None = None) -> float:
    """Parse a float environment variable."""
    raw = _env(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _float_any(names: list[str], default: str, minimum: float | None = None) -> float:
    """Parse the first non-empty float environment value from ``names``."""
    for name in names:
        if _env(name, ""):
            return _float(name, default, minimum)
    return _parse_float_value(names[0], default, minimum)


def _parse_int_value(name: str, raw: str, minimum: int | None) -> int:
    """Parse an integer literal for defaults."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _parse_float_value(name: str, raw: str, minimum: float | None) -> float:
    """Parse a float literal for defaults."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def config_summary(config: AppConfig) -> dict[str, Any]:
    """Return a non-secret summary useful for logs and dry runs."""
    return {
        "auth_mode": config.auth_mode,
        "has_api_key": bool(config.api_key),
        "oauth_configured": bool(
            config.oauth.token_endpoint
            and config.oauth.client_id
            and config.oauth.client_secret
        ),
        "ssl_verify": config.ssl.verify,
        "postgres_configured": bool(
            config.database.host
            and config.database.database
            and config.database.user
        ),
        "postgres_host": config.database.host,
        "postgres_database": config.database.database,
        "postgres_schema": config.database.schema,
        "llm_base_url": config.llm.base_url,
        "llm_profiles": {
            "small": {
                "model": config.llm.small.model,
                "max_tokens": config.llm.small.max_tokens,
                "timeout_seconds": config.llm.small.timeout_seconds,
            },
            "large": {
                "model": config.llm.large.model,
                "max_tokens": config.llm.large.max_tokens,
                "timeout_seconds": config.llm.large.timeout_seconds,
            },
        },
        "prompt_table_selector": {
            "model": config.prompt.model,
            "layer": config.prompt.layer,
            "name": config.prompt.default_prompt,
        },
    }

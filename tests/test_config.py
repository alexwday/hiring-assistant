"""Tests for dotenv parsing and typed config loading."""

from pathlib import Path

from utilities.config import load_config, read_env_file


def test_read_env_file_preserves_quoted_hash(tmp_path: Path):
    """It preserves quoted hashes and strips unquoted inline comments."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY='abc#123'",
                "LOG_LEVEL=DEBUG # comment",
                "EMPTY=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    values = read_env_file(env_file)

    assert values["OPENAI_API_KEY"] == "abc#123"
    assert values["LOG_LEVEL"] == "DEBUG"
    assert values["EMPTY"] == ""


def test_load_config_supports_aegis_aliases(tmp_path: Path, monkeypatch):
    """It supports Aegis-compatible environment variable aliases."""
    for name in (
        "AUTH_MODE",
        "LLM_AUTH_MODE",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_BASE_URL",
        "LLM_DEFAULT_URL",
        "LLM_MODEL",
        "LLM_MODEL_SMALL",
        "LLM_MODEL_LARGE",
        "LLM_MAX_TOKENS",
        "LLM_MAX_TOKENS_SMALL",
        "LLM_MAX_TOKENS_LARGE",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_AUTH_MODE=default\n"
        "OPENAI_API_KEY=test-key\n"
        "LLM_DEFAULT_URL=https://example.test/v1\n"
        "LLM_MODEL_SMALL=test-model\n"
        "LLM_MODEL_LARGE=large-model\n"
        "LLM_MAX_TOKENS_SMALL=99\n"
        "LLM_MAX_TOKENS_LARGE=199\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = load_config(env_file)

    assert config.auth_mode == "local"
    assert config.api_key == "test-key"
    assert config.llm.base_url == "https://example.test/v1"
    assert config.llm.small.model == "test-model"
    assert config.llm.small.max_tokens == 99
    assert config.llm.large.model == "large-model"
    assert config.llm.large.max_tokens == 199
    assert config.prompt.model == "base_llm_framework"
    assert config.prompt.layer == "default"
    assert config.prompt.default_prompt == "example"


def test_load_config_supports_rbc_vision_test_aliases(
    tmp_path: Path,
    monkeypatch,
):
    """It supports the env names used by rbc-vision-test."""
    for name in (
        "AUTH_MODE",
        "LLM_AUTH_MODE",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_BASE_URL",
        "LLM_DEFAULT_URL",
        "AZURE_BASE_URL",
        "LLM_MODEL",
        "LLM_MODEL_SMALL",
        "VISION_MODEL",
        "OAUTH_TOKEN_ENDPOINT",
        "OAUTH_ENDPOINT",
        "OAUTH_URL",
        "OAUTH_CLIENT_ID",
        "OAUTH_CLIENT_SECRET",
        "CLIENT_ID",
        "CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AUTH_MODE=oauth\n"
        "AZURE_BASE_URL=https://rbc.example.test/v1\n"
        "VISION_MODEL=gpt-5.4-mini\n"
        "OAUTH_URL=https://login.example.test/token\n"
        "CLIENT_ID=test-client\n"
        "CLIENT_SECRET=test-secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = load_config(env_file)

    assert config.auth_mode == "oauth"
    assert config.llm.base_url == "https://rbc.example.test/v1"
    assert config.llm.small.model == "gpt-5.4-mini"
    assert config.oauth.token_endpoint == "https://login.example.test/token"
    assert config.oauth.client_id == "test-client"
    assert config.oauth.client_secret == "test-secret"

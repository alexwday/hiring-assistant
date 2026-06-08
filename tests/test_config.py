from pathlib import Path

from utilities.config import load_config, read_env_file


def test_read_env_file_preserves_quoted_hash(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY='abc#123'\n"
        "LOG_LEVEL=DEBUG # comment\n"
        "EMPTY=\n",
        encoding="utf-8",
    )

    values = read_env_file(env_file)

    assert values["OPENAI_API_KEY"] == "abc#123"
    assert values["LOG_LEVEL"] == "DEBUG"
    assert values["EMPTY"] == ""


def test_load_config_supports_aegis_aliases(tmp_path: Path, monkeypatch):
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

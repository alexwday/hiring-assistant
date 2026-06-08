import pytest

from utilities.prompt_loader import validate_prompt_row


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


"""Base LLM framework entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from typing import Any

from connections.llm_connector import (
    LLMClient,
    extract_message_text,
    extract_tool_arguments,
)
from utilities.config import config_summary, load_config
from utilities.conversation import messages_from_prompt
from utilities.logging_setup import setup_logging
from utilities.prompt_loader import load_prompt
from utilities.ssl_setup import setup_ssl

logger = logging.getLogger(__name__)


def main() -> int:
    """Run setup, load a prompt, and call the configured LLM."""
    args = _parse_args()
    config = load_config(args.env)
    setup_logging(config.log_level, output_logs=config.output_logs)

    logger.info("Starting base LLM framework")
    ssl_result = setup_ssl(config)
    if not ssl_result.success:
        raise RuntimeError(ssl_result.error or "SSL setup failed")

    variables = _parse_variables(args.var)
    prompt = load_prompt(args.prompt or config.prompt.default_prompt, config=config)
    if variables:
        prompt = prompt.render(variables)
    if args.user_prompt:
        prompt = _replace_user_prompt(prompt, args.user_prompt)

    messages = messages_from_prompt(prompt)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "config": config_summary(config),
                    "ssl": {
                        "verify": ssl_result.verify,
                        "rbc_security_enabled": ssl_result.rbc_security_enabled,
                    },
                    "prompt": {
                        "name": prompt.name,
                        "model": prompt.model,
                        "layer": prompt.layer,
                        "model_size": prompt.model_size,
                        "tools": len(prompt.tools or []),
                        "tool_choice": prompt.tool_choice,
                        "settings": prompt.settings,
                    },
                    "messages": messages,
                },
                indent=2,
            )
        )
        return 0

    client = LLMClient(config=config, ssl_setup=ssl_result)
    try:
        response = client.call(
            messages=messages,
            tools=prompt.tools,
            tool_choice=prompt.tool_choice,
            settings=prompt.llm_settings(),
        )
    finally:
        client.close()

    tool_calls = extract_tool_arguments(response)
    text = extract_message_text(response)
    output: dict[str, Any] = {
        "prompt": prompt.name,
        "text": text,
        "tool_calls": tool_calls,
        "usage": response.get("usage"),
    }
    print(json.dumps(output, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run one LLM prompt.")
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to dotenv file. Defaults to .env.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt table name. Defaults to example.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Template variable for prompt rendering. May be repeated.",
    )
    parser.add_argument(
        "--user-prompt",
        default="",
        help="Override the prompt table user_prompt at runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and prompt loading without making an LLM call.",
    )
    return parser.parse_args()


def _parse_variables(raw_values: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI variables."""
    variables: dict[str, str] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError(f"Invalid --var {raw_value!r}; expected KEY=VALUE")
        key, value = raw_value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --var {raw_value!r}; key is blank")
        variables[key] = value
    return variables


def _replace_user_prompt(prompt, user_prompt: str):
    """Return a prompt copy with a runtime user prompt override."""
    return replace(prompt, user_prompt=user_prompt)


if __name__ == "__main__":
    raise SystemExit(main())

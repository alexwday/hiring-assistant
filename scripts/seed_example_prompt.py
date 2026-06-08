"""Seed the base framework example prompt into the PostgreSQL prompts table."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.config import load_config
from utilities.logging_setup import setup_logging
from utilities.prompt_loader import load_prompt, upsert_prompt

EXAMPLE_PROMPT = {
    "model": "base_llm_framework",
    "layer": "default",
    "name": "example",
    "description": "Minimal starter prompt for smoke testing the base LLM framework.",
    "version": "1.0.0",
    "model_size": "small",
    "settings": {},
    "tool_choice": None,
    "system_prompt": (
        "You are a concise assistant helping validate a reusable LLM framework."
    ),
    "user_prompt": (
        "Reply with one short sentence explaining that the framework is wired correctly."
    ),
    "tool_definition": None,
    "uses_global": [],
}

logger = logging.getLogger(__name__)


def main() -> int:
    """Seed the example prompt and verify it can be read back."""
    args = _parse_args()
    config = load_config(args.env)
    setup_logging(config.log_level, output_logs=config.output_logs)

    upsert_prompt(EXAMPLE_PROMPT, config=config)
    prompt = load_prompt("example", config=config)
    logger.info(
        "Seeded prompt: model=%s layer=%s name=%s version=%s model_size=%s",
        prompt.model,
        prompt.layer,
        prompt.name,
        prompt.version,
        prompt.model_size,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the example prompt.")
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to dotenv file. Defaults to .env.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

# Base RBC LLM Framework

Reusable starter framework for new projects that need a clean path from
configuration to prompt loading to an OpenAI-compatible LLM call.

## Structure

```text
connections/
  oauth_connector.py   # OAuth client-credentials token lifecycle
  llm_connector.py     # OpenAI-compatible chat client

utilities/
  config.py            # .env parsing and typed config
  ssl_setup.py         # RBC/system certificate setup
  prompt_loader.py     # PostgreSQL prompts table loading
  conversation.py      # message validation and history trimming
  logging_setup.py     # redacted local logging

main.py                # setup -> auth -> prompt -> LLM call
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then validate without external calls:

```bash
python -m scripts.seed_example_prompt
python main.py --dry-run
```

Run the example prompt:

```bash
python main.py --prompt example
```

## Prompt Storage

Prompts are loaded from PostgreSQL table `prompts`; there is no local YAML
fallback. The base framework reads:

```text
model = base_llm_framework
layer = default
name  = example
```

The table follows the Aegis prompt schema:

```sql
model TEXT NOT NULL
layer TEXT
name TEXT NOT NULL
description TEXT
comments TEXT
system_prompt TEXT
user_prompt TEXT
tool_definition JSONB
uses_global TEXT[]
version TEXT
created_at TIMESTAMPTZ
updated_at TIMESTAMPTZ
```

The seed script does not create this table. It verifies the table exists and
then inserts or updates the example prompt row. Create the table with a
database owner or migration account before running the seed script.

Prompt metadata such as `model_size` is stored as JSON in `comments`:

```json
{
  "model_size": "small",
  "settings": {
    "max_tokens": 400
  },
  "tool_choice": "auto"
}
```

Model profiles are defined in `.env`:

```bash
LLM_MODEL_SMALL=gpt-5.4
LLM_MAX_TOKENS_SMALL=2048

LLM_MODEL_LARGE=gpt-5.4
LLM_MAX_TOKENS_LARGE=4096
```

Render variables from the CLI:

```bash
python main.py --prompt example --var topic="RBC LLM adoption"
```

## Auth Modes

`AUTH_MODE=local` uses `OPENAI_API_KEY` or `API_KEY`.

`AUTH_MODE=oauth` uses `OAUTH_ENDPOINT`, `OAUTH_CLIENT_ID`,
`OAUTH_CLIENT_SECRET`, and `OAUTH_GRANT_TYPE`.

The config loader also accepts common Aegis aliases such as
`LLM_AUTH_MODE=default|oauth`, `LLM_DEFAULT_URL`, and legacy unsuffixed
`LLM_MODEL` values for the small profile.

## SSL

Set `SSL_VERIFY=true` for internal environments. The setup order is:

1. Try `rbc_security.enable_certs()`.
2. Warn if `rbc_security` is unavailable.
3. Fall back to system certificates.

Set `SSL_VERIFY=false` for local development against public endpoints.

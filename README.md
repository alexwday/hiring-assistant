# Hiring Assistant

Local resume screening app built on the RBC LLM framework. It runs as a local
HTML server, stores all uploaded and generated files inside this project, and
uses the existing OpenAI-compatible auth/base URL/SSL configuration so it works
both locally and in work environments with OAuth, custom endpoints, and custom
certificate handling.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

By default the app starts at:

```text
http://127.0.0.1:8000
```

The browser opens automatically unless `APP_OPEN_BROWSER=false` is set or
`--no-open-browser` is passed.

The interview-time assistant is available at:

```text
http://127.0.0.1:8000/interview
```

There is also a native Python interview app:

```bash
python interview_app.py
```

## Workflow

1. Create a project for one posting.
2. Paste the job posting and optional work/context notes.
3. Upload one or many PDF resumes.
4. Send uploaded resumes through local PII review.
5. Finalize redactions manually, or auto-redact all locally detected PII.
6. Process redacted resumes into LLM-readable markdown with non-contact metadata.
7. Select processed resumes for job-fit review.
8. Review the generated hiring-manager report and prescreen email template.

## Browser Interview Assistant

The separate `/interview` app lets you paste or upload a job posting, optional
context notes, and one candidate resume. It generates large-format cue cards for
openers, topics, follow-ups, transitions, watchouts, and closing prompts.

For live interviews, prefer the native app below. Browser audio capture requires
browser media permissions and cannot reliably capture system meeting audio
without a screen/tab sharing picker.

## Native Interview App

`python interview_app.py` launches a PySide6 desktop version of the same
interview assistant. It uses `sounddevice` to capture selected local input
devices, mixes them into 16 kHz mono audio, and transcribes locally with
`faster-whisper`. No audio is sent to the OpenAI transcription endpoint.

On first start, the app downloads the configured local Whisper model into:

```text
data/models/faster-whisper/
```

The default model is `small.en` for better accuracy. If the work computer feels
too slow, set `WHISPER_MODEL=base.en` for lower latency. Configure local
transcription with:

```bash
WHISPER_MODEL=small.en
WHISPER_MODEL_DIR=data/models/faster-whisper
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
WHISPER_PARTIAL_INTERVAL_SECONDS=1.0
WHISPER_STABLE_WINDOW_SECONDS=6.0
WHISPER_MAX_WINDOW_SECONDS=10.0
WHISPER_VAD_MIN_SILENCE_MS=300
```

The generated interview board stays fixed during the interview. Live refreshes
only update the top `Next 3 paths` cards and use the configured small chat model
(`LLM_MODEL_SMALL`, default `gpt-5.4-mini`). You can cap live response size with
`LLM_LIVE_SUGGESTION_MAX_TOKENS` when tuning latency.

For reliable Webex capture on macOS, route Webex speaker output into a virtual
audio input such as BlackHole or Loopback, then select:

- `Mic`: your microphone
- `Meeting audio`: the virtual Webex/BlackHole/Loopback input

If you already have one mixed input device containing both mic and meeting audio,
select it as `Mic` and leave `Meeting audio` as `None`.

Before an interview on your work computer:

1. Run `python interview_app.py`.
2. Load the job posting, context notes, and resume.
3. Click `Generate board`.
4. Select your mic and the Webex loopback input.
5. Click `Start` and speak for 10 seconds.
6. Confirm the partial transcript updates quickly and the `Next 3 paths` cards
   refresh without clearing the board.

Each project keeps its own uploaded, processed, and reviewed resumes. A resume is
only in one table at a time:

- `uploaded`: PDF is stored locally but has not been reviewed for PII.
- `pii_review`: local PII boxes are ready for human review.
- `redacted`: a finalized redacted PDF exists and is ready for LLM processing.
- `processing`: redacted pages are being processed by the vision model.
- `processed`: resume markdown and candidate metadata exist.
- `reviewing`: a job-fit report is being generated.
- `reviewed`: job-fit report and scores exist.

## Local Storage

Runtime files are stored under ignored `data/` directories:

```text
data/projects/
  index.json
  <project-id>/
    project.json
    documents/
      YYYY-MM-DD/
        uploaded/
        pii-pages/
        redacted/
        redacted-pages/
        pages/
        processed/
        reviewed/
```

No database is required. Do not commit `data/`; it contains resumes and review
outputs.

## PII Redaction

The app does not send the uploaded original PDF to the external LLM. Processing
first renders the PDF locally and runs local PyMuPDF/PyMuPDF4LLM-based text box
detection for the redacted PII categories:

- candidate names
- email addresses
- phone numbers
- web/profile URLs and personal links

The app intentionally does not auto-redact street addresses, city/location text,
or postal codes. The PII review page lets a human finalize detected boxes and
draw additional manual boxes. The project table also has an `Auto-redact
selected` action that applies every local detection without opening the review
page. After finalization, the app saves a redacted PDF, deletes the original
uploaded PDF for that document, and sends only redacted page images to the
vision model.

## Reset Local Data

All runtime data can be cleared from the UI with `Reset local data`, or on
startup:

```bash
python main.py --reset-data
```

The same behavior can be enabled through `.env`:

```bash
APP_RESET_DATA_ON_STARTUP=true
```

## LLM Configuration

The app uses the copied framework's `.env` settings:

```bash
AUTH_MODE=local
OPENAI_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_SMALL=gpt-5.4-mini
LLM_MAX_TOKENS_SMALL=32768
SSL_VERIFY=false
```

For OAuth/internal environments, set:

```bash
AUTH_MODE=oauth
OAUTH_ENDPOINT=
OAUTH_CLIENT_ID=
OAUTH_CLIENT_SECRET=
LLM_BASE_URL=
SSL_VERIFY=true
```

The small model profile is used for resume vision extraction, metadata
extraction, candidate review, and final top-10 reranking. Final reranking uses
`LLM_MAX_TOKENS_SMALL` as the model's completion-token limit, so keep it high
enough for the larger top-10 comparative prompt.

## LLM Concurrency

Selected resume processing and review batches run LLM calls in parallel. The
default is tuned for up to 8 simultaneous LLM calls across the whole local app:

```bash
APP_MAX_PARALLEL_LLM_CALLS=8
APP_PROCESS_WORKERS=8
APP_REVIEW_WORKERS=8
APP_PAGE_WORKERS=8
```

`APP_MAX_PARALLEL_LLM_CALLS` is the global cap. The worker values control how
many documents or resume pages can be queued at once, but the global cap keeps
the actual API call count bounded.

## PDF Rendering

Resume PDFs are rendered page by page using `pdftoppm` from poppler before being
sent to the vision model. On macOS with Homebrew:

```bash
brew install poppler
```

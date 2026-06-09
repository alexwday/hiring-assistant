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

## Workflow

1. Create a project for one posting.
2. Paste the job posting and optional work/context notes.
3. Upload one or many PDF resumes.
4. Send uploaded resumes through local PII review.
5. Finalize redactions so only redacted page images are sent to the LLM.
6. Process redacted resumes into LLM-readable markdown with non-contact metadata.
7. Select processed resumes for job-fit review.
8. Review the generated hiring-manager report and prescreen email template.

Each project keeps its own uploaded, processed, and reviewed resumes. A resume is
only in one table at a time:

- `uploaded`: PDF is stored locally but has not been reviewed for PII.
- `pii_review`: local PII boxes are ready for human review.
- `redacted`: a finalized redacted PDF exists and is ready for LLM processing.
- `processed`: resume markdown and candidate metadata exist.
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
detection for likely contact PII:

- email addresses
- phone numbers
- LinkedIn/profile URLs and personal links
- postal/ZIP codes
- address or location lines

The PII review page lets a human finalize detected boxes and draw additional
manual boxes. After finalization, the app saves a redacted PDF, deletes the
original uploaded PDF for that document, and sends only redacted page images to
the vision model. Candidate name is retained.

## LLM Configuration

The app uses the copied framework's `.env` settings:

```bash
AUTH_MODE=local
OPENAI_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_SMALL=gpt-5.4-mini
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
extraction, and candidate review.

## PDF Rendering

Resume PDFs are rendered page by page using `pdftoppm` from poppler before being
sent to the vision model. On macOS with Homebrew:

```bash
brew install poppler
```

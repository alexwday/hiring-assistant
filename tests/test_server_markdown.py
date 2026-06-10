"""Tests for rendered markdown document views."""

from types import SimpleNamespace

from hiring_assistant import server as server_module
from hiring_assistant.server import (
    HiringAssistantHandler,
    _backfill_candidate_name_hints,
    _document_label,
    _interview_context_from_form,
    _markdown_to_html,
    _privacy_filename,
)
from hiring_assistant.store import ProjectStore, UploadedFile


def test_markdown_renderer_formats_common_resume_sections():
    """It renders headings, bold text, bullets, and numbered items."""
    html = _markdown_to_html(
        "# Candidate Review\n\n"
        "## Scores\n"
        "- Education: **8/10**\n"
        "- Experience: 9/10\n\n"
        "1) Describe the project tradeoffs.\n"
    )

    assert "<h1>Candidate Review</h1>" in html
    assert "<h2>Scores</h2>" in html
    assert "<li>Education: <strong>8/10</strong></li>" in html
    assert "<ol>" in html


def test_markdown_renderer_escapes_raw_html():
    """It treats generated markdown as untrusted text."""
    html = _markdown_to_html("# Safe\n\n<script>alert('x')</script>")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_privacy_filename_excludes_local_candidate_name():
    """It keeps UI labels useful without sending names as filenames."""
    document = {
        "id": "abc123def456",
        "candidate_id": "C001",
        "candidate_name_hint": "Jordan Lee",
    }

    assert _document_label(document) == "Jordan Lee | Candidate C001"
    assert _privacy_filename(document) == "candidate-c001.pdf"


def test_redaction_preserves_local_candidate_name_hint(tmp_path, monkeypatch):
    """It keeps package-extracted names local after PII finalization."""
    store = ProjectStore(tmp_path / "data")
    project = store.create_project("GG08 Posting")
    document = store.add_uploads(
        project["id"],
        [UploadedFile("c001-jordan-lee.pdf", b"%PDF-1.4 test")],
    )[0]
    store.update_document(
        project["id"],
        document["id"],
        status="pii_review",
        candidate_id="C001",
        candidate_name_hint="Jordan Lee",
    )

    def fake_apply_redactions(source_pdf, target_pdf, redactions):
        target_pdf.parent.mkdir(parents=True, exist_ok=True)
        target_pdf.write_bytes(source_pdf.read_bytes())

    def fake_render_pdf_pages(pdf_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        page_path = output_dir / "page-001.png"
        page_path.write_bytes(b"png")
        return [page_path]

    monkeypatch.setattr(server_module, "apply_redactions", fake_apply_redactions)
    monkeypatch.setattr(server_module, "render_pdf_pages", fake_render_pdf_pages)
    monkeypatch.setattr(server_module, "delete_paths", lambda paths: None)

    handler = object.__new__(HiringAssistantHandler)
    handler.server = SimpleNamespace(store=store)
    handler._apply_document_redactions(project["id"], document["id"], [])

    updated_project = store.load_project(project["id"])
    updated = store.get_document(updated_project, document["id"])

    assert updated["status"] == "redacted"
    assert updated["candidate_name_hint"] == "Jordan Lee"
    assert updated["original_filename"] == "candidate-c001.pdf"


def test_backfill_candidate_name_hint_from_package_rows(tmp_path):
    """It repairs older finalized documents that had local names cleared."""
    store = ProjectStore(tmp_path / "data")
    project = store.create_project("GG08 Posting")
    document = store.add_uploads(
        project["id"],
        [UploadedFile("candidate-c001.pdf", b"%PDF-1.4 test")],
    )[0]
    project = store.load_project(project["id"])
    project["packages"] = [
        {
            "id": "package-1",
            "candidate_rows": [
                {"candidate_id": "C001", "candidate_name": "Jordan Lee"}
            ],
        }
    ]
    store.save_project(project)
    store.update_document(
        project["id"],
        document["id"],
        candidate_id="C001",
        source_package_id="package-1",
        candidate_name_hint="",
    )

    repaired = _backfill_candidate_name_hints(
        store,
        project["id"],
        store.load_project(project["id"]),
    )
    updated = store.get_document(repaired, document["id"])

    assert updated["candidate_name_hint"] == "Jordan Lee"
    assert _document_label(updated) == "Jordan Lee | Candidate C001"


def test_interview_context_uses_upload_field_names():
    """It keeps job, context, and resume uploads separated by form field."""
    context = _interview_context_from_form(
        {
            "job_posting": ["Posting text"],
            "work_context": ["Context text"],
            "resume_text": ["Resume text"],
        },
        [
            UploadedFile("job.md", b"Job file", field_name="job_file"),
            UploadedFile("context.txt", b"Context file", field_name="context_file"),
            UploadedFile("resume.md", b"Resume file", field_name="resume_file"),
        ],
    )

    assert context.job_posting == "Posting text\n\nJob file"
    assert context.work_context == "Context text\n\nContext file"
    assert context.resume_text == "Resume text\n\nResume file"

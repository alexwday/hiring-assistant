"""Tests for local PII detection and redaction."""

from pathlib import Path

import fitz

from hiring_assistant.pii_redaction import apply_redactions, detect_pii_boxes
from scripts.create_mock_resume_pdf import PAGES, build_pdf


def test_detect_pii_boxes_finds_resume_contact_details(tmp_path):
    """It finds contact/location lines without storing matched PII text."""
    pdf_path = _write_mock_pdf(tmp_path)

    result = detect_pii_boxes(pdf_path)
    detections = result["detections"]

    assert detections
    assert any(item["label"] == "Contact line" for item in detections)
    assert all("jordan.lee@example.com" not in str(item) for item in detections)
    assert all("416-555-0198" not in str(item) for item in detections)


def test_apply_redactions_removes_contact_details_from_pdf(tmp_path):
    """It writes a redacted PDF whose text no longer contains contact PII."""
    pdf_path = _write_mock_pdf(tmp_path)
    result = detect_pii_boxes(pdf_path)
    target_path = tmp_path / "redacted.pdf"

    apply_redactions(pdf_path, target_path, result["detections"])
    text = _pdf_text(target_path)

    assert target_path.exists()
    assert "JORDAN LEE" in text
    assert "jordan.lee@example.com" not in text
    assert "416-555-0198" not in text
    assert "linkedin.com" not in text
    assert "Toronto, ON" not in text


def _write_mock_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "resume.pdf"
    path.write_bytes(build_pdf(PAGES))
    return path


def _pdf_text(path: Path) -> str:
    document = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()

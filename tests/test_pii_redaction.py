"""Tests for local PII detection and redaction."""

from pathlib import Path

import fitz

from hiring_assistant.pii_redaction import apply_redactions, detect_pii_boxes
from scripts.create_mock_resume_pdf import PAGES, build_pdf


def test_detect_pii_boxes_finds_resume_contact_details(tmp_path):
    """It finds names, emails, phones, and URLs without storing matched text."""
    pdf_path = _write_mock_pdf(tmp_path)

    result = detect_pii_boxes(pdf_path)
    detections = result["detections"]
    labels = {item["label"] for item in detections}

    assert detections
    assert "Candidate name" in labels
    assert "Email" in labels
    assert "Phone" in labels
    assert "Web address" in labels
    assert "Postal code" not in labels
    assert "Address or location" not in labels
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
    assert "JORDAN LEE" not in text
    assert "jordan.lee@example.com" not in text
    assert "416-555-0198" not in text
    assert "linkedin.com" not in text
    assert "Toronto, ON" in text


def test_redaction_keeps_address_and_postal_code_when_mixed_with_contact_pii(
    tmp_path,
):
    """It only redacts name, email, phone, and URL in mixed contact lines."""
    pdf_path = tmp_path / "challenging.pdf"
    pdf_path.write_bytes(
        build_pdf(
            [
                [
                    "ALEX MORGAN-SMITH",
                    "123 Main Street, Toronto, ON M5V 2T6 | "
                    "alex.morgan-smith+ops@example.co.uk",
                    "Phone: +1 (416) 555-0198 ext 42 | "
                    "https://alex.example.dev/case-studies",
                    "LinkedIn: www.linkedin.com/in/alex-morgan-smith",
                    "SUMMARY",
                    "Built workflow automation and reporting tools.",
                ]
            ]
        )
    )

    result = detect_pii_boxes(pdf_path)
    labels = {item["label"] for item in result["detections"]}
    target_path = tmp_path / "challenging-redacted.pdf"

    apply_redactions(pdf_path, target_path, result["detections"])
    text = _pdf_text(target_path)

    assert labels == {"Candidate name", "Email", "Phone", "Web address"}
    assert "ALEX MORGAN-SMITH" not in text
    assert "alex.morgan-smith+ops@example.co.uk" not in text
    assert "416" not in text
    assert "alex.example.dev" not in text
    assert "linkedin.com" not in text
    assert "123 Main Street" in text
    assert "Toronto, ON" in text
    assert "M5V 2T6" in text


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

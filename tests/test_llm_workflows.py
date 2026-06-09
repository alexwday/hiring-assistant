"""Tests for resume LLM workflow helpers."""

from hiring_assistant.llm_workflows import _remove_pii_metadata


def test_remove_pii_metadata_strips_candidate_name_and_contact_fields():
    """It strips name/contact/location fields from stored LLM payloads."""
    cleaned = _remove_pii_metadata(
        {
            "candidate_name": "Jordan Lee",
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "phone": "416-555-0198",
            "website": "https://example.com",
            "current_or_recent_title": "Workflow Analyst",
        }
    )

    assert cleaned == {"current_or_recent_title": "Workflow Analyst"}

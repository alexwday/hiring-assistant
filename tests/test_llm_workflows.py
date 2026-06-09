"""Tests for resume LLM workflow helpers."""

from hiring_assistant.llm_workflows import (
    _normalize_review_payload,
    _remove_pii_metadata,
    render_review_markdown,
)


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


def test_review_payload_normalizes_decimal_scores_and_canada_flag():
    """It keeps one-decimal scoring and boolean-only Canada location signals."""
    payload = _normalize_review_payload(
        {
            "education_score": "9.74",
            "experience_score": 8,
            "projects_score": "7.25",
            "aggregate_score": "9.66",
            "holistic_score": 9,
            "located_in_canada": "yes",
            "relevant_projects": [{"name": "Pipeline", "score": "8.88"}],
        }
    )

    assert payload["education_score"] == 9.7
    assert payload["experience_score"] == 8.0
    assert payload["projects_score"] == 7.3
    assert payload["aggregate_score"] == 9.7
    assert payload["holistic_score"] == 9.0
    assert payload["screening_average"] == 9.4
    assert payload["located_in_canada"] is True
    assert payload["relevant_projects"][0]["score"] == 8.9


def test_review_markdown_renders_one_decimal_scores():
    """It preserves visible decimal precision in generated reports."""
    markdown = render_review_markdown(
        {
            "education_score": 9,
            "experience_score": 8.44,
            "projects_score": 7.95,
            "aggregate_score": 9.7,
            "holistic_score": 9,
            "screening_average": 9.35,
            "located_in_canada": True,
            "tradeoff_analysis": "Strong execution, lighter formal credentials.",
            "education_entries": [
                {
                    "university": "Example University",
                    "level": "Bachelor",
                    "completion": "graduated",
                    "program": "Information Systems",
                    "gpa": "3.8",
                    "fit_summary": "Useful applied systems background.",
                }
            ],
            "work_experience_fit_bullets": ["Led workflow intake redesign."],
            "hiring_manager_notes": ["This should not render."],
            "recommendation": "Advance to prescreen",
        }
    )

    assert "- Education: 9.0/10" in markdown
    assert "- Work experience: 8.4/10" in markdown
    assert "- Relevant projects: 8.0/10" in markdown
    assert "- Screening average: 9.4/10" in markdown
    assert "- Located in Canada: Yes" in markdown
    assert (
        "### Example University | Bachelor graduated | "
        "Information Systems | GPA: 3.8"
    ) in markdown
    assert "- Led workflow intake redesign." in markdown
    assert "Hiring Manager Notes" not in markdown
    assert "This should not render." not in markdown

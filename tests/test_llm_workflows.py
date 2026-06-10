"""Tests for resume LLM workflow helpers."""

from types import SimpleNamespace

import pytest

from hiring_assistant.llm_workflows import (
    ResumeLLMService,
    _normalize_review_payload,
    _remove_pii_metadata,
    parse_json_response,
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


def test_review_payload_normalizes_formula_scores_and_canada_flag():
    """It keeps formula scoring and boolean-only Canada location signals."""
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
    assert payload["aggregate_score"] == 8.2
    assert payload["holistic_score"] == 9.0
    assert payload["screening_average"] == 8.6
    assert payload["located_in_canada"] is True
    assert payload["relevant_projects"][0]["score"] == 8.9
    assert payload["score_rationale"]["aggregate_formula"] == (
        "Computed by app as 25% education (9.7/10) + 45% experience "
        "(8.0/10) + 30% projects (7.3/10) = 8.2/10."
    )


def test_review_payload_bounds_holistic_adjustments():
    """It keeps holistic adjustments within the configured scoring range."""
    payload = _normalize_review_payload(
        {
            "education_score": 6,
            "experience_score": 6,
            "projects_score": 6,
            "aggregate_score": 6,
            "holistic_score": 10,
        }
    )

    assert payload["aggregate_score"] == 6.0
    assert payload["holistic_score"] == 7.5
    assert payload["screening_average"] == 6.8
    assert "bounded holistic_score" in (
        payload["score_rationale"]["calibration_notes"]
    )


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
            "score_rationale": {
                "aggregate_formula": "0.25*9.0 + 0.45*8.4 + 0.30*8.0.",
                "holistic_adjustments": "Raised for production delivery.",
                "calibration_notes": "Needs more detail on LLM validation.",
            },
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
    assert "## Score Rationale" in markdown
    assert "Aggregate formula" in markdown
    assert "Raised for production delivery" in markdown
    assert (
        "### Example University | Bachelor graduated | "
        "Information Systems | GPA: 3.8"
    ) in markdown
    assert "- Led workflow intake redesign." in markdown
    assert "Hiring Manager Notes" not in markdown
    assert "This should not render." not in markdown


def test_parse_json_response_reports_empty_llm_content():
    """It turns blank model output into an actionable workflow error."""
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": ""},
            }
        ],
        "usage": {"prompt_tokens": 12000, "completion_tokens": 5000},
    }

    with pytest.raises(ValueError) as exc_info:
        parse_json_response(response, "final rerank")

    message = str(exc_info.value)
    assert "final rerank returned an empty LLM response" in message
    assert "finish_reason=length" in message
    assert "completion_tokens=5000" in message


def test_review_payload_uses_profile_token_limit(monkeypatch):
    """It does not override LLM_MAX_TOKENS_SMALL with a smaller review cap."""
    captured = {}

    def fake_guarded_call(client, messages, settings, **kwargs):
        captured["settings"] = settings
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"education_score": 6, "experience_score": 6, '
                        '"projects_score": 6, "holistic_score": 6}',
                    },
                }
            ]
        }

    monkeypatch.setattr(
        "hiring_assistant.llm_workflows._guarded_llm_call",
        fake_guarded_call,
    )
    service = ResumeLLMService(
        config=SimpleNamespace(),
        ssl_setup=SimpleNamespace(),
    )

    service._review_payload(
        client=SimpleNamespace(),
        resume_markdown="## Resume",
        metadata={},
        job_posting="Posting",
        work_context="Context",
    )

    assert captured["settings"] == {
        "model_size": "small",
        "response_format": {"type": "json_object"},
    }

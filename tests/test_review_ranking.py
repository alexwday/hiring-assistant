"""Tests for project-level reviewed resume ranking."""

from hiring_assistant.server import (
    ADVANCE_RECOMMENDATION,
    HOLD_RECOMMENDATION,
    _rank_reviewed_documents,
    _review_payload_to_html,
    _reviewed_table,
    _reviewed_export_html,
)
from hiring_assistant.store import ProjectStore


def _reviewed_doc(
    document_id: str,
    aggregate: float,
    holistic: float,
    located_in_canada: bool = False,
) -> dict:
    return {
        "id": document_id,
        "candidate_id": document_id.upper(),
        "status": "reviewed",
        "reviewed_at": f"2026-06-09T12:00:0{document_id}+00:00",
        "scores": {
            "aggregate": aggregate,
            "holistic": holistic,
            "located_in_canada": located_in_canada,
            "recommendation": "Model recommendation",
        },
    }


def test_reviewed_documents_rank_top_six_by_aggregate_holistic_average():
    """It assigns advance only to the top six reviewed resumes."""
    documents = [
        _reviewed_doc("a", 7.0, 7.0),
        _reviewed_doc("b", 9.7, 9.7),
        _reviewed_doc("c", 9.2, 9.4),
        _reviewed_doc("d", 8.8, 8.8),
        _reviewed_doc("e", 8.5, 8.7),
        _reviewed_doc("f", 8.2, 8.1),
        _reviewed_doc("g", 8.0, 8.0),
    ]

    ranked = _rank_reviewed_documents(documents)

    assert [item["document"]["id"] for item in ranked] == [
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "a",
    ]
    assert [item["recommendation"] for item in ranked[:6]] == [
        ADVANCE_RECOMMENDATION,
    ] * 6
    assert ranked[6]["recommendation"] == HOLD_RECOMMENDATION


def test_reviewed_table_renders_precise_scores_recommendations_and_canada_flag():
    """It renders one-decimal scores and ignores stale model recommendations."""
    documents = [
        _reviewed_doc("a", 7.0, 7.0),
        _reviewed_doc("b", 9.7, 9.7, located_in_canada=True),
        _reviewed_doc("c", 9.2, 9.4),
        _reviewed_doc("d", 8.8, 8.8),
        _reviewed_doc("e", 8.5, 8.7),
        _reviewed_doc("f", 8.2, 8.1),
        _reviewed_doc("g", 8.0, 8.0),
    ]

    html = _reviewed_table("project-1", {}, documents)

    assert html.count(ADVANCE_RECOMMENDATION) == 6
    assert html.count(HOLD_RECOMMENDATION) == 1
    assert "Model recommendation" not in html
    assert "9.7" in html
    assert "7.0" in html
    assert "&#x1F1E8;&#x1F1E6;" in html


def test_review_payload_html_renders_dashboard_and_compact_sections():
    """It renders report JSON as dashboard-oriented HTML, not raw markdown."""
    html = _review_payload_to_html(
        {
            "education_score": 9.1,
            "experience_score": 8.7,
            "projects_score": 9.4,
            "aggregate_score": 9.2,
            "holistic_score": 9.6,
            "screening_average": 9.4,
            "project_rank": 2,
            "located_in_canada": True,
            "recommendation": ADVANCE_RECOMMENDATION,
            "tradeoff_analysis": "Strong project evidence offsets a lighter GPA.",
            "education_entries": [
                {
                    "university": "Example University",
                    "level": "Bachelor",
                    "completion": "graduated",
                    "program": "Analytics",
                    "gpa": "3.5",
                    "fit_summary": "Relevant quantitative foundation.",
                }
            ],
            "work_experience_fit_bullets": ["Owned reporting workflow cleanup."],
            "relevant_projects": [
                {
                    "name": "Queue Dashboard",
                    "summary": "Built intake visibility.",
                    "role_relevance": "Relevant to operational triage.",
                }
            ],
            "unique_standouts": [
                {
                    "signal": "Self-taught automation",
                    "why_it_matters": "Can improve manual workflows.",
                    "confidence": "high",
                }
            ],
            "gaps_and_risks": [
                {
                    "gap": "Limited formal PM title",
                    "screening_follow_up": "Ask about ownership boundaries.",
                    "severity": "medium",
                }
            ],
            "prescreen_email_subject": "Follow-up questions",
            "prescreen_email_body": "Please describe the dashboard tradeoffs.",
            "hiring_manager_notes": ["Do not render this."],
        }
    )

    assert "score-dashboard" in html
    assert html.index("Tradeoff Analysis") < html.index("Education Fit")
    assert "Example University | Bachelor graduated | Analytics | GPA: 3.5" in html
    assert "<li>Owned reporting workflow cleanup.</li>" in html
    assert "Queue Dashboard" in html
    assert "standout-risk-table" in html
    assert "Hiring Manager Notes" not in html
    assert "Do not render this." not in html


def test_reviewed_export_html_has_expandable_rows_and_links(tmp_path):
    """It creates a project-level HTML export with expandable candidate rows."""
    store = ProjectStore(tmp_path / "data")
    project = {"id": "project-1", "name": "GG08 Analyst"}
    documents = [_reviewed_doc("b", 9.7, 9.7, located_in_canada=True)]
    project["final_rerank"] = {
        "payload": {
            "adjusted_rankings": [
                {
                    "candidate_key": "b",
                    "adjusted_rank": 1,
                    "decision_summary": (
                        "Best overall fit because the resume combines "
                        "production delivery with practical analytics judgment."
                    ),
                }
            ]
        }
    }

    html = _reviewed_export_html("project-1", project, documents, store)

    assert "Reviewed Results Export" in html
    assert "<details class=\"export-details\">" in html
    assert "reviewed-export-scroll" in html
    assert "/projects/project-1/files/b" in html
    assert "/projects/project-1/documents/b/review" in html
    assert "/projects/project-1/documents/b/markdown" in html
    assert "/projects/project-1/exports/reviewed?download=1" in html
    assert "Final thesis" in html
    assert "Best overall fit because" in html


def test_static_reviewed_export_embeds_top_ten_redacted_pdf(tmp_path):
    """It embeds redacted PDFs for top-10 candidates in the static export."""
    store = ProjectStore(tmp_path / "data")
    project = {"id": "project-1", "name": "GG08 Analyst"}
    pdf_path = store.project_dir("project-1") / "documents" / "b-redacted.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n% test redacted pdf\n")
    document = _reviewed_doc("b", 9.7, 9.7, located_in_canada=True)
    document["redacted_pdf_path"] = "documents/b-redacted.pdf"

    html = _reviewed_export_html("project-1", project, [document], store, static=True)

    assert "data:application/pdf;base64," in html
    assert "pdf-pane" in html
    assert "reviewed-export-scroll" in html
    assert "/projects/" not in html

"""Tests for project-level reviewed resume ranking."""

from hiring_assistant.server import (
    ADVANCE_RECOMMENDATION,
    HOLD_RECOMMENDATION,
    _rank_reviewed_documents,
    _reviewed_table,
)


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

    html = _reviewed_table("project-1", documents)

    assert html.count(ADVANCE_RECOMMENDATION) == 6
    assert html.count(HOLD_RECOMMENDATION) == 1
    assert "Model recommendation" not in html
    assert "9.7" in html
    assert "7.0" in html
    assert "&#x1F1E8;&#x1F1E6;" in html

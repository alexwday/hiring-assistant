"""Tests for live interview assistant workflow helpers."""

from hiring_assistant.interview_workflows import (
    normalize_interview_payload,
    normalize_next_paths_payload,
)


def test_interview_payload_normalizes_cards_and_named_sections():
    """It turns varied model JSON shapes into display cards."""
    payload = normalize_interview_payload(
        {
            "summary": "Start with the candidate's strongest project.",
            "opening": ["Walk me through the work you are proudest of."],
            "priority_topics": [
                {
                    "topic": "Data pipelines",
                    "prompt": "Ask how they designed pipeline reliability.",
                    "follow_ups": ["What failed?", "How did they monitor it?"],
                }
            ],
            "top_suggestions": [
                {"title": "Clarify scope", "text": "What was your exact role?"}
            ],
            "signals": ["Mentioned stakeholder handoffs"],
        }
    )

    assert payload["summary"] == "Start with the candidate's strongest project."
    assert payload["signals"] == ["Mentioned stakeholder handoffs"]
    assert payload["cards"][0] == {
        "kind": "opening",
        "title": "Opening",
        "text": "Walk me through the work you are proudest of.",
        "detail": "",
    }
    assert payload["cards"][1]["kind"] == "topic"
    assert payload["cards"][1]["title"] == "Data pipelines"
    assert payload["cards"][1]["detail"] == (
        "What failed? / How did they monitor it?"
    )
    assert payload["cards"][2]["kind"] == "live"


def test_interview_payload_dedupes_repeated_card_text():
    """It removes repeated prompts while preserving first occurrence order."""
    payload = normalize_interview_payload(
        {
            "opening": ["Tell me about your current role."],
            "follow_ups": ["Tell me about your current role.", "What changed?"],
        }
    )

    assert [card["text"] for card in payload["cards"]] == [
        "Tell me about your current role.",
        "What changed?",
    ]


def test_next_paths_payload_normalizes_exactly_three_cards():
    """It turns compact live suggestions into exactly three visible paths."""
    payload = normalize_next_paths_payload(
        {
            "summary": "Steer toward implementation detail.",
            "paths": [
                {
                    "title": "Clarify scope",
                    "prompt": "What part did you personally own?",
                    "why": "They described team results.",
                },
                {
                    "title": "Go deeper",
                    "prompt": "What failed during rollout?",
                    "why": "Tests practical judgment.",
                },
            ],
            "signals": ["Mentioned a migration"],
        }
    )

    assert payload["summary"] == "Steer toward implementation detail."
    assert payload["signals"] == ["Mentioned a migration"]
    assert len(payload["paths"]) == 3
    assert payload["paths"][0] == {
        "kind": "live",
        "title": "Clarify scope",
        "text": "What part did you personally own?",
        "detail": "They described team results.",
    }
    assert payload["paths"][2]["title"] == "Transition"

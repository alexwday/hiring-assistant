"""Tests for the native desktop interview assistant helpers."""

import numpy as np

from hiring_assistant.interview_desktop import (
    DEFAULT_WHISPER_MODEL_DIR,
    LiveSuggestionState,
    TranscriptStabilizer,
    WHISPER_SAMPLE_RATE,
    _float32_mono_16k,
    _initial_live_paths,
    whisper_model_dir,
)


def test_whisper_model_dir_defaults_to_project_data(monkeypatch):
    """It stores local Whisper model files under the project data directory."""
    monkeypatch.delenv("WHISPER_MODEL_DIR", raising=False)

    assert whisper_model_dir() == DEFAULT_WHISPER_MODEL_DIR


def test_float32_mono_16k_downmixes_and_resamples():
    """It converts stereo int16 frames into mono 16 kHz float audio."""
    stereo = np.array(
        [
            [1000, -1000],
            [3000, 1000],
            [-3000, -1000],
            [0, 2000],
        ],
        dtype=np.int16,
    )

    samples = _float32_mono_16k(stereo, source_rate=48000)

    assert len(samples) == 1
    assert samples.dtype == np.dtype("float32")
    assert WHISPER_SAMPLE_RATE == 16000
    assert -1.0 <= float(samples[0]) <= 1.0


def test_transcript_stabilizer_commits_repeated_prefix_only():
    """It keeps fresh decode text partial until consecutive decodes agree."""
    stabilizer = TranscriptStabilizer()

    first = stabilizer.update("we built the platform")
    second = stabilizer.update("we built the platform for hiring")
    third = stabilizer.update("we built the platform for hiring today")

    assert first.stable_text == ""
    assert first.partial_text == "we built the platform"
    assert second.stable_text == "we built the platform"
    assert second.partial_text == "for hiring"
    assert third.stable_text == "for hiring"
    assert third.partial_text == "today"


def test_transcript_stabilizer_force_commits_tentative_text():
    """It can commit the current partial text when audio pauses or stops."""
    stabilizer = TranscriptStabilizer()
    stabilizer.update("walk me through the migration")

    update = stabilizer.force_commit(pause=True)

    assert update.stable_text == "walk me through the migration"
    assert update.partial_text == ""
    assert update.pause is True


def test_transcript_stabilizer_handles_shifted_rolling_window():
    """It commits overlap when the rolling decode window slides forward."""
    stabilizer = TranscriptStabilizer()
    stabilizer.update("alpha beta gamma delta")

    update = stabilizer.update("beta gamma delta epsilon")

    assert update.stable_text == "beta gamma delta"
    assert update.partial_text == "epsilon"


def test_live_suggestion_state_throttles_and_discards_stale_responses():
    """It queues in-flight refreshes and ignores old response ids."""
    state = LiveSuggestionState()

    assert state.begin_request(10, 0.0) is None
    request_id = state.begin_request(30, 0.0)
    assert request_id == 1
    assert state.begin_request(60, 1.0) is None
    assert state.pending_refresh is True
    assert state.complete_request(999, 60, []) is False

    paths = [{"title": "Clarify", "text": "What was your role?", "detail": ""}]
    assert state.complete_request(1, 30, paths) is True
    assert state.current_paths == paths


def test_live_suggestion_state_respects_debounce_for_auto_refreshes():
    """It prevents automatic suggestion calls from firing too frequently."""
    state = LiveSuggestionState(last_refresh_words=30, last_request_at=10.0)

    assert state.begin_request(60, 12.0) is None
    assert state.begin_request(60, 16.0) == 1


def test_initial_live_paths_use_first_glanceable_board_cards():
    """It seeds the top path row from useful generated board cards."""
    paths = _initial_live_paths(
        [
            {"kind": "watchout", "title": "Risk", "text": "Clarify the gap."},
            {"kind": "opening", "title": "Open", "text": "Start here."},
            {"kind": "topic", "title": "Topic", "text": "Dig into scope."},
            {"kind": "candidate", "title": "Resume", "text": "Ask about project."},
            {"kind": "transition", "title": "Move", "text": "Shift topics."},
        ]
    )

    assert [path["title"] for path in paths] == ["Open", "Topic", "Resume"]
    assert all(path["kind"] == "live" for path in paths)

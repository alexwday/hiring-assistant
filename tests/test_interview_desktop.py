"""Tests for the native desktop interview assistant helpers."""

import numpy as np

from hiring_assistant.interview_desktop import (
    DEFAULT_WHISPER_MODEL_DIR,
    WHISPER_SAMPLE_RATE,
    _float32_mono_16k,
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

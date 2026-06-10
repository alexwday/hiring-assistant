"""Native desktop interview assistant with local Whisper transcription."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hiring_assistant.interview_workflows import InterviewContext, InterviewLLMService
from utilities.config import AppConfig, load_config, load_env_file
from utilities.logging_setup import setup_logging
from utilities.ssl_setup import SSLSetupResult, setup_ssl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPER_MODEL = "small.en"
DEFAULT_WHISPER_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "faster-whisper"
WHISPER_SAMPLE_RATE = 16000
CHUNK_MS = 100
SUGGESTION_WORD_DELTA = 25
SUGGESTION_MIN_WORDS = 12
SUGGESTION_DEBOUNCE_SECONDS = 5.0

QtCore: Any = None
QtGui: Any = None
QtWidgets: Any = None


@dataclass(frozen=True)
class AudioDevice:
    """One selectable input device."""

    index: int
    name: str
    channels: int
    sample_rate: int

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


@dataclass(frozen=True)
class TranscriptUpdate:
    """One live transcript update from the streaming transcriber."""

    stable_text: str = ""
    partial_text: str = ""
    word_count: int = 0
    pause: bool = False


@dataclass
class LiveSuggestionState:
    """State machine for throttled live suggestion refreshes."""

    current_paths: list[dict[str, str]] = field(default_factory=list)
    last_refresh_words: int = 0
    in_flight_request_id: int | None = None
    pending_refresh: bool = False
    last_request_at: float = -SUGGESTION_DEBOUNCE_SECONDS
    next_request_id: int = 0

    def begin_request(
        self,
        word_count: int,
        now: float,
        *,
        manual: bool = False,
        pause: bool = False,
        force: bool = False,
    ) -> int | None:
        """Return a request id when a live suggestion refresh should start."""
        if word_count < SUGGESTION_MIN_WORDS:
            return None
        if self.in_flight_request_id is not None:
            self.pending_refresh = True
            return None

        enough_new_words = word_count - self.last_refresh_words >= SUGGESTION_WORD_DELTA
        useful_pause = pause and word_count > self.last_refresh_words
        if not (manual or force or enough_new_words or useful_pause):
            return None
        if not (manual or force) and (
            now - self.last_request_at < SUGGESTION_DEBOUNCE_SECONDS
        ):
            return None

        self.next_request_id += 1
        self.in_flight_request_id = self.next_request_id
        self.last_request_at = now
        return self.next_request_id

    def complete_request(
        self,
        request_id: int,
        word_count: int,
        paths: list[dict[str, str]],
    ) -> bool:
        """Accept a response only if it belongs to the active request."""
        if request_id != self.in_flight_request_id:
            return False
        self.in_flight_request_id = None
        self.last_refresh_words = word_count
        if paths:
            self.current_paths = paths[:3]
        return True

    def fail_request(self, request_id: int) -> bool:
        """Clear an active request after an error."""
        if request_id != self.in_flight_request_id:
            return False
        self.in_flight_request_id = None
        return True


class TranscriptStabilizer:
    """Commit only text that agrees across consecutive rolling decodes."""

    def __init__(
        self,
    ):
        self.previous_text = ""
        self.committed_text = ""

    def update(self, decoded_text: str, *, pause: bool = False) -> TranscriptUpdate:
        """Process one tentative decode and return stable/new partial text."""
        current = _normalize_transcript_text(decoded_text)
        agreed = _common_word_agreement(self.previous_text, current)
        stable_text = _new_words_after_overlap(self.committed_text, agreed)
        if stable_text:
            self.committed_text = _join_transcript(self.committed_text, stable_text)
        self.previous_text = current
        partial_text = _new_words_after_overlap(self.committed_text, current)
        return TranscriptUpdate(
            stable_text=stable_text,
            partial_text=partial_text,
            word_count=len(self.committed_text.split()),
            pause=pause,
        )

    def force_commit(self, *, pause: bool = False) -> TranscriptUpdate:
        """Commit the current tentative text when the stream pauses or stops."""
        stable_text = _new_words_after_overlap(self.committed_text, self.previous_text)
        if stable_text:
            self.committed_text = _join_transcript(self.committed_text, stable_text)
        return TranscriptUpdate(
            stable_text=stable_text,
            partial_text="",
            word_count=len(self.committed_text.split()),
            pause=pause,
        )


class StreamingWhisperTranscriber:
    """Background faster-whisper transcriber with rolling partial decodes."""

    def __init__(
        self,
        on_update: Callable[[TranscriptUpdate], None],
        on_status: Callable[[str], None],
    ):
        self.on_update = on_update
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)
        self.worker: threading.Thread | None = None
        self.model: Any = None
        self.stabilizer = TranscriptStabilizer()
        self.rolling_buffer: list[np.ndarray] = []
        self.rolling_samples = 0

    def start(self) -> None:
        """Start loading the model and transcribing queued audio."""
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        """Stop transcription after flushing any pending audio."""
        self.stop_event.set()

    def send_audio(self, audio: np.ndarray) -> None:
        """Queue mono float32 16 kHz audio for transcription."""
        if audio.size == 0:
            return
        try:
            self.audio_queue.put_nowait(audio.astype(np.float32, copy=False))
        except queue.Full:
            self.on_status("Audio queue is full; dropping a chunk.")

    def _run(self) -> None:
        try:
            model = self._load_model()
        except Exception as exc:
            self.on_status(f"Whisper model error: {exc}")
            return

        partial_interval = _env_float("WHISPER_PARTIAL_INTERVAL_SECONDS", 1.0)
        stable_window_seconds = _env_float("WHISPER_STABLE_WINDOW_SECONDS", 6.0)
        max_window_seconds = _env_float("WHISPER_MAX_WINDOW_SECONDS", 10.0)
        pause_seconds = _env_float("WHISPER_PAUSE_SECONDS", 1.2)
        speech_rms = _env_float("WHISPER_SPEECH_RMS", 0.008)
        stable_window_samples = int(stable_window_seconds * WHISPER_SAMPLE_RATE)
        max_window_samples = int(max_window_seconds * WHISPER_SAMPLE_RATE)
        last_decode_at = 0.0
        last_voice_at = time.monotonic()
        speech_seen_since_pause = False
        last_partial_text = ""

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                chunk = np.array([], dtype=np.float32)

            now = time.monotonic()
            if chunk.size:
                self._append_rolling(chunk, max_window_samples)
                if _rms(chunk) >= speech_rms:
                    last_voice_at = now
                    speech_seen_since_pause = True

            if (
                self.rolling_samples >= WHISPER_SAMPLE_RATE
                and now - last_decode_at >= partial_interval
            ):
                last_decode_at = now
                update = self._decode_partial(model, stable_window_samples)
                if update and (
                    update.stable_text or update.partial_text != last_partial_text
                ):
                    self.on_update(update)
                    last_partial_text = update.partial_text

            if speech_seen_since_pause and now - last_voice_at >= pause_seconds:
                update = self.stabilizer.force_commit(pause=True)
                if update.stable_text or update.partial_text != last_partial_text or update.pause:
                    self.on_update(update)
                    last_partial_text = update.partial_text
                speech_seen_since_pause = False

        final_update = self._decode_partial(model, stable_window_samples)
        if final_update and final_update.stable_text:
            self.on_update(final_update)
        forced = self.stabilizer.force_commit(pause=True)
        if forced.stable_text or forced.pause:
            self.on_update(forced)

    def _load_model(self) -> Any:
        model_name = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip()
        model_dir = whisper_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        self.on_status(
            f"Loading faster-whisper model '{model_name}' from {model_dir}..."
        )
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "download_root": str(model_dir),
        }
        cpu_threads = _env_int("WHISPER_CPU_THREADS", 0)
        if cpu_threads > 0:
            kwargs["cpu_threads"] = cpu_threads
        model = WhisperModel(model_name, **kwargs)
        self.model = model
        self.on_status(f"Loaded faster-whisper model '{model_name}'.")
        return model

    def _append_rolling(self, chunk: np.ndarray, max_samples: int) -> None:
        self.rolling_buffer.append(chunk.astype(np.float32, copy=False))
        self.rolling_samples += chunk.size
        while self.rolling_samples > max_samples and self.rolling_buffer:
            overflow = self.rolling_samples - max_samples
            first = self.rolling_buffer[0]
            if first.size <= overflow:
                self.rolling_samples -= first.size
                self.rolling_buffer.pop(0)
                continue
            self.rolling_buffer[0] = first[overflow:]
            self.rolling_samples -= overflow
            break

    def _rolling_audio(self, max_samples: int) -> np.ndarray:
        if not self.rolling_buffer:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self.rolling_buffer).astype(np.float32, copy=False)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        return audio

    def _decode_partial(
        self,
        model: Any,
        stable_window_samples: int,
    ) -> TranscriptUpdate | None:
        audio = self._rolling_audio(stable_window_samples)
        if audio.size < WHISPER_SAMPLE_RATE:
            return None
        try:
            text = self._transcribe_audio(model, audio)
        except Exception as exc:
            self.on_status(f"Whisper transcription error: {exc}")
            return None
        if not text:
            return None
        return self.stabilizer.update(text)

    def _transcribe_audio(self, model: Any, audio: np.ndarray) -> str:
        language = os.getenv("WHISPER_LANGUAGE", "en").strip() or "en"
        vad_ms = _env_int("WHISPER_VAD_MIN_SILENCE_MS", 300)
        segments, _info = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": vad_ms},
        )
        return _normalize_transcript_text(
            " ".join(str(segment.text).strip() for segment in segments if segment.text)
        )


class AudioCapture:
    """Capture one or two input devices and send mixed audio to Whisper."""

    def __init__(
        self,
        devices: list[AudioDevice],
        on_audio: Callable[[np.ndarray], None],
        on_status: Callable[[str], None],
    ):
        self.devices = devices
        self.on_audio = on_audio
        self.on_status = on_status
        self.streams: list[Any] = []
        self.stop_event = threading.Event()
        self.device_queues: dict[int, queue.Queue[np.ndarray]] = {}
        self.mixer_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start device streams and the mixer loop."""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError:
            self.on_status("Install sounddevice to capture audio devices.")
            return

        self.stop_event.clear()
        for device in self.devices:
            self.device_queues[device.index] = queue.Queue(maxsize=50)
            channels = min(max(device.channels, 1), 2)
            blocksize = max(120, int(device.sample_rate * (CHUNK_MS / 1000)))

            def callback(
                indata: np.ndarray,
                _frames: int,
                _time_info: Any,
                status: Any,
                *,
                source_device: AudioDevice = device,
            ) -> None:
                if status:
                    self.on_status(str(status))
                audio = _float32_mono_16k(indata.copy(), source_device.sample_rate)
                target_queue = self.device_queues[source_device.index]
                try:
                    target_queue.put_nowait(audio)
                except queue.Full:
                    pass

            stream = sd.InputStream(
                device=device.index,
                channels=channels,
                samplerate=device.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                callback=callback,
            )
            stream.start()
            self.streams.append(stream)
            self.on_status(f"Capturing {device.label}")

        self.mixer_thread = threading.Thread(target=self._mix_loop, daemon=True)
        self.mixer_thread.start()

    def stop(self) -> None:
        """Stop all audio streams."""
        self.stop_event.set()
        for stream in self.streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Failed closing audio stream")
        self.streams = []

    def _mix_loop(self) -> None:
        while not self.stop_event.is_set():
            chunks: list[np.ndarray] = []
            for device_queue in self.device_queues.values():
                try:
                    chunks.append(device_queue.get(timeout=0.02))
                except queue.Empty:
                    continue
            if not chunks:
                time.sleep(0.03)
                continue
            max_len = max(chunk.size for chunk in chunks)
            padded = [
                np.pad(chunk, (0, max_len - chunk.size), mode="constant")
                if chunk.size < max_len
                else chunk
                for chunk in chunks
            ]
            mixed = np.sum(padded, axis=0) / max(1, len(padded))
            self.on_audio(np.clip(mixed, -1.0, 1.0).astype(np.float32))


class InterviewDesktopApp:
    """PySide6 desktop UI for the live interview assistant."""

    def __init__(self, config: AppConfig, ssl_setup: SSLSetupResult):
        _ensure_qt()
        self.config = config
        self.ssl_setup = ssl_setup
        self.context: InterviewContext | None = None
        self.cards: list[dict[str, str]] = []
        self.live_state = LiveSuggestionState()
        self.transcriber: StreamingWhisperTranscriber | None = None
        self.audio_capture: AudioCapture | None = None
        self.devices = list_input_devices()
        self.device_labels = [device.label for device in self.devices]
        self.window = QtWidgets.QMainWindow()
        self.bridge = _Bridge()
        self.bridge.status.connect(self._set_status)
        self.bridge.transcript.connect(self._apply_transcript_update)
        self.bridge.payload.connect(self._apply_payload)
        self.bridge.live_payload.connect(self._apply_live_payload)
        self.bridge.live_error.connect(self._apply_live_error)
        self._build_ui()

    def show(self) -> None:
        self.window.show()

    def _build_ui(self) -> None:
        self.window.setWindowTitle("Interview Assistant")
        self.window.resize(1500, 900)
        root = QtWidgets.QWidget()
        self.window.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 1)
        self.job_text = _labeled_text(left, "Job posting")
        _button(left, "Load job file", lambda: self._load_file(self.job_text))
        self.context_text = _labeled_text(left, "Additional context")
        _button(left, "Load context file", lambda: self._load_file(self.context_text))
        self.resume_text = _labeled_text(left, "Candidate resume")
        _button(left, "Load resume file", lambda: self._load_file(self.resume_text))
        _button(left, "Generate board", self._generate_board)
        left.addStretch(1)

        center = QtWidgets.QVBoxLayout()
        layout.addLayout(center, 3)
        controls = QtWidgets.QHBoxLayout()
        center.addLayout(controls)
        controls.addWidget(QtWidgets.QLabel("Mic"))
        self.mic_select = QtWidgets.QComboBox()
        self.mic_select.addItems(self.device_labels)
        controls.addWidget(self.mic_select, 1)
        controls.addWidget(QtWidgets.QLabel("Meeting audio"))
        self.meeting_select = QtWidgets.QComboBox()
        self.meeting_select.addItems(["None", *self.device_labels])
        controls.addWidget(self.meeting_select, 1)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._start_audio)
        controls.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_audio)
        controls.addWidget(self.stop_button)
        _button(controls, "Refresh", lambda: self._refresh_suggestions(manual=True))

        self.status_label = QtWidgets.QLabel("Ready.")
        center.addWidget(self.status_label)

        live_label = QtWidgets.QLabel("Next 3 paths")
        live_label.setStyleSheet("font-size: 22px; font-weight: 800;")
        center.addWidget(live_label)
        self.live_paths_container = QtWidgets.QWidget()
        self.live_paths_grid = QtWidgets.QGridLayout(self.live_paths_container)
        center.addWidget(self.live_paths_container)
        self._render_live_paths([])

        board_label = QtWidgets.QLabel("Interview board")
        board_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        center.addWidget(board_label)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.cards_container = QtWidgets.QWidget()
        self.cards_grid = QtWidgets.QGridLayout(self.cards_container)
        scroll.setWidget(self.cards_container)
        center.addWidget(scroll, 1)
        self._render_cards(
            [
                {"title": "Opening", "text": "Generate the board to load prompts."},
                {"title": "Topic", "text": "Live suggestions will appear here."},
            ]
        )

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        transcript_label = QtWidgets.QLabel("Transcript")
        transcript_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        right.addWidget(transcript_label)
        self.transcript_text = QtWidgets.QTextEdit()
        self.transcript_text.setReadOnly(False)
        right.addWidget(self.transcript_text, 1)
        right.addWidget(QtWidgets.QLabel("Partial"))
        self.partial_text = QtWidgets.QLabel("")
        self.partial_text.setWordWrap(True)
        self.partial_text.setStyleSheet(
            "color: #475467; font-size: 16px; font-style: italic;"
        )
        self.partial_text.setMinimumHeight(70)
        right.addWidget(self.partial_text)
        right.addWidget(QtWidgets.QLabel("Manual transcript notes"))
        self.manual_text = QtWidgets.QTextEdit()
        self.manual_text.setFixedHeight(150)
        right.addWidget(self.manual_text)
        _button(right, "Copy transcript", self._copy_transcript)

    def _load_file(self, text_widget: Any) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Load interview input",
            str(PROJECT_ROOT),
            "Interview inputs (*.pdf *.txt *.md *.markdown);;All files (*)",
        )
        if not path:
            return
        try:
            text = _read_input_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.window, "Could not read file", str(exc))
            return
        text_widget.setPlainText(text)

    def _generate_board(self) -> None:
        context = self._read_context()
        if not context.job_posting.strip() or not context.resume_text.strip():
            QtWidgets.QMessageBox.critical(
                self.window,
                "Missing inputs",
                "Job posting and candidate resume are required.",
            )
            return
        self.context = context
        self._set_status("Generating interview board...")
        threading.Thread(target=self._generate_board_worker, args=(context,), daemon=True).start()

    def _generate_board_worker(self, context: InterviewContext) -> None:
        try:
            payload = InterviewLLMService(self.config, self.ssl_setup).prepare_interview(
                context
            )
        except Exception as exc:
            self.bridge.status.emit(f"Board error: {exc}")
            return
        self.bridge.payload.emit(payload, "Board ready.")

    def _start_audio(self) -> None:
        if self.context is None:
            self.context = self._read_context()
        if not self.context.job_posting.strip() or not self.context.resume_text.strip():
            QtWidgets.QMessageBox.critical(
                self.window,
                "Missing inputs",
                "Generate or provide job posting and resume before starting audio.",
            )
            return
        devices = self._selected_audio_devices()
        if not devices:
            QtWidgets.QMessageBox.critical(
                self.window,
                "No audio device",
                "Select at least one input device.",
            )
            return
        self._set_status("Starting faster-whisper streaming transcription...")
        self.transcriber = StreamingWhisperTranscriber(
            on_update=self.bridge.transcript.emit,
            on_status=self.bridge.status.emit,
        )
        self.transcriber.start()
        self.audio_capture = AudioCapture(
            devices=devices,
            on_audio=self.transcriber.send_audio,
            on_status=self.bridge.status.emit,
        )
        self.audio_capture.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_audio(self) -> None:
        if self.audio_capture is not None:
            self.audio_capture.stop()
            self.audio_capture = None
        if self.transcriber is not None:
            self.transcriber.stop()
            self.transcriber = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_status("Stopped.")

    def _refresh_suggestions(
        self,
        *,
        manual: bool = False,
        pause: bool = False,
        force: bool = False,
    ) -> None:
        if self.context is None:
            self.context = self._read_context()
        if not self.context.job_posting.strip() or not self.context.resume_text.strip():
            if manual:
                self._set_status("Provide job posting and resume before refreshing.")
            return

        transcript = self._stable_transcript_value()
        manual_notes = self.manual_text.toPlainText().strip()
        word_count = len(f"{transcript} {manual_notes}".split())
        if word_count < SUGGESTION_MIN_WORDS:
            if manual:
                self._set_status("Waiting for more transcript before refreshing.")
            return

        in_flight = self.live_state.in_flight_request_id is not None
        request_id = self.live_state.begin_request(
            word_count,
            time.monotonic(),
            manual=manual,
            pause=pause,
            force=force,
        )
        if request_id is None:
            if manual and in_flight:
                self._set_status("Refresh running; queued one more pass.")
            return

        self._set_status("Refreshing next paths...")
        current_paths = list(self.live_state.current_paths)
        threading.Thread(
            target=self._suggestions_worker,
            args=(request_id, transcript, manual_notes, current_paths, word_count),
            daemon=True,
        ).start()

    def _suggestions_worker(
        self,
        request_id: int,
        transcript: str,
        manual_notes: str,
        current_paths: list[dict[str, str]],
        word_count: int,
    ) -> None:
        try:
            payload = InterviewLLMService(self.config, self.ssl_setup).suggest_next_paths(
                context=self.context or self._read_context(),
                transcript_tail=transcript,
                current_paths=current_paths,
                manual_notes=manual_notes,
            )
        except Exception as exc:
            self.bridge.live_error.emit(request_id, f"Refresh error: {exc}")
            return
        self.bridge.live_payload.emit(
            request_id,
            payload,
            "Next paths refreshed.",
            word_count,
        )

    def _apply_transcript_update(self, update: TranscriptUpdate) -> None:
        if not isinstance(update, TranscriptUpdate):
            return
        if update.stable_text:
            self._append_stable_transcript(update.stable_text)
        self.partial_text.setText(update.partial_text)
        if update.stable_text or update.pause:
            self._maybe_refresh_suggestions(pause=update.pause)

    def _append_stable_transcript(self, text: str) -> None:
        current = self.transcript_text.toPlainText().strip()
        updated = f"{current}\n\n{text}".strip() if current else text
        self.transcript_text.setPlainText(updated)
        self.transcript_text.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _maybe_refresh_suggestions(self, *, pause: bool = False) -> None:
        self._refresh_suggestions(manual=False, pause=pause)

    def _apply_payload(self, payload: dict[str, Any], status: str) -> None:
        self.cards = list(payload.get("cards") or [])
        self._render_cards(self.cards)
        self.live_state.current_paths = _initial_live_paths(self.cards)
        self._render_live_paths(self.live_state.current_paths)
        self._set_status(status)

    def _apply_live_payload(
        self,
        request_id: int,
        payload: dict[str, Any],
        status: str,
        word_count: int,
    ) -> None:
        paths = list(payload.get("paths") or [])
        if not self.live_state.complete_request(request_id, word_count, paths):
            self._set_status("Ignored stale next-path response.")
            return
        self._render_live_paths(self.live_state.current_paths)
        self._set_status(status)
        if self.live_state.pending_refresh:
            self.live_state.pending_refresh = False
            QtCore.QTimer.singleShot(
                0,
                lambda: self._refresh_suggestions(force=True),
            )

    def _apply_live_error(self, request_id: int, status: str) -> None:
        if not self.live_state.fail_request(request_id):
            return
        self._set_status(status)
        if self.live_state.pending_refresh:
            self.live_state.pending_refresh = False
            QtCore.QTimer.singleShot(
                0,
                lambda: self._refresh_suggestions(force=True),
            )

    def _render_cards(self, cards: list[dict[str, str]]) -> None:
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        visible_cards = cards[:10] or [{"title": "Suggestions", "text": "No cards yet."}]
        for index, card in enumerate(visible_cards):
            frame = QtWidgets.QFrame()
            frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            frame.setStyleSheet(
                "QFrame { background: white; border: 1px solid #d9dee7; "
                "border-radius: 8px; }"
            )
            box = QtWidgets.QVBoxLayout(frame)
            title = QtWidgets.QLabel(card.get("title") or card.get("kind") or "Suggestion")
            title.setStyleSheet("color: #667085; font-size: 12px; font-weight: 700;")
            box.addWidget(title)
            text = QtWidgets.QLabel(card.get("text") or "")
            text.setWordWrap(True)
            text.setStyleSheet("font-size: 25px; font-weight: 800; line-height: 1.15;")
            box.addWidget(text)
            if card.get("detail"):
                detail = QtWidgets.QLabel(card["detail"])
                detail.setWordWrap(True)
                detail.setStyleSheet("color: #667085; font-size: 13px;")
                box.addWidget(detail)
            box.addStretch(1)
            self.cards_grid.addWidget(frame, index // 2, index % 2)

    def _render_live_paths(self, paths: list[dict[str, str]]) -> None:
        while self.live_paths_grid.count():
            item = self.live_paths_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        visible_paths = paths[:3] or [
            {
                "title": "Generate board",
                "text": "Start audio to get live next-path suggestions.",
                "detail": "",
            },
            {
                "title": "Listen",
                "text": "Stable transcript will trigger refreshes automatically.",
                "detail": "",
            },
            {
                "title": "Refresh",
                "text": "Use Refresh when you want new paths on demand.",
                "detail": "",
            },
        ]
        for index, path in enumerate(visible_paths):
            frame = QtWidgets.QFrame()
            frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            frame.setStyleSheet(
                "QFrame { background: #f8fafc; border: 1px solid #cbd5e1; "
                "border-radius: 8px; }"
            )
            box = QtWidgets.QVBoxLayout(frame)
            title = QtWidgets.QLabel(path.get("title") or "Next path")
            title.setStyleSheet("color: #344054; font-size: 13px; font-weight: 800;")
            box.addWidget(title)
            text = QtWidgets.QLabel(path.get("text") or path.get("prompt") or "")
            text.setWordWrap(True)
            text.setStyleSheet("font-size: 27px; font-weight: 850; line-height: 1.12;")
            box.addWidget(text)
            detail_text = path.get("detail") or path.get("why") or ""
            if detail_text:
                detail = QtWidgets.QLabel(detail_text)
                detail.setWordWrap(True)
                detail.setStyleSheet("color: #475467; font-size: 13px;")
                box.addWidget(detail)
            box.addStretch(1)
            self.live_paths_grid.addWidget(frame, 0, index)

    def _read_context(self) -> InterviewContext:
        return InterviewContext(
            job_posting=self.job_text.toPlainText().strip(),
            work_context=self.context_text.toPlainText().strip(),
            resume_text=self.resume_text.toPlainText().strip(),
        )

    def _selected_audio_devices(self) -> list[AudioDevice]:
        selected = []
        for label in (self.mic_select.currentText(), self.meeting_select.currentText()):
            if not label or label == "None":
                continue
            device = _device_by_label(self.devices, label)
            if device and device not in selected:
                selected.append(device)
        return selected

    def _stable_transcript_value(self) -> str:
        return self.transcript_text.toPlainText().strip()

    def _transcript_value(self) -> str:
        return "\n\n".join(
            part
            for part in [
                self._stable_transcript_value(),
                self.manual_text.toPlainText().strip(),
            ]
            if part
        )

    def _copy_transcript(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._transcript_value())
        self._set_status("Transcript copied.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


def list_input_devices() -> list[AudioDevice]:
    """Return input devices from PortAudio/sounddevice."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError:
        return []

    devices: list[AudioDevice] = []
    for index, info in enumerate(sd.query_devices()):
        channels = int(info.get("max_input_channels") or 0)
        if channels <= 0:
            continue
        sample_rate = int(info.get("default_samplerate") or 48000)
        devices.append(
            AudioDevice(
                index=index,
                name=str(info.get("name") or f"Input {index}"),
                channels=channels,
                sample_rate=sample_rate,
            )
        )
    return devices


def whisper_model_dir() -> Path:
    """Return the project-local Whisper model directory."""
    raw = os.getenv("WHISPER_MODEL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_WHISPER_MODEL_DIR


def main() -> int:
    """Launch the native desktop interview assistant."""
    _ensure_qt()
    parser = argparse.ArgumentParser(description="Run the native interview app.")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()
    env_path = Path(args.env)
    if env_path.exists():
        load_env_file(env_path)
    config = load_config(args.env)
    setup_logging(config.log_level, output_logs=config.output_logs)
    ssl_setup = setup_ssl(config)
    if not ssl_setup.success:
        raise RuntimeError(ssl_setup.error or "SSL setup failed")
    app = QtWidgets.QApplication([])
    window = InterviewDesktopApp(config, ssl_setup)
    window.show()
    return app.exec()


def _ensure_qt() -> None:
    global QtCore, QtGui, QtWidgets
    if QtWidgets is not None:
        return
    try:
        from PySide6 import QtCore as qt_core
        from PySide6 import QtGui as qt_gui
        from PySide6 import QtWidgets as qt_widgets
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySide6 is required for the native interview app. Install project "
            "requirements with `python -m pip install -r requirements.txt`."
        ) from exc
    QtCore = qt_core
    QtGui = qt_gui
    QtWidgets = qt_widgets


def _Bridge() -> Any:
    _ensure_qt()

    class Bridge(QtCore.QObject):
        status = QtCore.Signal(str)
        transcript = QtCore.Signal(object)
        payload = QtCore.Signal(dict, str)
        live_payload = QtCore.Signal(int, dict, str, int)
        live_error = QtCore.Signal(int, str)

    return Bridge()


def _labeled_text(layout: Any, label: str) -> Any:
    title = QtWidgets.QLabel(label)
    title.setStyleSheet("font-size: 14px; font-weight: 700;")
    layout.addWidget(title)
    text = QtWidgets.QTextEdit()
    text.setMinimumHeight(120)
    layout.addWidget(text)
    return text


def _button(layout: Any, text: str, callback: Callable[[], None]) -> Any:
    button = QtWidgets.QPushButton(text)
    button.clicked.connect(callback)
    layout.addWidget(button)
    return button


def _read_input_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("PyMuPDF is required to read PDF files") from exc
        document = fitz.open(path)
        try:
            return "\n\n".join(page.get_text("text").strip() for page in document)
        finally:
            document.close()
    if suffix in {".txt", ".md", ".markdown", ""}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path.name}")


def _device_by_label(devices: list[AudioDevice], label: str) -> AudioDevice | None:
    for device in devices:
        if device.label == label:
            return device
    return None


def _initial_live_paths(cards: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred_kinds = {"opening", "topic", "candidate", "follow-up", "transition"}
    paths: list[dict[str, str]] = []
    for card in cards:
        if card.get("kind") not in preferred_kinds:
            continue
        paths.append(
            {
                "kind": "live",
                "title": card.get("title") or "Next path",
                "text": card.get("text") or "",
                "detail": card.get("detail") or "",
            }
        )
        if len(paths) == 3:
            break
    return paths


def _float32_mono_16k(indata: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert int16 or float input frames to mono 16 kHz float32 audio."""
    samples = np.asarray(indata)
    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)
    if np.issubdtype(np.asarray(indata).dtype, np.integer):
        samples = samples / 32768.0
    if source_rate != WHISPER_SAMPLE_RATE and len(samples) > 1:
        output_len = max(1, int(round(len(samples) * WHISPER_SAMPLE_RATE / source_rate)))
        source_positions = np.linspace(0, len(samples) - 1, num=len(samples))
        target_positions = np.linspace(0, len(samples) - 1, num=output_len)
        samples = np.interp(target_positions, source_positions, samples)
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def _normalize_transcript_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _word_key(word: str) -> str:
    return word.strip(".,?!:;\"'()[]{}").lower()


def _common_word_prefix(left: str, right: str) -> str:
    left_words = _normalize_transcript_text(left).split()
    right_words = _normalize_transcript_text(right).split()
    matched: list[str] = []
    for left_word, right_word in zip(left_words, right_words):
        if _word_key(left_word) != _word_key(right_word):
            break
        matched.append(right_word)
    return " ".join(matched)


def _common_word_agreement(left: str, right: str) -> str:
    """Return stable text shared by consecutive decodes.

    Rolling windows eventually slide forward, so agreement can be either a
    prefix match or a suffix-of-left/prefix-of-right overlap.
    """
    left_words = _normalize_transcript_text(left).split()
    right_words = _normalize_transcript_text(right).split()
    if not left_words or not right_words:
        return ""

    best_size = len(_common_word_prefix(left, right).split())
    left_keys = [_word_key(word) for word in left_words]
    right_keys = [_word_key(word) for word in right_words]
    max_size = min(len(left_keys), len(right_keys))
    for size in range(best_size + 1, max_size + 1):
        if left_keys[-size:] == right_keys[:size]:
            best_size = size
    return " ".join(right_words[:best_size])


def _new_words_after_overlap(existing: str, candidate: str) -> str:
    candidate_words = _normalize_transcript_text(candidate).split()
    if not candidate_words:
        return ""
    existing_words = _normalize_transcript_text(existing).split()
    if not existing_words:
        return " ".join(candidate_words)

    candidate_keys = [_word_key(word) for word in candidate_words]
    existing_keys = [_word_key(word) for word in existing_words]
    tail_keys = existing_keys[-120:]

    if len(candidate_keys) <= len(tail_keys):
        for offset in range(0, len(tail_keys) - len(candidate_keys) + 1):
            if tail_keys[offset : offset + len(candidate_keys)] == candidate_keys:
                return ""

    overlap = 0
    max_overlap = min(len(existing_keys), len(candidate_keys))
    for size in range(1, max_overlap + 1):
        if existing_keys[-size:] == candidate_keys[:size]:
            overlap = size
    return " ".join(candidate_words[overlap:])


def _join_transcript(existing: str, addition: str) -> str:
    existing = _normalize_transcript_text(existing)
    addition = _normalize_transcript_text(addition)
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing} {addition}"


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main())

"""Native desktop interview assistant with local Whisper transcription."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hiring_assistant.interview_workflows import InterviewContext, InterviewLLMService
from utilities.config import AppConfig, load_config, load_env_file
from utilities.logging_setup import setup_logging
from utilities.ssl_setup import SSLSetupResult, setup_ssl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "whisper"
WHISPER_SAMPLE_RATE = 16000
CHUNK_MS = 100
SUGGESTION_WORD_DELTA = 70
SUGGESTION_MIN_WORDS = 12

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


class LocalWhisperTranscriber:
    """Background local Whisper transcriber for live audio chunks."""

    def __init__(
        self,
        on_transcript: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)
        self.worker: threading.Thread | None = None
        self.model: Any = None

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

        buffer: list[np.ndarray] = []
        buffered_seconds = 0.0
        last_voice_at = time.monotonic()
        silence_seconds = _env_float("WHISPER_SILENCE_SECONDS", 0.9)
        min_chunk_seconds = _env_float("WHISPER_MIN_CHUNK_SECONDS", 4.0)
        max_chunk_seconds = _env_float("WHISPER_MAX_CHUNK_SECONDS", 12.0)
        speech_rms = _env_float("WHISPER_SPEECH_RMS", 0.008)

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.15)
            except queue.Empty:
                chunk = np.array([], dtype=np.float32)

            if chunk.size:
                buffer.append(chunk)
                buffered_seconds += chunk.size / WHISPER_SAMPLE_RATE
                if _rms(chunk) >= speech_rms:
                    last_voice_at = time.monotonic()

            silence_elapsed = time.monotonic() - last_voice_at
            should_flush = (
                buffered_seconds >= min_chunk_seconds
                and silence_elapsed >= silence_seconds
            ) or buffered_seconds >= max_chunk_seconds
            if should_flush and buffer:
                self._transcribe_buffer(model, buffer)
                buffer = []
                buffered_seconds = 0.0
                last_voice_at = time.monotonic()

        if buffer:
            self._transcribe_buffer(model, buffer)

    def _load_model(self) -> Any:
        model_name = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip()
        model_dir = whisper_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        self.on_status(
            f"Loading local Whisper model '{model_name}' from {model_dir}..."
        )
        import whisper  # type: ignore[import-not-found]

        model = whisper.load_model(model_name, download_root=str(model_dir))
        self.model = model
        self.on_status(f"Loaded local Whisper model '{model_name}'.")
        return model

    def _transcribe_buffer(self, model: Any, buffer: list[np.ndarray]) -> None:
        audio = np.concatenate(buffer).astype(np.float32, copy=False)
        if audio.size < WHISPER_SAMPLE_RATE:
            return
        try:
            language = os.getenv("WHISPER_LANGUAGE", "en").strip() or "en"
            result = model.transcribe(
                audio,
                language=language,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            self.on_status(f"Whisper transcription error: {exc}")
            return
        text = str(result.get("text") or "").strip()
        if text:
            self.on_transcript(text)


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
        self.last_suggestion_words = 0
        self.suggestion_in_flight = False
        self.transcriber: LocalWhisperTranscriber | None = None
        self.audio_capture: AudioCapture | None = None
        self.devices = list_input_devices()
        self.device_labels = [device.label for device in self.devices]
        self.window = QtWidgets.QMainWindow()
        self.bridge = _Bridge()
        self.bridge.status.connect(self._set_status)
        self.bridge.transcript.connect(self._append_transcript)
        self.bridge.payload.connect(self._apply_payload)
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
        _button(controls, "Refresh", self._refresh_suggestions)

        self.status_label = QtWidgets.QLabel("Ready.")
        center.addWidget(self.status_label)
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
        self._set_status("Starting local Whisper transcription...")
        self.transcriber = LocalWhisperTranscriber(
            on_transcript=self.bridge.transcript.emit,
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

    def _refresh_suggestions(self) -> None:
        if self.context is None:
            self.context = self._read_context()
        transcript = self._transcript_value()
        if len(transcript.split()) < SUGGESTION_MIN_WORDS:
            self._set_status("Waiting for more transcript before refreshing.")
            return
        if self.suggestion_in_flight:
            return
        self.suggestion_in_flight = True
        self._set_status("Refreshing live suggestions...")
        threading.Thread(target=self._suggestions_worker, args=(transcript,), daemon=True).start()

    def _suggestions_worker(self, transcript: str) -> None:
        try:
            payload = InterviewLLMService(self.config, self.ssl_setup).suggest_live(
                context=self.context or self._read_context(),
                transcript=transcript,
                previous_suggestions=[card.get("text", "") for card in self.cards],
            )
        except Exception as exc:
            self.bridge.status.emit(f"Refresh error: {exc}")
            self.suggestion_in_flight = False
            return
        self.bridge.payload.emit(payload, "Suggestions refreshed.")
        self.last_suggestion_words = len(transcript.split())
        self.suggestion_in_flight = False

    def _append_transcript(self, text: str) -> None:
        current = self.transcript_text.toPlainText().strip()
        updated = f"{current}\n\n{text}".strip() if current else text
        self.transcript_text.setPlainText(updated)
        self.transcript_text.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self._maybe_refresh_suggestions()

    def _maybe_refresh_suggestions(self) -> None:
        words = len(self._transcript_value().split())
        if words >= SUGGESTION_MIN_WORDS and (
            words - self.last_suggestion_words >= SUGGESTION_WORD_DELTA
        ):
            self._refresh_suggestions()

    def _apply_payload(self, payload: dict[str, Any], status: str) -> None:
        self.cards = list(payload.get("cards") or [])
        self._render_cards(self.cards)
        self._set_status(status)

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

    def _transcript_value(self) -> str:
        return "\n\n".join(
            part
            for part in [
                self.transcript_text.toPlainText().strip(),
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
        transcript = QtCore.Signal(str)
        payload = QtCore.Signal(dict, str)

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


if __name__ == "__main__":
    raise SystemExit(main())

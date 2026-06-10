"""LLM and realtime helpers for the live interview assistant."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from connections.llm_connector import LLMClient
from connections.oauth_connector import OAuthClient
from hiring_assistant.llm_workflows import _guarded_llm_call, parse_json_response
from utilities.config import AppConfig
from utilities.ssl_setup import SSLSetupResult

logger = logging.getLogger(__name__)

INTERVIEW_PREP_PROMPT_VERSION = "interview-prep-v1"
INTERVIEW_LIVE_PROMPT_VERSION = "interview-live-suggestions-v1"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
MAX_CONTEXT_CHARS = 18000
MAX_TRANSCRIPT_CHARS = 14000


@dataclass(frozen=True)
class InterviewContext:
    """Inputs used by the interview assistant prompts."""

    job_posting: str
    work_context: str
    resume_text: str


class InterviewLLMService:
    """Generate initial and live interview cue cards."""

    def __init__(self, config: AppConfig, ssl_setup: SSLSetupResult):
        self.config = config
        self.ssl_setup = ssl_setup

    def prepare_interview(self, context: InterviewContext) -> dict[str, Any]:
        """Create the initial interview board from job and candidate context."""
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            response = _guarded_llm_call(
                client,
                messages=_prepare_messages(context),
                settings={
                    "model_size": "small",
                    "response_format": {"type": "json_object"},
                },
            )
        finally:
            client.close()

        payload = parse_json_response(response, "interview preparation")
        normalized = normalize_interview_payload(payload)
        normalized["prompt_versions"] = {"interview_prep": INTERVIEW_PREP_PROMPT_VERSION}
        return normalized

    def suggest_live(
        self,
        context: InterviewContext,
        transcript: str,
        previous_suggestions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create fresh live suggestions from the current transcript."""
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            response = _guarded_llm_call(
                client,
                messages=_live_messages(context, transcript, previous_suggestions or []),
                settings={
                    "model_size": "small",
                    "response_format": {"type": "json_object"},
                },
            )
        finally:
            client.close()

        payload = parse_json_response(response, "live interview suggestions")
        normalized = normalize_interview_payload(payload)
        normalized["prompt_versions"] = {
            "interview_live": INTERVIEW_LIVE_PROMPT_VERSION
        }
        return normalized


def create_realtime_transcription_answer(
    offer_sdp: str,
    config: AppConfig,
    ssl_setup: SSLSetupResult,
) -> str:
    """Create a server-mediated WebRTC answer for realtime transcription."""
    if not offer_sdp.strip():
        raise ValueError("Missing WebRTC offer SDP")

    token = _api_bearer_token(config, ssl_setup)
    session = _realtime_session_config()
    url = f"{config.llm.base_url.rstrip('/')}/realtime/calls"
    headers = {
        "Authorization": f"Bearer {token}",
        "OpenAI-Safety-Identifier": "local-hiring-assistant",
    }
    files = {
        "sdp": ("offer.sdp", offer_sdp, "application/sdp"),
        "session": (None, json.dumps(session), "application/json"),
    }
    logger.info(
        "Creating realtime transcription call: url=%s model=%s transcription=%s",
        url,
        session.get("model"),
        (
            session.get("audio", {})
            .get("input", {})
            .get("transcription", {})
            .get("model")
        ),
    )
    with httpx.Client(
        verify=ssl_setup.verify_value,
        timeout=httpx.Timeout(30.0),
    ) as client:
        response = client.post(url, headers=headers, files=files)
    if response.status_code >= 400:
        detail = response.text[:1000]
        raise RuntimeError(
            f"Realtime transcription setup failed "
            f"({response.status_code}): {detail}"
        )
    return response.text


def normalize_interview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model JSON into the frontend card schema."""
    cards: list[dict[str, str]] = []
    explicit_cards = payload.get("cards")
    if isinstance(explicit_cards, list):
        cards.extend(_normalize_cards(explicit_cards, "suggestion"))

    field_map = [
        ("opening", "opening", "Opening"),
        ("opening_prompts", "opening", "Opening"),
        ("topics", "topic", "Topic"),
        ("priority_topics", "topic", "Topic"),
        ("candidate_threads", "candidate", "Candidate thread"),
        ("candidate_specific_questions", "candidate", "Candidate question"),
        ("follow_ups", "follow-up", "Follow-up"),
        ("top_suggestions", "live", "Live suggestion"),
        ("transitions", "transition", "Transition"),
        ("closing", "closing", "Closing"),
        ("watchouts", "watchout", "Watchout"),
        ("risks_or_gaps", "watchout", "Watchout"),
    ]
    for field_name, kind, fallback_title in field_map:
        cards.extend(
            _normalize_cards(payload.get(field_name), kind, fallback_title)
        )

    deduped = _dedupe_cards(cards)
    return {
        "summary": _string(payload.get("summary") or payload.get("interview_strategy")),
        "cards": deduped[:14],
        "signals": _string_list(payload.get("signals") or payload.get("transcript_signals"))[
            :6
        ],
    }


def _prepare_messages(context: InterviewContext) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a quiet interview copilot for a hiring manager. "
                "Create concise, glanceable prompts that can sit on screen "
                "during a live interview. Use a practical, conversational tone. "
                "Do not invent facts about the candidate. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create an interview cue board for this candidate.\n\n"
                "Return JSON with these keys:\n"
                "- summary: one short interview strategy sentence\n"
                "- opening: 3 concise ways to start\n"
                "- priority_topics: 5 topics worth covering\n"
                "- candidate_specific_questions: 6 resume-grounded questions\n"
                "- transitions: 5 short transition lines\n"
                "- follow_ups: 5 follow-up prompts for deeper detail\n"
                "- watchouts: 4 gaps, ambiguities, or signals to listen for\n"
                "- closing: 2 closing prompts\n\n"
                "Each item can be a string or an object with title, text, and detail. "
                "Keep text short enough to read at a glance.\n\n"
                f"Job posting:\n{_trim(context.job_posting, MAX_CONTEXT_CHARS)}\n\n"
                f"Additional context:\n{_trim(context.work_context, 9000)}\n\n"
                f"Candidate resume:\n{_trim(context.resume_text, MAX_CONTEXT_CHARS)}"
            ),
        },
    ]


def _live_messages(
    context: InterviewContext,
    transcript: str,
    previous_suggestions: list[str],
) -> list[dict[str, str]]:
    previous = "\n".join(f"- {_trim(item, 220)}" for item in previous_suggestions[:12])
    return [
        {
            "role": "system",
            "content": (
                "You are a live interview copilot. Based on the transcript so far, "
                "suggest what the interviewer could ask or say next. Prefer fresh, "
                "useful prompts over generic interview advice. Do not evaluate the "
                "candidate for a final decision. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate fresh live suggestions for the next few minutes.\n\n"
                "Return JSON with these keys:\n"
                "- summary: one short read on where to steer next\n"
                "- top_suggestions: 5 best next things to ask or say\n"
                "- follow_ups: 4 deeper follow-ups tied to what was just said\n"
                "- transitions: 3 natural ways to move topics\n"
                "- watchouts: 3 things to clarify or listen for\n"
                "- signals: 3 brief transcript signals you used\n\n"
                "Each suggestion should be short, specific, and spoken-language ready. "
                "Avoid repeating previous suggestions unless the transcript makes one "
                "still clearly important.\n\n"
                f"Previous visible suggestions:\n{previous or '- none'}\n\n"
                f"Job posting:\n{_trim(context.job_posting, 9000)}\n\n"
                f"Additional context:\n{_trim(context.work_context, 7000)}\n\n"
                f"Candidate resume:\n{_trim(context.resume_text, 9000)}\n\n"
                f"Live transcript tail:\n{_trim_tail(transcript, MAX_TRANSCRIPT_CHARS)}"
            ),
        },
    ]


def _realtime_session_config() -> dict[str, Any]:
    realtime_model = os.getenv("REALTIME_MODEL", DEFAULT_REALTIME_MODEL).strip()
    transcription_model = os.getenv(
        "REALTIME_TRANSCRIPTION_MODEL",
        DEFAULT_REALTIME_TRANSCRIPTION_MODEL,
    ).strip()
    language = os.getenv("REALTIME_TRANSCRIPTION_LANGUAGE", "en").strip() or "en"
    delay = os.getenv("REALTIME_TRANSCRIPTION_DELAY", "low").strip() or "low"
    return {
        "type": "realtime",
        "model": realtime_model,
        "output_modalities": ["text"],
        "instructions": (
            "You are only supporting live interview transcription. Do not speak. "
            "Do not create assistant responses unless explicitly requested."
        ),
        "audio": {
            "input": {
                "transcription": {
                    "model": transcription_model,
                    "language": language,
                    "delay": delay,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.45,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                },
            }
        },
        "include": ["item.input_audio_transcription.logprobs"],
    }


def _api_bearer_token(config: AppConfig, ssl_setup: SSLSetupResult) -> str:
    if config.auth_mode == "local":
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY or API_KEY is required for realtime audio")
        return config.api_key
    oauth_client = OAuthClient(config.oauth, verify=ssl_setup.verify_value)
    return oauth_client.get_token()


def _normalize_cards(
    value: Any,
    kind: str,
    fallback_title: str = "Suggestion",
) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        items: list[Any] = list(value.values())
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    cards: list[dict[str, str]] = []
    for item in items:
        title = fallback_title
        text = ""
        detail = ""
        if isinstance(item, dict):
            title = _string(
                item.get("title")
                or item.get("label")
                or item.get("topic")
                or item.get("target_topic")
                or fallback_title
            )
            text = _string(
                item.get("text")
                or item.get("question")
                or item.get("prompt")
                or item.get("line")
                or item.get("suggestion")
                or item.get("description")
            )
            detail = _string(
                item.get("detail")
                or item.get("why")
                or item.get("rationale")
                or item.get("note")
            )
            follow_ups = _string_list(item.get("follow_ups"))
            if follow_ups and not detail:
                detail = " / ".join(follow_ups[:2])
        else:
            text = _string(item)

        if not text and title != fallback_title:
            text = title
            title = fallback_title
        if not text:
            continue
        cards.append(
            {
                "kind": kind,
                "title": _trim(title, 80),
                "text": _trim(text, 260),
                "detail": _trim(detail, 220),
            }
        )
    return cards


def _dedupe_cards(cards: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for card in cards:
        key = card["text"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_string(item) for item in value if _string(item)]
    if isinstance(value, dict):
        return [_string(item) for item in value.values() if _string(item)]
    text = _string(value)
    return [text] if text else []


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def _trim(value: str, limit: int) -> str:
    text = _string(value)
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + "\n[truncated]"


def _trim_tail(value: str, limit: int) -> str:
    text = _string(value)
    if len(text) <= limit:
        return text
    return "[earlier transcript omitted]\n" + text[-limit:].lstrip()

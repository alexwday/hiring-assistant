"""LLM workflows for resume extraction and candidate review."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from connections.llm_connector import LLMClient, extract_message_text
from utilities.config import AppConfig
from utilities.ssl_setup import SSLSetupResult

logger = logging.getLogger(__name__)

MARKDOWN_PROMPT_VERSION = "resume-page-markdown-v2"
METADATA_PROMPT_VERSION = "resume-metadata-v2"
REVIEW_PROMPT_VERSION = "resume-job-review-v4"
FINAL_RERANK_PROMPT_VERSION = "resume-final-rerank-v2"
ONE_DECIMAL = Decimal("0.1")
HOLISTIC_ADJUSTMENT_LIMIT = Decimal("1.5")
AGGREGATE_WEIGHTS = {
    "education_score": Decimal("0.25"),
    "experience_score": Decimal("0.45"),
    "projects_score": Decimal("0.30"),
}
DEFAULT_PARALLEL_LLM_CALLS = 8
DEFAULT_PAGE_WORKERS = 8
_LLM_SEMAPHORE_LOCK = threading.Lock()
_LLM_SEMAPHORE: threading.BoundedSemaphore | None = None
_LLM_SEMAPHORE_LIMIT = 0


@dataclass(frozen=True)
class ProcessedResume:
    """Markdown and metadata extracted from one resume."""

    markdown: str
    metadata: dict[str, Any]
    prompt_versions: dict[str, str]


@dataclass(frozen=True)
class ResumeReview:
    """Structured review and rendered report for one candidate."""

    report_markdown: str
    review_payload: dict[str, Any]
    scores: dict[str, Any]
    prompt_versions: dict[str, str]


@dataclass(frozen=True)
class FinalRerank:
    """Final comparative ranking across the top reviewed candidates."""

    payload: dict[str, Any]
    prompt_versions: dict[str, str]


class ResumeLLMService:
    """Run resume OCR/extraction and job-fit review prompts."""

    def __init__(self, config: AppConfig, ssl_setup: SSLSetupResult):
        self.config = config
        self.ssl_setup = ssl_setup

    def process_resume_pages(
        self,
        page_paths: list[Path],
        original_filename: str,
    ) -> ProcessedResume:
        """Convert page images into markdown and extract candidate metadata."""
        page_markdown = self._extract_pages_markdown_parallel(
            page_paths=page_paths,
            original_filename=original_filename,
        )
        markdown = self._combine_page_markdown(page_markdown, original_filename)
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            metadata = self._extract_metadata(client, markdown)
        finally:
            client.close()

        return ProcessedResume(
            markdown=markdown,
            metadata=metadata,
            prompt_versions={
                "markdown": MARKDOWN_PROMPT_VERSION,
                "metadata": METADATA_PROMPT_VERSION,
            },
        )

    def _extract_pages_markdown_parallel(
        self,
        page_paths: list[Path],
        original_filename: str,
    ) -> list[str]:
        page_count = len(page_paths)
        workers = _worker_count(
            "APP_PAGE_WORKERS",
            DEFAULT_PAGE_WORKERS,
            page_count,
        )
        if workers <= 1:
            client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
            try:
                return [
                    self._extract_page_markdown(
                        client=client,
                        page_path=page_path,
                        original_filename=original_filename,
                        page_number=index,
                        page_count=page_count,
                    )
                    for index, page_path in enumerate(page_paths, start=1)
                ]
            finally:
                client.close()

        page_markdown = [""] * page_count
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._extract_page_markdown_with_client,
                    page_path,
                    original_filename,
                    index,
                    page_count,
                ): index
                for index, page_path in enumerate(page_paths, start=1)
            }
            for future in as_completed(futures):
                page_index = futures[future]
                page_markdown[page_index - 1] = future.result()
        return page_markdown

    def _extract_page_markdown_with_client(
        self,
        page_path: Path,
        original_filename: str,
        page_number: int,
        page_count: int,
    ) -> str:
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            return self._extract_page_markdown(
                client=client,
                page_path=page_path,
                original_filename=original_filename,
                page_number=page_number,
                page_count=page_count,
            )
        finally:
            client.close()

    def review_resume(
        self,
        resume_markdown: str,
        metadata: dict[str, Any],
        job_posting: str,
        work_context: str,
    ) -> ResumeReview:
        """Review one processed resume against a job posting."""
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            payload = self._review_payload(
                client=client,
                resume_markdown=resume_markdown,
                metadata=metadata,
                job_posting=job_posting,
                work_context=work_context,
            )
        finally:
            client.close()

        report = render_review_markdown(payload)
        screening_average = _screening_average(
            payload.get("aggregate_score"),
            payload.get("holistic_score"),
        )
        scores = {
            "education": payload.get("education_score"),
            "experience": payload.get("experience_score"),
            "projects": payload.get("projects_score"),
            "aggregate": payload.get("aggregate_score"),
            "holistic": payload.get("holistic_score"),
            "screening_average": screening_average,
            "located_in_canada": _boolish(payload.get("located_in_canada")),
            "recommendation": payload.get("recommendation"),
        }
        return ResumeReview(
            report_markdown=report,
            review_payload=payload,
            scores=scores,
            prompt_versions={"review": REVIEW_PROMPT_VERSION},
        )

    def final_rerank_candidates(
        self,
        candidates: list[dict[str, Any]],
        job_posting: str,
        work_context: str,
    ) -> FinalRerank:
        """Run a final comparative judge over the top reviewed candidates."""
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            payload = self._final_rerank_payload(
                client=client,
                candidates=candidates,
                job_posting=job_posting,
                work_context=work_context,
            )
        finally:
            client.close()
        return FinalRerank(
            payload=payload,
            prompt_versions={"final_rerank": FINAL_RERANK_PROMPT_VERSION},
        )

    def _extract_page_markdown(
        self,
        client: LLMClient,
        page_path: Path,
        original_filename: str,
        page_number: int,
        page_count: int,
    ) -> str:
        image_url = _image_data_url(page_path)
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert redacted resume PDF page images into clean "
                    "markdown for downstream LLM ingestion and candidate "
                    "screening. Preserve factual content, dates, employers, "
                    "education, skills, projects, publications, certifications, "
                    "metrics, and role scope. Normalize layout into readable "
                    "headings, tables, and bullet lists. Preserve visible "
                    "redaction markers as [REDACTED]. Do not infer missing "
                    "facts, guess hidden text, or recreate names, contact "
                    "details, URLs, addresses, or other redacted PII. "
                    "Return only markdown for this page."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Resume file: {original_filename}\n"
                            f"Page {page_number} of {page_count}.\n"
                            "Transcribe and normalize this page into markdown."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
        ]
        response = _guarded_llm_call(
            client,
            messages=messages,
            settings={
                "model_size": "small",
                "max_tokens": 8192,
                "temperature": 0,
            },
        )
        return extract_message_text(response).strip()

    def _extract_metadata(
        self,
        client: LLMClient,
        resume_markdown: str,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract non-PII resume metadata as strict JSON. Include "
                    "only values directly supported by the resume text. Current "
                    "or recent title, employer, skills, education highlights, "
                    "and experience estimates are allowed. Never include "
                    "candidate name, address, phone, email, URLs, LinkedIn, "
                    "GitHub, portfolio, exact city, postal code, or other "
                    "contact/location details. Use empty strings for unknown "
                    "scalar fields and empty arrays for unknown lists."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return JSON with this schema:\n"
                    "{\n"
                    '  "current_or_recent_title": "",\n'
                    '  "current_or_recent_employer": "",\n'
                    '  "education_highlights": [],\n'
                    '  "skills": [],\n'
                    '  "years_of_experience_estimate": ""\n'
                    "}\n\n"
                    f"Resume markdown:\n{resume_markdown}"
                ),
            },
        ]
        response = _guarded_llm_call(
            client,
            messages=messages,
            settings={
                "model_size": "small",
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
        )
        payload = parse_json_response(response, "metadata extraction")
        return _remove_pii_metadata(payload)

    def _review_payload(
        self,
        client: LLMClient,
        resume_markdown: str,
        metadata: dict[str, Any],
        job_posting: str,
        work_context: str,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a rigorous hiring-manager screening assistant for "
                    "an individual resume review stage. Evaluate only explicit, "
                    "job-relevant evidence from the redacted resume markdown, "
                    "non-PII metadata, job posting, and additional work context. "
                    "This stage scores one candidate in isolation; the app later "
                    "ranks all reviewed candidates by the average of "
                    "aggregate_score and holistic_score, and a separate final "
                    "rerank compares the top candidates. Do not assign final "
                    "top-six placement here. Do not infer protected "
                    "characteristics, citizenship, nationality, age, gender, or "
                    "hidden PII. Treat additional work context as role-specific "
                    "evaluation guidance only when supported by resume evidence. "
                    "Look for direct fit, transferable experience, project "
                    "evidence, practical execution, level fit, and non-obvious "
                    "strengths that could make a candidate effective even if "
                    "their profile is non-traditional. Be specific about "
                    "evidence, gaps, uncertainty, and any scoring adjustment. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review this candidate against the job posting. Return JSON with "
                    "the exact fields listed below. Scores are 0.0-10.0, where 10.0 "
                    "is an exceptional match. Use one decimal place for every score "
                    "(for example 8.6 or 9.7), and do not default to whole-number "
                    "scores unless the evidence exactly supports them.\n\n"
                    "Scoring rubric and calibration:\n"
                    "- 9.0-10.0: exceptional direct evidence for this role; few "
                    "material concerns.\n"
                    "- 8.0-8.9: strong match with clear role-relevant evidence and "
                    "manageable gaps.\n"
                    "- 7.0-7.9: workable match with relevant transferable evidence "
                    "but meaningful follow-up questions.\n"
                    "- 6.0-6.9: partial fit; some relevant evidence but important "
                    "role gaps or uncertainty.\n"
                    "- 4.0-5.9: weak or indirect fit; limited evidence for core "
                    "needs.\n"
                    "- 0.0-3.9: little or no evidence for the dimension.\n\n"
                    "Dimension scoring rules:\n"
                    "- education_score measures job-relevant program, level, "
                    "completion, technical/quantitative foundation, and stated "
                    "academic evidence. Do not reward school prestige by itself.\n"
                    "- experience_score measures job-relevant work scope, domain "
                    "fit, stakeholder exposure, production delivery, analytical "
                    "or LLM implementation, ownership, and level fit.\n"
                    "- projects_score measures hands-on project relevance, "
                    "technical depth, complexity, delivery evidence, and how well "
                    "the project maps to the job.\n"
                    "- aggregate_score is the reproducible standalone evidence "
                    "score using this formula: 25% education_score + 45% "
                    "experience_score + 30% projects_score, rounded to one "
                    "decimal. Do not add work-context preference bonuses to "
                    "aggregate_score outside the three dimensions.\n"
                    "- holistic_score starts from aggregate_score and may move up "
                    "or down by at most 1.5 points for explicit, job-relevant "
                    "context: hidden-gem indicators, unusually strong or weak "
                    "evidence quality, current RBC experience, large reputable "
                    "financial-institution experience, shareholder-facing work, "
                    "LLM code delivery, validation or production delivery, "
                    "Canada-based availability when relevant to the posting, or "
                    "over/under-level risk for a junior/data-scientist role. "
                    "Apply these factors only when the resume supports them, do "
                    "not double-count them, and explain the adjustment.\n"
                    "- The app calculates screening_average later as the average "
                    "of aggregate_score and holistic_score. The top-six project "
                    "recommendation is assigned later by ranking all reviewed "
                    "candidates, not by this standalone prompt.\n\n"
                    "Required JSON schema:\n"
                    "{\n"
                    '  "tradeoff_analysis": "",\n'
                    '  "score_rationale": {\n'
                    '    "aggregate_formula": "",\n'
                    '    "holistic_adjustments": "",\n'
                    '    "calibration_notes": ""\n'
                    "  },\n"
                    '  "education_summary": "",\n'
                    '  "education_entries": [\n'
                    '    {"university": "", "level": "", "completion": "", '
                    '"program": "", "gpa": "", "fit_summary": ""}\n'
                    "  ],\n"
                    '  "education_score": 0.0,\n'
                    '  "experience_summary": "",\n'
                    '  "work_experience_fit_bullets": [],\n'
                    '  "experience_score": 0.0,\n'
                    '  "relevant_projects": [\n'
                    '    {"name": "", "evidence": "", '
                    '"role_relevance": "", "summary": "", "score": 0.0}\n'
                    "  ],\n"
                    '  "projects_score": 0.0,\n'
                    '  "unique_standouts": [\n'
                    '    {"signal": "", "why_it_matters": "", "confidence": ""}\n'
                    "  ],\n"
                    '  "gaps_and_risks": [\n'
                    '    {"gap": "", "screening_follow_up": "", "severity": ""}\n'
                    "  ],\n"
                    '  "aggregate_score": 0.0,\n'
                    '  "holistic_score": 0.0,\n'
                    '  "located_in_canada": false,\n'
                    '  "recommendation": "",\n'
                    '  "recommendation_rationale": "",\n'
                    '  "prescreen_email_subject": "",\n'
                    '  "prescreen_email_body": ""\n'
                    "}\n\n"
                    "score_rationale.aggregate_formula should briefly show how "
                    "the weighted aggregate was formed from the three dimension "
                    "scores. score_rationale.holistic_adjustments should name any "
                    "supported context-based adjustment and why it is fair. "
                    "score_rationale.calibration_notes should mention any major "
                    "uncertainty, missing evidence, or reason the score is not "
                    "higher.\n\n"
                    "Education entries should be one item per credential. Use the "
                    "best available non-PII school/program facts only. Completion "
                    "should be either graduated or years completed in brackets, such "
                    "as (3 years completed). Work experience fit must be quick bullet "
                    "points. Relevant projects should have only project name plus a "
                    "brief summary and role relevance.\n\n"
                    "The prescreen email must be in a fixed, ready-to-send format. "
                    "Questions should be grounded in this candidate's resume and hard "
                    "to answer well with generic AI text: ask for concrete examples, "
                    "specific tradeoffs, implementation details, numbers, decisions, "
                    "or lessons learned.\n\n"
                    "Set located_in_canada to true only when the resume explicitly "
                    "indicates the candidate is Canada-based. Store only the boolean; "
                    "do not include city, address, postal code, email, phone, URLs, "
                    "or the candidate name anywhere in the response. The app will "
                    "assign the final advance/hold recommendation later by ranking "
                    "all reviewed candidates in this project.\n\n"
                    f"Candidate metadata:\n{json.dumps(metadata, indent=2)}\n\n"
                    f"Job posting:\n{job_posting}\n\n"
                    f"Additional work context:\n{work_context or '(none provided)'}\n\n"
                    f"Resume markdown:\n{resume_markdown}"
                ),
            },
        ]
        response = _guarded_llm_call(
            client,
            messages=messages,
            settings={
                "model_size": "small",
                "max_tokens": 5000,
                "response_format": {"type": "json_object"},
            },
        )
        payload = parse_json_response(response, "resume review")
        return _normalize_review_payload(_remove_pii_metadata(payload))

    def _final_rerank_payload(
        self,
        client: LLMClient,
        candidates: list[dict[str, Any]],
        job_posting: str,
        work_context: str,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are the final hiring manager judge for a shortlist. You compare "
            "the top candidates against each other after they have already been "
            "reviewed individually. This is the only stage with a holistic view "
            "of the top candidates together, so use it to make the final "
            "comparative ordering and identify the strongest top-six screen "
            "group. Use the job posting and work context below as the decision "
            "frame. Focus on role fit, evidence quality, delivery risk, level "
            "fit, practical readiness, and non-obvious strengths. Start from "
            "the original standalone rank and scores, but you may move "
            "candidates up or down when direct comparison shows stronger "
            "evidence, better level fit, stronger role-specific context, or "
            "higher risk than the standalone review captured. Apply work-context "
            "preferences only when supported by resume evidence and explain any "
            "material movement. Do not infer protected characteristics, "
            "citizenship, nationality, age, gender, or hidden PII. Do not use "
            "candidate names; candidates are identified only by candidate_key.\n\n"
            f"Job posting:\n{job_posting}\n\n"
            f"Additional work context:\n{work_context or '(none provided)'}\n\n"
            "Return strict JSON only. Return every candidate_key exactly once. "
            "Adjusted ranks must be unique integers starting at 1, where 1 is the "
            "strongest final recommendation after comparative review. Adjusted "
            "ranks 1-6 are the final recommended prescreen group; ranks 7 and "
            "below are below the final screen line. If fewer than six candidates "
            "are truly credible, still rank all candidates but state that concern "
            "in overall_notes rather than inflating weak evidence. For each "
            "candidate, write a decision_summary that is a concise, standalone "
            "thesis statement explaining the positioning in plain hiring-manager "
            "language. It should be quick to read, specific enough to be useful, "
            "and not buried in extra detail."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Final-rerank these candidates. Each record contains the "
                    "standalone report output and the redacted markdown resume "
                    "used for the original review. Return JSON with this schema:\n"
                    "{\n"
                    '  "adjusted_rankings": [\n'
                    '    {"candidate_key": "", "adjusted_rank": 1, '
                    '"decision_summary": "", "rationale": "", '
                    '"relative_strengths": "", '
                    '"relative_risks": ""}\n'
                    "  ],\n"
                    '  "overall_notes": ""\n'
                    "}\n\n"
                    "decision_summary should be 1-2 sentences, usually under "
                    "60 words. It must state the concise conclusion for that "
                    "candidate's placement without requiring the reader to inspect "
                    "the full report first. Rationale should explain the main "
                    "comparative decision, including whether the candidate is "
                    "inside or outside the final top-six screen group and why. "
                    "relative_strengths and relative_risks should focus on "
                    "differences that matter against the other top candidates, "
                    "not a generic summary.\n\n"
                    f"Candidates:\n{json.dumps(candidates, indent=2)}"
                ),
            },
        ]
        response = _guarded_llm_call(
            client,
            messages=messages,
            settings={
                "model_size": "small",
                "response_format": {"type": "json_object"},
            },
        )
        payload = parse_json_response(response, "final rerank")
        return _normalize_final_rerank_payload(payload, candidates)

    def _combine_page_markdown(
        self,
        page_markdown: list[str],
        original_filename: str,
    ) -> str:
        sections = [f"# Resume: {original_filename}", ""]
        for index, markdown in enumerate(page_markdown, start=1):
            sections.extend([f"## Page {index}", markdown.strip(), ""])
        return "\n".join(sections).strip() + "\n"


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fenced responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response was not a JSON object")
    return payload


def parse_json_response(response: dict[str, Any], workflow_name: str) -> dict[str, Any]:
    """Parse a JSON object response with useful LLM diagnostics."""
    text = extract_message_text(response).strip()
    if not text:
        raise ValueError(
            f"{workflow_name} returned an empty LLM response "
            f"({_response_diagnostics(response)}). Increase the model "
            "completion-token limit or reduce the prompt size."
        )
    try:
        return parse_json_object(text)
    except json.JSONDecodeError as exc:
        preview = text[:500].replace("\n", "\\n")
        raise ValueError(
            f"{workflow_name} returned non-JSON text "
            f"({_response_diagnostics(response)}): {preview!r}"
        ) from exc


def _response_diagnostics(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return "choices=0"
    first = choices[0] or {}
    finish_reason = first.get("finish_reason", "")
    message = first.get("message") or {}
    content = message.get("content")
    content_type = type(content).__name__
    refusal = message.get("refusal")
    details = [
        f"finish_reason={finish_reason or 'unknown'}",
        f"content_type={content_type}",
    ]
    if refusal:
        details.append("refusal_present=true")
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is not None:
        details.append(f"prompt_tokens={prompt_tokens}")
    if completion_tokens is not None:
        details.append(f"completion_tokens={completion_tokens}")
    return ", ".join(details)


def _guarded_llm_call(
    client: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _llm_call_slot():
        return client.call(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            settings=settings,
        )


@contextlib.contextmanager
def _llm_call_slot() -> Any:
    semaphore = _llm_call_semaphore()
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _llm_call_semaphore() -> threading.BoundedSemaphore:
    global _LLM_SEMAPHORE, _LLM_SEMAPHORE_LIMIT
    limit = _env_int("APP_MAX_PARALLEL_LLM_CALLS", DEFAULT_PARALLEL_LLM_CALLS)
    with _LLM_SEMAPHORE_LOCK:
        if _LLM_SEMAPHORE is None or _LLM_SEMAPHORE_LIMIT != limit:
            _LLM_SEMAPHORE = threading.BoundedSemaphore(limit)
            _LLM_SEMAPHORE_LIMIT = limit
        return _LLM_SEMAPHORE


def _worker_count(env_name: str, default: int, item_count: int) -> int:
    if item_count <= 1:
        return 1
    return min(item_count, _env_int(env_name, default))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value}")
    return value


def render_review_markdown(payload: dict[str, Any]) -> str:
    """Render a structured review payload into markdown."""
    lines = [
        "# Candidate Review",
        "",
        "## Scores",
        f"- Education: {_score(payload.get('education_score'))}/10",
        f"- Work experience: {_score(payload.get('experience_score'))}/10",
        f"- Relevant projects: {_score(payload.get('projects_score'))}/10",
        f"- Aggregate score: {_score(payload.get('aggregate_score'))}/10",
        f"- Holistic judgment score: {_score(payload.get('holistic_score'))}/10",
        f"- Screening average: {_score(payload.get('screening_average'))}/10",
        f"- Located in Canada: {_yes_no(payload.get('located_in_canada'))}",
        f"- Recommendation: {payload.get('recommendation', '')}",
    ]
    score_rationale = _score_rationale_markdown(payload)
    if score_rationale:
        lines.extend(["", "## Score Rationale", *score_rationale])
    lines.extend(
        [
            "",
            "## Tradeoff Analysis",
            str(payload.get("tradeoff_analysis", "")).strip(),
            "",
            "## Education Fit",
        ]
    )
    education_entries = payload.get("education_entries") or []
    if education_entries:
        for entry in education_entries:
            lines.extend(
                [
                    f"### {_education_title(entry)}",
                    str(entry.get("fit_summary", "")).strip(),
                    "",
                ]
            )
    else:
        lines.extend([str(payload.get("education_summary", "")).strip(), ""])
    lines.extend(
        [
            "## Work Experience Fit",
        ]
    )
    work_bullets = payload.get("work_experience_fit_bullets") or []
    if work_bullets:
        lines.extend(f"- {bullet}" for bullet in work_bullets)
    else:
        lines.append(str(payload.get("experience_summary", "")).strip())
    lines.extend(
        [
            "",
            "## Highly Relevant Projects",
        ]
    )
    projects = payload.get("relevant_projects") or []
    if projects:
        for project in projects:
            summary = project.get("summary") or project.get("evidence", "")
            relevance = project.get("role_relevance", "")
            lines.extend(
                [
                    f"### {project.get('name') or 'Project'}",
                    f"- Score: {_score(project.get('score'))}/10",
                    f"- Summary: {summary}",
                    f"- Relevance: {relevance}",
                ]
            )
    else:
        lines.append("No highly relevant projects were identified.")

    lines.extend(["", "## Unique Standout Signals"])
    standouts = payload.get("unique_standouts") or []
    if standouts:
        for standout in standouts:
            lines.append(
                "- "
                f"{standout.get('signal', '')}: "
                f"{standout.get('why_it_matters', '')} "
                f"(confidence: {standout.get('confidence', '')})"
            )
    else:
        lines.append("No unique standout signals were identified.")

    lines.extend(["", "## Gaps, Risks, And Follow-Ups"])
    gaps = payload.get("gaps_and_risks") or []
    if gaps:
        for gap in gaps:
            lines.append(
                "- "
                f"{gap.get('gap', '')} "
                f"(severity: {gap.get('severity', '')}). "
                f"Follow-up: {gap.get('screening_follow_up', '')}"
            )
    else:
        lines.append("No major gaps were identified.")

    if payload.get("project_recommendation_rationale"):
        lines.extend(
            [
                "",
                "## Project Rank Recommendation",
                str(payload.get("project_recommendation_rationale", "")).strip(),
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation Rationale",
            str(payload.get("recommendation_rationale", "")).strip(),
            "",
            "## Prescreen Email",
            f"Subject: {payload.get('prescreen_email_subject', '')}",
            "",
            str(payload.get("prescreen_email_body", "")).strip(),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _remove_pii_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove contact and location fields from model metadata."""
    pii_keys = {
        "address",
        "email",
        "github",
        "linkedin",
        "location",
        "phone",
        "portfolio",
        "portfolio_or_github",
        "postal_code",
        "website",
        "candidate_name",
        "name",
    }
    return {key: value for key, value in payload.items() if key not in pii_keys}


def _score_rationale_markdown(payload: dict[str, Any]) -> list[str]:
    rationale = payload.get("score_rationale") or {}
    if not isinstance(rationale, dict):
        return []
    lines = []
    for label, key in (
        ("Aggregate formula", "aggregate_formula"),
        ("Holistic adjustments", "holistic_adjustments"),
        ("Calibration notes", "calibration_notes"),
    ):
        value = str(rationale.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize review score precision and boolean-only location signal."""
    normalized = dict(payload)
    for key in (
        "education_score",
        "experience_score",
        "projects_score",
        "aggregate_score",
        "holistic_score",
    ):
        if key in normalized:
            normalized[key] = _normalize_score(normalized[key])

    aggregate_score = _aggregate_score(normalized)
    if aggregate_score is not None:
        normalized["aggregate_score"] = aggregate_score
        _set_computed_score_rationale(normalized, aggregate_score)
    if "holistic_score" in normalized and aggregate_score is not None:
        original_holistic = normalized["holistic_score"]
        bounded_holistic = _bounded_holistic_score(
            normalized["holistic_score"],
            aggregate_score,
        )
        normalized["holistic_score"] = bounded_holistic
        if original_holistic != bounded_holistic:
            _append_calibration_note(
                normalized,
                "App scoring guardrail bounded holistic_score to within "
                "1.5 points of aggregate_score.",
            )

    projects = normalized.get("relevant_projects")
    if isinstance(projects, list):
        normalized_projects = []
        for project in projects:
            if isinstance(project, dict):
                project = dict(project)
                if "score" in project:
                    project["score"] = _normalize_score(project["score"])
            normalized_projects.append(project)
        normalized["relevant_projects"] = normalized_projects

    normalized["located_in_canada"] = _boolish(
        normalized.get("located_in_canada", False)
    )
    normalized["screening_average"] = _screening_average(
        normalized.get("aggregate_score"),
        normalized.get("holistic_score"),
    )
    return normalized


def _normalize_final_rerank_payload(
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_keys = [
        str(candidate.get("candidate_key") or "")
        for candidate in candidates
    ]
    allowed = set(candidate_keys)
    used_ranks: set[int] = set()
    normalized_records = []
    raw_records = payload.get("adjusted_rankings") or []
    if not isinstance(raw_records, list):
        raw_records = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        candidate_key = str(record.get("candidate_key") or "").strip()
        if candidate_key not in allowed:
            continue
        rank = _positive_int(record.get("adjusted_rank"))
        if rank is None or rank in used_ranks:
            continue
        used_ranks.add(rank)
        normalized_records.append(
            {
                "candidate_key": candidate_key,
                "adjusted_rank": rank,
                "decision_summary": _decision_summary(record),
                "rationale": str(record.get("rationale") or "").strip(),
                "relative_strengths": str(
                    record.get("relative_strengths") or ""
                ).strip(),
                "relative_risks": str(record.get("relative_risks") or "").strip(),
            }
        )

    ranked_keys = {record["candidate_key"] for record in normalized_records}
    next_rank = 1
    for candidate_key in candidate_keys:
        if candidate_key in ranked_keys:
            continue
        while next_rank in used_ranks:
            next_rank += 1
        used_ranks.add(next_rank)
        normalized_records.append(
            {
                "candidate_key": candidate_key,
                "adjusted_rank": next_rank,
                "decision_summary": (
                    "No model ranking was returned for this candidate."
                ),
                "rationale": "No model ranking was returned for this candidate.",
                "relative_strengths": "",
                "relative_risks": "",
            }
        )
    normalized_records.sort(key=lambda record: record["adjusted_rank"])
    return {
        "adjusted_rankings": normalized_records,
        "overall_notes": str(payload.get("overall_notes") or "").strip(),
    }


def _decision_summary(record: dict[str, Any]) -> str:
    summary = str(record.get("decision_summary") or "").strip()
    if summary:
        return summary
    return str(record.get("rationale") or "").strip()


def _normalize_score(value: Any) -> Any:
    score = _decimal_score(value)
    if score is None:
        return value
    return float(score.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def _aggregate_score(payload: dict[str, Any]) -> float | None:
    weighted = Decimal("0")
    for key, weight in AGGREGATE_WEIGHTS.items():
        score = _decimal_score(payload.get(key))
        if score is None:
            return None
        weighted += score * weight
    return float(weighted.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def _set_computed_score_rationale(
    payload: dict[str, Any],
    aggregate_score: float,
) -> None:
    rationale = payload.get("score_rationale") or {}
    if not isinstance(rationale, dict):
        rationale = {}
    else:
        rationale = dict(rationale)
    rationale["aggregate_formula"] = (
        "Computed by app as 25% education "
        f"({_score(payload.get('education_score'))}/10) + 45% experience "
        f"({_score(payload.get('experience_score'))}/10) + 30% projects "
        f"({_score(payload.get('projects_score'))}/10) = "
        f"{_score(aggregate_score)}/10."
    )
    payload["score_rationale"] = rationale


def _append_calibration_note(payload: dict[str, Any], note: str) -> None:
    rationale = payload.get("score_rationale") or {}
    if not isinstance(rationale, dict):
        rationale = {}
    else:
        rationale = dict(rationale)
    existing = str(rationale.get("calibration_notes") or "").strip()
    rationale["calibration_notes"] = " ".join(
        part for part in (existing, note) if part
    )
    payload["score_rationale"] = rationale


def _bounded_holistic_score(value: Any, aggregate: Any) -> Any:
    holistic = _decimal_score(value)
    aggregate_score = _decimal_score(aggregate)
    if holistic is None or aggregate_score is None:
        return value
    lower_bound = max(Decimal("0"), aggregate_score - HOLISTIC_ADJUSTMENT_LIMIT)
    upper_bound = min(Decimal("10"), aggregate_score + HOLISTIC_ADJUSTMENT_LIMIT)
    bounded = min(max(holistic, lower_bound), upper_bound)
    return float(bounded.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _education_title(entry: dict[str, Any]) -> str:
    university = str(entry.get("university") or "Education").strip()
    level = str(entry.get("level") or "").strip()
    completion = str(entry.get("completion") or "").strip()
    program = str(entry.get("program") or "").strip()
    gpa = str(entry.get("gpa") or "").strip()
    parts = [university]
    if level:
        parts.append(f"{level} {completion}".strip())
    if program:
        parts.append(program)
    if gpa:
        parts.append(f"GPA: {gpa}")
    return " | ".join(parts)


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _screening_average(aggregate: Any, holistic: Any) -> float | None:
    aggregate_score = _decimal_score(aggregate)
    holistic_score = _decimal_score(holistic)
    if aggregate_score is None or holistic_score is None:
        return None
    average = (aggregate_score + holistic_score) / Decimal("2")
    return float(average.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def _decimal_score(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return score if score.is_finite() else None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "canada"}
    return False


def _yes_no(value: Any) -> str:
    return "Yes" if _boolish(value) else "No"


def _score(value: Any) -> str:
    if value in (None, ""):
        return ""
    score = _decimal_score(value)
    if score is None:
        return str(value)
    return str(score.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))

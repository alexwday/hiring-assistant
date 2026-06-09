"""LLM workflows for resume extraction and candidate review."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from connections.llm_connector import LLMClient, extract_message_text
from utilities.config import AppConfig
from utilities.ssl_setup import SSLSetupResult

logger = logging.getLogger(__name__)

MARKDOWN_PROMPT_VERSION = "resume-page-markdown-v1"
METADATA_PROMPT_VERSION = "resume-metadata-v1"
REVIEW_PROMPT_VERSION = "resume-job-review-v2"
ONE_DECIMAL = Decimal("0.1")


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
        client = LLMClient(config=self.config, ssl_setup=self.ssl_setup)
        try:
            page_markdown = []
            for index, page_path in enumerate(page_paths, start=1):
                page_markdown.append(
                    self._extract_page_markdown(
                        client=client,
                        page_path=page_path,
                        original_filename=original_filename,
                        page_number=index,
                        page_count=len(page_paths),
                    )
                )
            markdown = self._combine_page_markdown(page_markdown, original_filename)
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
                    "You convert resume PDF page images into clean markdown for "
                    "downstream LLM ingestion. Preserve factual content, "
                    "dates, employers, education, skills, projects, publications, "
                    "and certifications. Normalize layout into readable headings "
                    "and bullet lists. Do not infer missing facts or recreate "
                    "redacted names or contact details. "
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
        response = client.call(
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
                    "Extract resume metadata as strict JSON. Include only values "
                    "supported by the resume text. Use empty strings for unknown "
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
        response = client.call(
            messages=messages,
            settings={
                "model_size": "small",
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
        )
        text = extract_message_text(response)
        payload = parse_json_object(text)
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
                    "You are a rigorous hiring-manager screening assistant. Evaluate "
                    "only job-relevant evidence from the resume, job posting, and "
                    "work context. Do not infer protected characteristics. Look for "
                    "direct fit, transferable experience, project evidence, practical "
                    "execution, and non-obvious strengths that could make a candidate "
                    "effective even if their profile is non-traditional. Be specific "
                    "about evidence, gaps, and uncertainty. Return strict JSON only."
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
                    "Required JSON schema:\n"
                    "{\n"
                    '  "education_summary": "",\n'
                    '  "education_score": 0.0,\n'
                    '  "experience_summary": "",\n'
                    '  "experience_score": 0.0,\n'
                    '  "relevant_projects": [\n'
                    '    {"name": "", "evidence": "", '
                    '"role_relevance": "", "score": 0.0}\n'
                    "  ],\n"
                    '  "projects_score": 0.0,\n'
                    '  "unique_standouts": [\n'
                    '    {"signal": "", "why_it_matters": "", "confidence": ""}\n'
                    "  ],\n"
                    '  "gaps_and_risks": [\n'
                    '    {"gap": "", "screening_follow_up": "", "severity": ""}\n'
                    "  ],\n"
                    '  "tradeoff_analysis": "",\n'
                    '  "aggregate_score": 0.0,\n'
                    '  "holistic_score": 0.0,\n'
                    '  "located_in_canada": false,\n'
                    '  "recommendation": "",\n'
                    '  "recommendation_rationale": "",\n'
                    '  "prescreen_email_subject": "",\n'
                    '  "prescreen_email_body": "",\n'
                    '  "hiring_manager_notes": []\n'
                    "}\n\n"
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
        response = client.call(
            messages=messages,
            settings={
                "model_size": "small",
                "max_tokens": 5000,
                "response_format": {"type": "json_object"},
            },
        )
        text = extract_message_text(response)
        payload = parse_json_object(text)
        return _normalize_review_payload(_remove_pii_metadata(payload))

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
        "",
        "## Education Fit",
        str(payload.get("education_summary", "")).strip(),
        "",
        "## Work Experience Fit",
        str(payload.get("experience_summary", "")).strip(),
        "",
        "## Highly Relevant Projects",
    ]
    projects = payload.get("relevant_projects") or []
    if projects:
        for project in projects:
            lines.extend(
                [
                    f"### {project.get('name') or 'Project'}",
                    f"- Score: {_score(project.get('score'))}/10",
                    f"- Evidence: {project.get('evidence', '')}",
                    f"- Role relevance: {project.get('role_relevance', '')}",
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

    lines.extend(
        [
            "",
            "## Tradeoff Analysis",
            str(payload.get("tradeoff_analysis", "")).strip(),
        ]
    )
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
            "",
            "## Hiring Manager Notes",
        ]
    )
    notes = payload.get("hiring_manager_notes") or []
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No additional notes.")
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


def _normalize_score(value: Any) -> Any:
    score = _decimal_score(value)
    if score is None:
        return value
    return float(score.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


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

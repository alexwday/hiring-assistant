"""Local PII screening and PDF redaction helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import fitz

EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}"
)
URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|gitlab\.com|"
    r"bitbucket\.org|[\w.-]+\.(?:com|ca|io|dev|me|net|org))/[^\s)]+",
    re.IGNORECASE,
)


def detect_pii_boxes(
    pdf_path: Path,
    candidate_names: list[str] | None = None,
) -> dict[str, Any]:
    """Detect likely PII boxes in a PDF without sending data externally."""
    document = fitz.open(pdf_path)
    try:
        pages = []
        detections = []
        normalized_names = _normalized_candidate_names(candidate_names or [])
        for page_index, page in enumerate(document):
            rect = page.rect
            page_info = {
                "page_index": page_index,
                "width": rect.width,
                "height": rect.height,
            }
            pages.append(page_info)
            detections.extend(
                _detect_page_pii(
                    page=page,
                    page_info=page_info,
                    candidate_names=normalized_names,
                    page_index=page_index,
                )
            )
        return {"pages": pages, "detections": detections}
    finally:
        document.close()


def apply_redactions(
    source_pdf: Path,
    target_pdf: Path,
    redactions: list[dict[str, Any]],
) -> None:
    """Apply selected normalized redaction boxes and save a new PDF."""
    document = fitz.open(source_pdf)
    try:
        by_page: dict[int, list[dict[str, Any]]] = {}
        for redaction in redactions:
            page_index = int(redaction.get("page_index", -1))
            if page_index < 0:
                continue
            by_page.setdefault(page_index, []).append(redaction)

        for page_index, page_redactions in by_page.items():
            if page_index >= len(document):
                continue
            page = document[page_index]
            rect = page.rect
            for redaction in page_redactions:
                box = _normalized_to_rect(redaction, rect.width, rect.height)
                if box.is_empty or box.width < 1 or box.height < 1:
                    continue
                page.add_redact_annot(box, fill=(0, 0, 0))
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

        target_pdf.parent.mkdir(parents=True, exist_ok=True)
        document.save(target_pdf, garbage=4, deflate=True)
    finally:
        document.close()


def delete_paths(paths: list[Path]) -> None:
    """Delete files or directories when they exist."""
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def _detect_page_pii(
    page: fitz.Page,
    page_info: dict[str, Any],
    candidate_names: list[str],
    page_index: int,
) -> list[dict[str, Any]]:
    words = page.get_text("words", sort=True)
    line_map: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for word in words:
        key = (int(word[5]), int(word[6]))
        line_map.setdefault(key, []).append(word)

    detections: list[dict[str, Any]] = []
    for line_number, line_words in enumerate(line_map.values()):
        line_words = sorted(line_words, key=lambda item: item[7])
        line_text = " ".join(str(word[4]) for word in line_words)
        line_box = _bbox_for_words(line_words)
        if _line_matches_candidate_name(line_text, candidate_names) or (
            page_index == 0 and line_number == 0 and _line_looks_like_name(line_text)
        ):
            detections.append(_detection(page_info, line_box, "Candidate name", "line"))
            continue
        detections.extend(_detect_line_contact_items(line_words, page_info))
    return _dedupe_detections(detections)


def _normalized_candidate_names(values: list[str]) -> list[str]:
    names = []
    for value in values:
        normalized = _normalize_name(value)
        if normalized and " " in normalized:
            names.append(normalized)
    return names


def _line_matches_candidate_name(text: str, candidate_names: list[str]) -> bool:
    normalized = _normalize_name(text)
    return any(name in normalized for name in candidate_names)


def _line_looks_like_name(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if any(
        marker in lowered
        for marker in (
            "@",
            "address",
            "candidate",
            "curriculum",
            "email",
            "linkedin",
            "phone",
            "resume",
            "résumé",
            "summary",
            "www.",
        )
    ):
        return False
    tokens = [token.strip(".,:;()[]{}") for token in cleaned.split()]
    if not 2 <= len(tokens) <= 5:
        return False
    alpha_tokens = [
        token
        for token in tokens
        if re.fullmatch(r"[A-Za-z][A-Za-z'-]+", token)
    ]
    return len(alpha_tokens) == len(tokens)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z]+", " ", value.lower()).strip()


def _detect_line_contact_items(
    words: list[tuple[Any, ...]],
    page_info: dict[str, Any],
) -> list[dict[str, Any]]:
    line_text, spans = _line_text_with_spans(words)
    detections = []
    for label, pattern in (
        ("Email", EMAIL_RE),
        ("Phone", PHONE_RE),
        ("Web address", URL_RE),
    ):
        for match in pattern.finditer(line_text):
            matched_words = [
                word
                for word, start, end in spans
                if start < match.end() and end > match.start()
            ]
            if matched_words:
                detections.append(
                    _detection(
                        page_info,
                        _bbox_for_words(matched_words),
                        label,
                        "regex",
                    )
                )
    return detections


def _line_text_with_spans(
    words: list[tuple[Any, ...]],
) -> tuple[str, list[tuple[tuple[Any, ...], int, int]]]:
    parts = []
    spans = []
    cursor = 0
    for word in words:
        if parts:
            parts.append(" ")
            cursor += 1
        text = str(word[4])
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((word, start, cursor))
    return "".join(parts), spans


def _bbox_for_words(words: list[tuple[Any, ...]]) -> tuple[float, float, float, float]:
    return (
        min(float(word[0]) for word in words),
        min(float(word[1]) for word in words),
        max(float(word[2]) for word in words),
        max(float(word[3]) for word in words),
    )


def _detection(
    page_info: dict[str, Any],
    bbox: tuple[float, float, float, float],
    label: str,
    source: str,
) -> dict[str, Any]:
    width = float(page_info["width"])
    height = float(page_info["height"])
    x0, y0, x1, y1 = bbox
    padding_x = 1.5
    inset_y = min(2.0, max(0.0, (y1 - y0) / 4))
    return {
        "page_index": page_info["page_index"],
        "x0": max(0.0, (x0 - padding_x) / width),
        "y0": max(0.0, (y0 + inset_y) / height),
        "x1": min(1.0, (x1 + padding_x) / width),
        "y1": min(1.0, (y1 - inset_y) / height),
        "label": label,
        "source": source,
    }


def _dedupe_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for detection in detections:
        key = (
            detection["page_index"],
            round(detection["x0"], 3),
            round(detection["y0"], 3),
            round(detection["x1"], 3),
            round(detection["y1"], 3),
            detection["label"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(detection)
    return deduped


def _normalized_to_rect(
    redaction: dict[str, Any],
    width: float,
    height: float,
) -> fitz.Rect:
    x0 = float(redaction.get("x0", 0.0)) * width
    y0 = float(redaction.get("y0", 0.0)) * height
    x1 = float(redaction.get("x1", 0.0)) * width
    y1 = float(redaction.get("y1", 0.0)) * height
    return fitz.Rect(
        min(x0, x1),
        min(y0, y1),
        max(x0, x1),
        max(y0, y1),
    )

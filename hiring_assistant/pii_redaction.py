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
CANADIAN_POSTAL_RE = re.compile(
    r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b",
    re.IGNORECASE,
)
US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|gitlab\.com|"
    r"bitbucket\.org|[\w.-]+\.(?:com|ca|io|dev|me|net|org))/[^\s)]+",
    re.IGNORECASE,
)
STREET_RE = re.compile(
    r"\b\d{1,6}\s+[\w .'-]+"
    r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|"
    r"boulevard|blvd\.?|court|ct\.?|crescent|cres\.?|way|place|pl\.?)\b",
    re.IGNORECASE,
)
REGION_RE = re.compile(
    r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*"
    r"(?:AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT|"
    r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|"
    r"MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|"
    r"TX|UT|VA|VT|WA|WI|WV|WY)\b"
)


def detect_pii_boxes(pdf_path: Path) -> dict[str, Any]:
    """Detect likely PII boxes in a PDF without sending data externally."""
    document = fitz.open(pdf_path)
    try:
        pages = []
        detections = []
        for page_index, page in enumerate(document):
            rect = page.rect
            page_info = {
                "page_index": page_index,
                "width": rect.width,
                "height": rect.height,
            }
            pages.append(page_info)
            detections.extend(_detect_page_pii(page, page_info))
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
        line_is_contact = _line_has_contact_pii(line_text)
        line_is_address = _line_has_address_pii(line_text, line_number)
        if line_is_contact or line_is_address:
            label = "Contact line" if line_is_contact else "Address or location"
            detections.append(_detection(page_info, line_box, label, "line"))
            continue

        for word in line_words:
            word_text = str(word[4])
            label = _word_pii_label(word_text)
            if label:
                detections.append(_detection(page_info, word[:4], label, "word"))
    return _dedupe_detections(detections)


def _line_has_contact_pii(text: str) -> bool:
    lowered = text.lower()
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or URL_RE.search(text)
        or "linkedin" in lowered
    )


def _line_has_address_pii(text: str, line_number: int) -> bool:
    if CANADIAN_POSTAL_RE.search(text) or STREET_RE.search(text):
        return True
    if line_number <= 12 and REGION_RE.search(text):
        return True
    return bool(line_number <= 12 and US_ZIP_RE.search(text))


def _word_pii_label(text: str) -> str:
    lowered = text.lower()
    if EMAIL_RE.search(text):
        return "Email"
    if URL_RE.search(text) or "linkedin" in lowered:
        return "Profile URL"
    if CANADIAN_POSTAL_RE.search(text) or US_ZIP_RE.search(text):
        return "Postal code"
    return ""


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

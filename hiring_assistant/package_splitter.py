"""Local PDF package analysis and resume splitting helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from hiring_assistant.store import slugify


@dataclass(frozen=True)
class SplitPdf:
    """One split resume PDF created from a larger package."""

    filename: str
    path: Path
    candidate_name: str
    candidate_id: str
    start_page: int
    end_page: int


def analyze_resume_package(pdf_path: Path, index_pages: int) -> dict[str, Any]:
    """Infer candidate resume ranges from index-page attachment links."""
    document = fitz.open(pdf_path)
    try:
        page_count = len(document)
        normalized_index_pages = max(1, min(int(index_pages or 1), page_count))
        rows = _linked_candidate_rows(document, normalized_index_pages)
        warnings = []
        if not rows:
            warnings.append(
                "No internal attachment links were detected; add page ranges manually."
            )
        rows = _fill_end_pages(rows, page_count)
        return {
            "page_count": page_count,
            "index_pages": normalized_index_pages,
            "candidate_rows": rows,
            "warnings": warnings,
        }
    finally:
        document.close()


def split_resume_package(
    source_pdf: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
    index_pages: int,
) -> list[SplitPdf]:
    """Create one PDF per selected candidate page range."""
    document = fitz.open(source_pdf)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        page_count = len(document)
        used_filenames: set[str] = set()
        created = []
        for ordinal, row in enumerate(rows, start=1):
            start_page = int(row.get("start_page") or 0)
            end_page = int(row.get("end_page") or 0)
            if start_page <= int(index_pages):
                raise ValueError("Split ranges must start after the index pages")
            if start_page < 1 or end_page > page_count or start_page > end_page:
                raise ValueError(
                    f"Invalid page range {start_page}-{end_page} "
                    f"for a {page_count}-page PDF"
                )

            filename = _unique_filename(
                _candidate_filename(
                    candidate_name=str(row.get("candidate_name") or ""),
                    candidate_id=str(row.get("candidate_id") or ""),
                    ordinal=ordinal,
                ),
                used_filenames,
                output_dir,
            )
            target = output_dir / filename
            split_doc = fitz.open()
            try:
                split_doc.insert_pdf(
                    document,
                    from_page=start_page - 1,
                    to_page=end_page - 1,
                )
                split_doc.save(target, garbage=4, deflate=True)
            finally:
                split_doc.close()

            created.append(
                SplitPdf(
                    filename=filename,
                    path=target,
                    candidate_name=str(row.get("candidate_name") or ""),
                    candidate_id=str(row.get("candidate_id") or ""),
                    start_page=start_page,
                    end_page=end_page,
                )
            )
        return created
    finally:
        document.close()


def _linked_candidate_rows(
    document: fitz.Document,
    index_pages: int,
) -> list[dict[str, Any]]:
    rows = []
    seen_targets: set[int] = set()
    for page_index in range(index_pages):
        page = document[page_index]
        for link in sorted(page.get_links(), key=_link_sort_key):
            target_page = _link_target_page(link)
            if target_page is None or target_page < index_pages:
                continue
            if target_page in seen_targets:
                continue
            seen_targets.add(target_page)
            link_rect = fitz.Rect(link.get("from"))
            row_words = _words_on_link_row(page, link_rect)
            left_words = [
                word for word in row_words if float(word[0]) < link_rect.x0 - 1
            ]
            row_text = _words_text(row_words)
            source_text = _words_text(left_words) if left_words else row_text
            parsed = _parse_candidate_reference(source_text, len(rows) + 1)
            if not parsed["candidate_id"]:
                candidate_id = _extract_candidate_id_from_text(
                    row_text,
                    parsed["candidate_name"],
                )
                if candidate_id:
                    parsed["candidate_id"] = candidate_id
                    parsed["source_text"] = " ".join(
                        part
                        for part in (
                            parsed["candidate_name"],
                            parsed["candidate_id"],
                        )
                        if part
                    )
                    parsed["confidence"] = "high"
            rows.append(
                {
                    "candidate_name": parsed["candidate_name"],
                    "candidate_id": parsed["candidate_id"],
                    "start_page": target_page + 1,
                    "end_page": target_page + 1,
                    "index_page": page_index + 1,
                    "source_text": parsed["source_text"],
                    "confidence": parsed["confidence"],
                }
            )
    return sorted(rows, key=lambda row: int(row["start_page"]))


def _fill_end_pages(
    rows: list[dict[str, Any]],
    page_count: int,
) -> list[dict[str, Any]]:
    filled = []
    for index, row in enumerate(rows):
        next_start = (
            int(rows[index + 1]["start_page"])
            if index + 1 < len(rows)
            else page_count + 1
        )
        updated = dict(row)
        updated["end_page"] = max(int(row["start_page"]), next_start - 1)
        filled.append(updated)
    return filled


def _link_target_page(link: dict[str, Any]) -> int | None:
    target = link.get("page")
    if target is None:
        return None
    try:
        return int(target)
    except (TypeError, ValueError):
        return None


def _link_sort_key(link: dict[str, Any]) -> tuple[float, float]:
    try:
        rect = fitz.Rect(link.get("from"))
    except Exception:
        return (0.0, 0.0)
    return (float(rect.y0), float(rect.x0))


def _words_on_link_row(page: fitz.Page, link_rect: fitz.Rect) -> list[tuple[Any, ...]]:
    words = page.get_text("words", sort=True)
    if not words:
        return []
    lines = _word_lines(words)
    if not lines:
        return []
    center_y = (link_rect.y0 + link_rect.y1) / 2
    overlapping_lines = [
        line
        for line in lines
        if _vertical_overlap(_line_bbox(line), link_rect) > 0
    ]
    candidates = overlapping_lines or lines
    best_line = min(
        candidates,
        key=lambda line: (
            abs(_line_center_y(line) - center_y),
            -_vertical_overlap(_line_bbox(line), link_rect),
        ),
    )
    return sorted(best_line, key=lambda word: (float(word[1]), float(word[0])))


def _word_lines(words: list[tuple[Any, ...]]) -> list[list[tuple[Any, ...]]]:
    groups: list[list[tuple[Any, ...]]] = []
    for word in sorted(words, key=lambda item: _word_center_y(item)):
        if groups and abs(_word_center_y(word) - _line_center_y(groups[-1])) <= 4:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [
        sorted(group, key=lambda word: (float(word[1]), float(word[0])))
        for group in groups
    ]


def _word_center_y(word: tuple[Any, ...]) -> float:
    return (float(word[1]) + float(word[3])) / 2


def _line_center_y(words: list[tuple[Any, ...]]) -> float:
    return (
        min(float(word[1]) for word in words)
        + max(float(word[3]) for word in words)
    ) / 2


def _line_bbox(words: list[tuple[Any, ...]]) -> fitz.Rect:
    return fitz.Rect(
        min(float(word[0]) for word in words),
        min(float(word[1]) for word in words),
        max(float(word[2]) for word in words),
        max(float(word[3]) for word in words),
    )


def _vertical_overlap(line_rect: fitz.Rect, link_rect: fitz.Rect) -> float:
    return max(
        0.0,
        min(float(line_rect.y1), float(link_rect.y1) + 3)
        - max(float(line_rect.y0), float(link_rect.y0) - 3),
    )


def _words_text(words: list[tuple[Any, ...]]) -> str:
    return " ".join(str(word[4]) for word in sorted(words, key=lambda word: word[0]))


def _parse_candidate_reference(text: str, ordinal: int) -> dict[str, str]:
    cleaned = _clean_row_text(text)
    tokens = cleaned.split()
    id_index = -1
    candidate_id = ""
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index].strip(",:;()[]{}")
        if _looks_like_candidate_id(token):
            id_index = index
            candidate_id = token
            break

    if id_index >= 0:
        name_tokens = _candidate_name_tokens(tokens[:id_index])
    else:
        name_tokens = _candidate_name_tokens(tokens)

    candidate_name = _clean_candidate_name(" ".join(name_tokens))
    if not candidate_name:
        candidate_name = f"Candidate {ordinal}"

    confidence = "high" if candidate_id else "medium"
    source_text = " ".join(part for part in (candidate_name, candidate_id) if part)
    return {
        "candidate_name": candidate_name,
        "candidate_id": candidate_id,
        "source_text": source_text,
        "confidence": confidence,
    }


def _candidate_name_tokens(tokens: list[str]) -> list[str]:
    cut_after = -1
    for index, token in enumerate(tokens):
        cleaned = token.strip(",:;()[]{}")
        if _looks_like_row_marker(cleaned) or _looks_like_candidate_id(cleaned):
            cut_after = index
    return tokens[cut_after + 1:]


def _extract_candidate_id_from_text(text: str, candidate_name: str) -> str:
    tokens = _clean_row_text(text).split()
    if not tokens:
        return ""
    normalized_tokens = [_normalize_token(token) for token in tokens]
    name_tokens = [
        _normalize_token(token)
        for token in candidate_name.split()
        if _normalize_token(token)
    ]
    start_index = _candidate_id_search_start(normalized_tokens, name_tokens)
    for token in tokens[start_index:]:
        cleaned = token.strip(",:;()[]{}")
        if _looks_like_candidate_id(cleaned):
            return cleaned
    for token in tokens:
        cleaned = token.strip(",:;()[]{}")
        if _looks_like_candidate_id(cleaned):
            return cleaned
    return ""


def _candidate_id_search_start(
    normalized_tokens: list[str],
    name_tokens: list[str],
) -> int:
    if not name_tokens:
        return 0
    for index in range(0, len(normalized_tokens) - len(name_tokens) + 1):
        if normalized_tokens[index: index + len(name_tokens)] == name_tokens:
            return index + len(name_tokens)
    return 0


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _clean_row_text(text: str) -> str:
    cleaned = _strip_leading_row_numbers(text)
    cleaned = re.sub(
        r"\b(candidate\s+name|candidate\s+id|attachments?|resume|pdf)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _strip_leading_row_numbers(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:|")


def _clean_candidate_name(value: str) -> str:
    cleaned = _strip_leading_row_numbers(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:|")
    cleaned = re.sub(r"\b(view|open|download|attachment)\b", "", cleaned, flags=re.I)
    cleaned = _strip_leading_row_numbers(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" -:|")


def _strip_leading_row_numbers(value: str) -> str:
    cleaned = value.strip()
    previous = ""
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"^\s*\d{1,4}\s*[\.)]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*(?:\d{1,4}\s+){1,8}(?=[A-Z])", "", cleaned)
    return cleaned


def _looks_like_row_marker(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}[\.)]?", token))


def _looks_like_candidate_id(token: str) -> bool:
    if not (3 <= len(token) <= 40):
        return False
    if not any(char.isdigit() for char in token):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", token))


def _candidate_filename(candidate_name: str, candidate_id: str, ordinal: int) -> str:
    parts = []
    if candidate_id.strip():
        parts.append(candidate_id.strip())
    if candidate_name.strip():
        parts.append(candidate_name.strip())
    stem = "-".join(slugify(part, fallback="") for part in parts if part.strip())
    return f"{stem or f'candidate-{ordinal}'}.pdf"


def _unique_filename(filename: str, used: set[str], output_dir: Path) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    candidate = f"{stem}{suffix}"
    counter = 2
    while candidate in used or (output_dir / candidate).exists():
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate

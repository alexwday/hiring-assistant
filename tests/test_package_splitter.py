"""Tests for combined PDF package analysis and splitting."""

from pathlib import Path

import fitz

from hiring_assistant.package_splitter import (
    analyze_resume_package,
    split_resume_package,
)


def test_analyze_and_split_linked_resume_package(tmp_path: Path):
    """It infers split ranges from index-page links and writes resume PDFs."""
    source_pdf = tmp_path / "package.pdf"
    _write_linked_package(source_pdf)

    analysis = analyze_resume_package(source_pdf, index_pages=1)

    rows = analysis["candidate_rows"]
    assert analysis["page_count"] == 3
    assert len(rows) == 2
    assert rows[0]["candidate_name"] == "Jordan Lee"
    assert rows[0]["candidate_id"] == "C001"
    assert rows[0]["start_page"] == 2
    assert rows[0]["end_page"] == 2
    assert rows[1]["candidate_name"] == "Priya Shah"
    assert rows[1]["candidate_id"] == "C002"
    assert rows[1]["start_page"] == 3
    assert rows[1]["end_page"] == 3

    outputs = split_resume_package(
        source_pdf=source_pdf,
        rows=rows,
        output_dir=tmp_path / "split",
        index_pages=1,
    )

    assert [output.filename for output in outputs] == [
        "c001-jordan-lee.pdf",
        "c002-priya-shah.pdf",
    ]
    first = fitz.open(outputs[0].path)
    try:
        assert len(first) == 1
        text = first[0].get_text()
        assert "Jordan Lee resume page" in text
        assert "Candidate Name" not in text
    finally:
        first.close()


def test_analyze_cleans_row_numbers_and_adjacent_index_text(tmp_path: Path):
    """It does not let adjacent numbered rows leak into the candidate name."""
    source_pdf = tmp_path / "package-with-row-numbers.pdf"
    _write_row_number_package(source_pdf)

    analysis = analyze_resume_package(source_pdf, index_pages=1)

    rows = analysis["candidate_rows"]
    assert rows[0]["candidate_name"] == "Morgan Patel"
    assert rows[0]["candidate_id"] == "C007"
    assert rows[0]["source_text"] == "Morgan Patel C007"
    assert rows[1]["candidate_name"] == "Jordan Lee"
    assert rows[1]["candidate_id"] == "C008"
    assert rows[1]["source_text"] == "Jordan Lee C008"
    assert rows[2]["candidate_name"] == "Priya Shah"
    assert rows[2]["candidate_id"] == "C009"
    assert rows[2]["source_text"] == "Priya Shah C009"


def _write_linked_package(path: Path) -> None:
    document = fitz.open()
    try:
        index = document.new_page()
        index.insert_text((72, 72), "Candidate Name  Candidate ID  Attachments")
        index.insert_text((72, 110), "Jordan Lee  C001  Resume")
        index.insert_text((72, 148), "Priya Shah  C002  Resume")

        first_resume = document.new_page()
        first_resume.insert_text((72, 72), "Jordan Lee resume page")
        second_resume = document.new_page()
        second_resume.insert_text((72, 72), "Priya Shah resume page")

        index = document[0]
        index.insert_link(
            {
                "kind": fitz.LINK_GOTO,
                "from": fitz.Rect(220, 98, 280, 118),
                "page": 1,
            }
        )
        index.insert_link(
            {
                "kind": fitz.LINK_GOTO,
                "from": fitz.Rect(220, 136, 280, 156),
                "page": 2,
            }
        )
        document.save(path)
    finally:
        document.close()


def _write_row_number_package(path: Path) -> None:
    document = fitz.open()
    try:
        index = document.new_page()
        index.insert_text((72, 72), "Candidate Name  Candidate ID  Attachments")
        index.insert_text((72, 104), "7. Morgan Patel  C007  Resume")
        index.insert_text((72, 124), "8. Jordan Lee  C008  Resume")
        index.insert_text((72, 144), "9. Priya Shah  C009  Resume")

        for name in ("Morgan Patel", "Jordan Lee", "Priya Shah"):
            page = document.new_page()
            page.insert_text((72, 72), f"{name} resume page")

        index = document[0]
        for target_page, y0 in enumerate((94, 114, 134), start=1):
            index.insert_link(
                {
                    "kind": fitz.LINK_GOTO,
                    "from": fitz.Rect(220, y0, 280, y0 + 28),
                    "page": target_page,
                }
            )
        document.save(path)
    finally:
        document.close()

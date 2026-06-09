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

"""Tests for rendered markdown document views."""

from hiring_assistant.server import _markdown_to_html


def test_markdown_renderer_formats_common_resume_sections():
    """It renders headings, bold text, bullets, and numbered items."""
    html = _markdown_to_html(
        "# Candidate Review\n\n"
        "## Scores\n"
        "- Education: **8/10**\n"
        "- Experience: 9/10\n\n"
        "1) Describe the project tradeoffs.\n"
    )

    assert "<h1>Candidate Review</h1>" in html
    assert "<h2>Scores</h2>" in html
    assert "<li>Education: <strong>8/10</strong></li>" in html
    assert "<ol>" in html


def test_markdown_renderer_escapes_raw_html():
    """It treats generated markdown as untrusted text."""
    html = _markdown_to_html("# Safe\n\n<script>alert('x')</script>")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html

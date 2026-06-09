"""Generate a synthetic resume PDF for local hiring assistant testing."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 54
TOP = 742
LEADING = 14


PAGES = [
    [
        "JORDAN LEE",
        "Toronto, ON | jordan.lee@example.com | 416-555-0198",
        "linkedin.com/in/jordan-lee-data | github.com/jordanlee-data",
        "",
        "SUMMARY",
        "Data and operations analyst with six years of experience building",
        "automation, reporting, and AI-assisted workflow tools for high-volume",
        "service teams. Known for translating messy stakeholder requirements into",
        "practical tools, clear documentation, and measurable cycle-time reductions.",
        "",
        "EDUCATION",
        "University of Waterloo - Bachelor of Mathematics, Statistics, 2018",
        "Relevant coursework: statistical modeling, databases, optimization,",
        "experimental design, business analytics.",
        "",
        "WORK EXPERIENCE",
        "Senior Business Analyst - Northstar Financial Services, 2022-Present",
        "- Built a Python and SQL intake triage tool that reduced manual case",
        "  routing time by 38 percent across a 45-person operations team.",
        "- Partnered with compliance, product, and frontline managers to define",
        "  audit-ready requirements for an internal document review workflow.",
        "- Designed dashboards that exposed backlog drivers, SLA risk, and quality",
        "  exceptions for weekly leadership reviews.",
        "- Piloted LLM-assisted summarization for policy exception notes, including",
        "  human review checkpoints and prompt/version tracking.",
        "",
        "Operations Data Analyst - MapleCare Health Network, 2019-2022",
        "- Automated Excel-heavy staffing reports using Python, VBA, and Power Query,",
        "  saving roughly 12 hours per week for regional coordinators.",
        "- Developed a queue prioritization model for intake referrals that improved",
        "  same-day review rates from 54 percent to 71 percent.",
        "- Led workshops with non-technical users to map current-state workflows and",
        "  convert pain points into implementation-ready user stories.",
    ],
    [
        "PROJECTS",
        "Resume Screening Assistant Prototype",
        "- Built a local FastAPI prototype that parsed resumes into markdown,",
        "  extracted candidate metadata, and generated job-fit review summaries.",
        "- Added duplicate detection, reviewer notes, and CSV export for shortlist",
        "  handoff. Used prompt templates to keep review criteria consistent.",
        "",
        "Policy Exception Trend Miner",
        "- Created a text classification workflow for recurring exception themes in",
        "  customer support notes. Combined keyword rules, embedding similarity, and",
        "  reviewer feedback to improve precision over three iterations.",
        "- Findings helped managers revise training material and reduce repeat",
        "  escalations by 18 percent in one quarter.",
        "",
        "SKILLS",
        "Python, SQL, pandas, Power BI, Excel, VBA, Git, APIs, prompt design,",
        "workflow mapping, stakeholder interviews, operations analytics, data",
        "quality review, documentation, risk controls.",
        "",
        "CERTIFICATIONS",
        "Microsoft Power BI Data Analyst Associate, 2023",
        "Lean Six Sigma Green Belt, 2021",
        "",
        "SELECTED ACCOMPLISHMENTS",
        "- Mentored three analysts on reproducible reporting practices.",
        "- Wrote a plain-language controls guide adopted by two adjacent teams.",
        "- Built a lightweight issue taxonomy that helped managers spot hidden",
        "  process failures before they became monthly SLA misses.",
    ],
]


def main() -> int:
    """Generate the sample PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="/private/tmp/hiring-assistant-sample/jordan_lee_resume.pdf",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_pdf(PAGES))
    print(output_path)
    return 0


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a small text-only PDF using standard PDF objects."""
    objects: list[bytes] = []
    page_object_numbers = []
    font_object_number = 3

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_lines in pages:
        content_object_number = len(objects) + 2
        page_object_number = len(objects) + 1
        page_object_numbers.append(page_object_number)
        content = page_content_stream(page_lines)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\n"
            b"stream\n" + content + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(
        "ascii"
    )

    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def page_content_stream(lines: list[str]) -> bytes:
    """Return one PDF page content stream."""
    commands = [
        "BT",
        "/F1 11 Tf",
        f"{LEADING} TL",
        f"{LEFT} {TOP} Td",
    ]
    first = True
    for line in wrap_lines(lines):
        if first:
            first = False
        else:
            commands.append("T*")
        commands.append(f"({escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def wrap_lines(lines: list[str]) -> list[str]:
    """Wrap long resume lines for the fixed-width PDF page."""
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        width = 92 if line.startswith(("  ", "- ")) else 88
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def escape_pdf_text(value: str) -> str:
    """Escape text for PDF literal strings."""
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


if __name__ == "__main__":
    raise SystemExit(main())

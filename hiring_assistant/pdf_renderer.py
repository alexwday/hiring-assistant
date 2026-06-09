"""Render uploaded resume PDFs into page images for vision processing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PDFRenderError(RuntimeError):
    """Raised when a PDF cannot be rendered into images."""


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 180,
) -> list[Path]:
    """Render a PDF into PNG page images using the local poppler toolchain."""
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise PDFRenderError(
            "pdftoppm is required to render PDF pages. Install poppler locally."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    command = [
        executable,
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise PDFRenderError(f"Failed to render PDF: {error}")

    pages = sorted(output_dir.glob("page-*.png"), key=_page_sort_key)
    if not pages:
        raise PDFRenderError("PDF rendering completed but no page images were produced")
    return pages


def _page_sort_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0

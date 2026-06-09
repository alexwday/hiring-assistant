"""Local HTML server for the hiring assistant."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import threading
import webbrowser
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from hiring_assistant.llm_workflows import ResumeLLMService
from hiring_assistant.package_splitter import (
    analyze_resume_package,
    split_resume_package,
)
from hiring_assistant.pii_redaction import (
    apply_redactions,
    delete_paths,
    detect_pii_boxes,
)
from hiring_assistant.pdf_renderer import render_pdf_pages
from hiring_assistant.store import ProjectStore, UploadedFile, relative_path, utc_now
from utilities.config import AppConfig, load_config, load_env_file
from utilities.logging_setup import setup_logging
from utilities.ssl_setup import SSLSetupResult, setup_ssl

logger = logging.getLogger(__name__)


class HiringAssistantHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying shared application dependencies."""

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        store: ProjectStore,
        config: AppConfig,
        ssl_setup: SSLSetupResult,
    ):
        super().__init__(server_address, request_handler_class)
        self.store = store
        self.config = config
        self.ssl_setup = ssl_setup


class HiringAssistantHandler(BaseHTTPRequestHandler):
    """Small stdlib HTML app for project-based resume review."""

    server: HiringAssistantHTTPServer

    def do_GET(self) -> None:
        """Route GET requests."""
        parsed = urlparse(self.path)
        segments = _path_segments(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        try:
            if not segments:
                self._send_html(self._render_index(query))
                return
            if len(segments) == 2 and segments[0] == "projects":
                self._send_html(self._render_project(segments[1], query))
                return
            if (
                len(segments) == 3
                and segments[0] == "projects"
                and segments[2] == "packages"
            ):
                self._send_html(self._render_packages(segments[1], query))
                return
            if (
                len(segments) == 4
                and segments[0] == "projects"
                and segments[2] == "packages"
            ):
                self._send_html(
                    self._render_package(
                        project_id=segments[1],
                        package_id=segments[3],
                        query=query,
                    )
                )
                return
            if (
                len(segments) == 5
                and segments[0] == "projects"
                and segments[2] == "documents"
            ):
                if segments[4] == "pii":
                    self._render_pii_review(
                        project_id=segments[1],
                        document_id=segments[3],
                    )
                    return
                self._render_document_text(
                    project_id=segments[1],
                    document_id=segments[3],
                    view=segments[4],
                )
                return
            if (
                len(segments) == 4
                and segments[0] == "projects"
                and segments[2] == "files"
            ):
                self._send_pdf(project_id=segments[1], document_id=segments[3])
                return
            if (
                len(segments) == 3
                and segments[0] == "projects"
                and segments[2] == "assets"
            ):
                self._send_project_asset(project_id=segments[1], query=query)
                return
            self._not_found()
        except KeyError as exc:
            self._send_error_page(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            logger.exception("Unhandled GET error")
            self._send_error_page(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        """Route POST requests."""
        parsed = urlparse(self.path)
        segments = _path_segments(parsed.path)

        try:
            if len(segments) == 1 and segments[0] == "reset":
                self._reset_local_data()
                return
            if len(segments) == 1 and segments[0] == "projects":
                self._create_project()
                return
            if len(segments) == 3 and segments[0] == "projects":
                project_id = segments[1]
                action = segments[2]
                if action == "upload":
                    self._upload_resumes(project_id)
                    return
                if action == "packages":
                    self._upload_package(project_id)
                    return
                if action == "context":
                    self._save_context(project_id)
                    return
                if action == "process":
                    self._process_resumes(project_id)
                    return
                if action == "auto-redact":
                    self._auto_redact_resumes(project_id)
                    return
                if action == "review":
                    self._review_resumes(project_id)
                    return
            if (
                len(segments) == 5
                and segments[0] == "projects"
                and segments[2] == "documents"
                and segments[4] == "redactions"
            ):
                self._finalize_redactions(
                    project_id=segments[1],
                    document_id=segments[3],
                )
                return
            if (
                len(segments) == 5
                and segments[0] == "projects"
                and segments[2] == "packages"
                and segments[4] == "split"
            ):
                self._split_package(
                    project_id=segments[1],
                    package_id=segments[3],
                )
                return
            self._not_found()
        except KeyError as exc:
            self._send_error_page(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            logger.exception("Unhandled POST error")
            self._send_error_page(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format_: str, *args: Any) -> None:
        """Send access logs through the app logger."""
        logger.info("HTTP %s", format_ % args)

    def _render_index(self, query: dict[str, list[str]]) -> str:
        projects = self.server.store.list_projects()
        rows = []
        for project in projects:
            href = f"/projects/{quote(project['id'])}"
            rows.append(
                "<tr>"
                f"<td><a href=\"{href}\">{_h(project['name'])}</a></td>"
                f"<td>{project.get('document_count', 0)}</td>"
                f"<td>{_h(_format_ts(project.get('updated_at', '')))}</td>"
                "</tr>"
            )
        if not rows:
            rows.append(
                "<tr><td colspan=\"3\" class=\"muted\">No projects yet.</td></tr>"
            )

        body = f"""
        <div class="app-shell">
          {_banner("Project Workspace", "", "")}
          {_flash(query)}
          <div class="app-layout">
            {_project_sidebar(projects, "")}
            <section class="main-pane">
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Projects</h2>
                    <p class="muted">Select a project to review resumes.</p>
                  </div>
                </div>
                <table>
                  <thead>
                    <tr><th>Name</th><th>Resumes</th><th>Updated</th></tr>
                  </thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </section>
            </section>
            <aside class="right-pane">
              <h2>Job Details</h2>
              <p class="muted">
                Choose or create a project before adding a job posting and
                context notes.
              </p>
              {_reset_form()}
            </aside>
          </div>
        </div>
        """
        return _page("Hiring Assistant", body)

    def _render_project(self, project_id: str, query: dict[str, list[str]]) -> str:
        project = self.server.store.load_project(project_id)
        documents = project.get("documents", [])
        uploaded = [
            doc
            for doc in documents
            if doc.get("status")
            in {"uploaded", "pii_review", "redacted", "processing"}
        ]
        processed = [
            doc
            for doc in documents
            if doc.get("status") in {"processed", "reviewing"}
        ]
        reviewed = [doc for doc in documents if doc.get("status") == "reviewed"]
        active_count = sum(
            1
            for doc in documents
            if doc.get("status") in {"processing", "reviewing"}
        )
        projects = self.server.store.list_projects()
        stats = (
            _stat("Unprocessed", len(uploaded))
            + _stat("Processed", len(processed))
            + _stat("Reviewed", len(reviewed))
        )
        upload_form = f"""
        <a class="banner-link" href="/projects/{quote(project_id)}/packages">
          Package splitter
        </a>
        <form method="post" action="/projects/{quote(project_id)}/upload"
              enctype="multipart/form-data" class="banner-upload">
          <label>
            Upload PDFs
            <input name="resumes" type="file" accept="application/pdf,.pdf"
                   multiple required>
          </label>
          <button type="submit">Upload</button>
        </form>
        """
        body = f"""
        <div class="app-shell">
          {_banner(project['name'], "Project-local resume review", stats, upload_form)}
          {_flash(query)}
          <div class="app-layout">
            {_project_sidebar(projects, project_id)}
            <section class="main-pane table-stack">
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Unprocessed Documents</h2>
                    <p class="muted">Uploaded PDFs waiting for markdown extraction.</p>
                  </div>
                  <span class="count-pill">{len(uploaded)}</span>
                </div>
                {_uploaded_table(project_id, uploaded)}
              </section>
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Processed Documents</h2>
                    <p class="muted">Markdown and candidate metadata are ready.</p>
                  </div>
                  <span class="count-pill">{len(processed)}</span>
                </div>
                {_processed_table(project_id, processed)}
              </section>
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Reviewed Resumes</h2>
                    <p class="muted">Completed job-fit reports and scores.</p>
                  </div>
                  <span class="count-pill">{len(reviewed)}</span>
                </div>
                {_reviewed_table(project_id, reviewed)}
              </section>
            </section>
            <aside class="right-pane">
              <h2>Job Details</h2>
              <form method="post" action="/projects/{quote(project_id)}/context"
                    class="context-form">
                <label>
                  Current job posting
                  <textarea name="job_posting" rows="18">{
                    _h(project.get('job_posting', ''))
                  }</textarea>
                </label>
                <label>
                  Additional work/context notes
                  <textarea name="work_context" rows="8">{
                    _h(project.get('work_context', ''))
                  }</textarea>
                </label>
                <button type="submit">Save details</button>
              </form>
              <div class="side-note">
                <strong>Status model</strong>
                <p>
                  Each resume appears in exactly one table: unprocessed,
                  processed, or reviewed.
                </p>
              </div>
              {_reset_form()}
            </aside>
          </div>
        </div>
        {_auto_refresh_script(active_count)}
        """
        return _page(project["name"], body)

    def _render_packages(
        self,
        project_id: str,
        query: dict[str, list[str]],
    ) -> str:
        project = self.server.store.load_project(project_id)
        projects = self.server.store.list_projects()
        packages = project.get("packages", [])
        rows = "".join(_package_table_row(project_id, package) for package in packages)
        if not rows:
            rows = _empty_row(7, "No package uploads yet.")

        body = f"""
        <div class="app-shell">
          {_banner(project['name'], "PDF package splitter", "")}
          {_flash(query)}
          <div class="app-layout">
            {_project_sidebar(projects, project_id)}
            <section class="main-pane table-stack">
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Upload Package</h2>
                    <p class="muted">Combined PDF with an index and linked resumes.</p>
                  </div>
                </div>
                <form method="post" action="/projects/{quote(project_id)}/packages"
                      enctype="multipart/form-data" class="package-upload-form">
                  <label>
                    Package PDF
                    <input name="package_pdf" type="file"
                           accept="application/pdf,.pdf" required>
                  </label>
                  <label>
                    Index pages
                    <input name="index_pages" type="number" min="1" value="2">
                  </label>
                  <button type="submit">Analyze package</button>
                </form>
              </section>
              <section class="table-panel">
                <div class="table-panel-header">
                  <div>
                    <h2>Packages</h2>
                    <p class="muted">Open a package to edit split ranges.</p>
                  </div>
                  <span class="count-pill">{len(packages)}</span>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>File</th><th>Status</th><th>Pages</th><th>Index</th>
                      <th>Rows</th><th>Created</th><th>Action</th>
                    </tr>
                  </thead>
                  <tbody>{rows}</tbody>
                </table>
              </section>
            </section>
            <aside class="right-pane">
              <h2>Project Tools</h2>
              <a class="side-tool" href="/projects/{quote(project_id)}">
                Resume tables
              </a>
              <div class="side-note">
                <strong>Local only</strong>
                <p>The package index is read locally and is not sent to the LLM.</p>
              </div>
            </aside>
          </div>
        </div>
        """
        return _page(f"Packages: {project['name']}", body)

    def _render_package(
        self,
        project_id: str,
        package_id: str,
        query: dict[str, list[str]],
    ) -> str:
        project = self.server.store.load_project(project_id)
        package = self.server.store.get_package(project, package_id)
        rows = package.get("candidate_rows") or []
        if not rows:
            rows = [_fallback_split_row(package)]
        row_html = "".join(
            _split_row_html(row, index, package)
            for index, row in enumerate(rows)
        )
        warnings = "".join(
            f"<div class=\"flash error\">{_h(warning)}</div>"
            for warning in package.get("warnings", [])
        )

        body = f"""
        <div class="document-shell package-shell">
          <nav class="top-nav">
            <a href="/projects/{quote(project_id)}/packages">Back to packages</a>
            <a href="/projects/{quote(project_id)}">Resume tables</a>
          </nav>
          {_flash(query)}
          {warnings}
          <article class="document-card">
            <header class="document-header">
              <span class="eyebrow">Package Splitter</span>
              <h1>{_h(package.get('original_filename', package_id))}</h1>
              <p>
                {_h(package.get('page_count', 0))} pages;
                {_h(package.get('index_pages', 1))} index page(s)
              </p>
            </header>
            <form method="post"
                  action="/projects/{quote(project_id)}/packages/{quote(package_id)}/split"
                  class="split-form">
              <div class="redaction-toolbar">
                <div>
                  <strong>Candidate Splits</strong>
                  <span class="muted">
                    Edit names, IDs, and page ranges before creating resumes.
                  </span>
                </div>
                <button type="button" class="secondary-button"
                        data-add-split-row>Add row</button>
              </div>
              <div class="split-table-wrap">
                <table class="split-table">
                  <thead>
                    <tr>
                      <th>Use</th><th>Candidate name</th><th>ID</th>
                      <th>Start</th><th>End</th><th>Index row</th>
                    </tr>
                  </thead>
                  <tbody data-split-rows>{row_html}</tbody>
                </table>
              </div>
              <div class="table-action split-action">
                <button type="submit">Create uploaded resumes</button>
              </div>
            </form>
          </article>
        </div>
        {_package_split_script(package)}
        """
        return _page(f"Split: {package.get('original_filename', package_id)}", body)

    def _render_document_text(
        self,
        project_id: str,
        document_id: str,
        view: str,
    ) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if view == "markdown":
            title = f"Markdown: {document.get('original_filename', document_id)}"
            path_key = "markdown_path"
        elif view == "review":
            title = f"Review: {document.get('original_filename', document_id)}"
            path_key = "review_markdown_path"
        else:
            self._not_found()
            return

        relative = document.get(path_key)
        if not relative:
            self._send_error_page(HTTPStatus.NOT_FOUND, "Document output not found")
            return
        path = self.server.store.resolve_project_path(project_id, relative)
        text = path.read_text(encoding="utf-8")
        rendered = _markdown_to_html(text)
        view_label = "Review report" if view == "review" else "Resume markdown"
        body = f"""
        <div class="document-shell">
          <nav class="top-nav">
            <a href="/projects/{quote(project_id)}">Back to project</a>
          </nav>
          <article class="document-card">
            <header class="document-header">
              <span class="eyebrow">{_h(view_label)}</span>
              <h1>{_h(title)}</h1>
              <p>{_h(document.get('original_filename', ''))}</p>
            </header>
            <div class="markdown-body">{rendered}</div>
          </article>
        </div>
        """
        self._send_html(_page(title, body))

    def _render_pii_review(self, project_id: str, document_id: str) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if document.get("status") == "uploaded":
            self._screen_one_resume(project_id, document_id)
            project = self.server.store.load_project(project_id)
            document = self.server.store.get_document(project, document_id)

        page_paths = document.get("pii_page_image_paths") or document.get(
            "redacted_page_image_paths",
            [],
        )
        detections = document.get("pii_detections") or []
        pages_html = []
        for page_index, path in enumerate(page_paths):
            image_url = _asset_href(project_id, path)
            page_boxes = [
                detection
                for detection in detections
                if int(detection.get("page_index", -1)) == page_index
            ]
            boxes_html = "".join(_redaction_box_html(box) for box in page_boxes)
            pages_html.append(
                f"""
                <section class="redaction-page" data-page-index="{page_index}">
                  <img src="{image_url}" alt="Resume page {page_index + 1}">
                  <div class="redaction-overlay">{boxes_html}</div>
                </section>
                """
            )

        body = f"""
        <div class="document-shell redaction-shell">
          <nav class="top-nav">
            <a href="/projects/{quote(project_id)}">Back to project</a>
          </nav>
          <article class="document-card">
            <header class="document-header">
              <span class="eyebrow">PII Review</span>
              <h1>{_h(document.get('original_filename', document_id))}</h1>
              <p>{len(detections)} local detections</p>
            </header>
            <form method="post"
                  action="/projects/{quote(project_id)}/documents/{quote(document_id)}/redactions"
                  class="redaction-form">
              <input type="hidden" name="redactions_json" value="[]">
              <input type="hidden" name="detections_json"
                     value="{_h(json.dumps(detections))}">
              <input type="hidden" name="selection_dirty" value="false">
              <div class="redaction-toolbar">
                <div>
                  <strong>Redactions</strong>
                  <span class="muted">Detected boxes are selected by default.</span>
                </div>
                <button type="submit">Finalize redacted PDF</button>
              </div>
              <div class="redaction-pages">{''.join(pages_html)}</div>
            </form>
          </article>
        </div>
        {_redaction_script()}
        """
        self._send_html(_page("PII Review", body))

    def _send_pdf(self, project_id: str, document_id: str) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        path = self.server.store.resolve_project_path(project_id, document["pdf_path"])
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Content-Disposition",
            f"inline; filename=\"{document.get('original_filename', 'resume.pdf')}\"",
        )
        self.end_headers()
        self.wfile.write(payload)

    def _send_project_asset(
        self,
        project_id: str,
        query: dict[str, list[str]],
    ) -> None:
        relative = _first(query, "path")
        if not relative:
            self._not_found()
            return
        path = self.server.store.resolve_project_path(project_id, relative)
        if not path.exists() or not path.is_file():
            self._not_found()
            return
        content_type = (
            "image/png"
            if path.suffix.lower() == ".png"
            else "application/octet-stream"
        )
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _create_project(self) -> None:
        form, _files = self._parse_post()
        name = _first(form, "name").strip()
        if not name:
            self._redirect("/?message=Project%20name%20is%20required&level=error")
            return
        project = self.server.store.create_project(name)
        self._redirect(
            f"/projects/{quote(project['id'])}?"
            + urlencode({"message": "Project created"})
        )

    def _reset_local_data(self) -> None:
        self.server.store.reset_all()
        self._redirect("/?message=Local%20project%20data%20deleted")

    def _upload_resumes(self, project_id: str) -> None:
        _form, files = self._parse_post()
        created = self.server.store.add_uploads(project_id, files)
        message = f"Uploaded {len(created)} PDF resume(s)"
        self._redirect_project(project_id, message)

    def _upload_package(self, project_id: str) -> None:
        form, files = self._parse_post()
        if not files:
            self._redirect(
                f"/projects/{quote(project_id)}/packages?"
                + urlencode({"message": "Select a package PDF", "level": "error"})
            )
            return
        index_pages = _parse_positive_int(_first(form, "index_pages"), 2)
        package = self.server.store.create_package_upload(
            project_id,
            files[0],
            index_pages=index_pages,
        )
        try:
            source_pdf = self.server.store.resolve_project_path(
                project_id,
                package["source_pdf_path"],
            )
            analysis = analyze_resume_package(source_pdf, index_pages=index_pages)
            self.server.store.update_package(
                project_id,
                package["id"],
                page_count=analysis["page_count"],
                index_pages=analysis["index_pages"],
                candidate_rows=analysis["candidate_rows"],
                warnings=analysis["warnings"],
                analyzed_at=utc_now(),
                error="",
            )
            message = (
                f"Found {len(analysis['candidate_rows'])} linked resume range(s)"
            )
            self._redirect(
                f"/projects/{quote(project_id)}/packages/{quote(package['id'])}?"
                + urlencode({"message": message})
            )
        except Exception as exc:
            logger.exception("Package analysis failed: %s", package["id"])
            self.server.store.update_package(
                project_id,
                package["id"],
                status="draft",
                error=str(exc),
            )
            self._redirect(
                f"/projects/{quote(project_id)}/packages/{quote(package['id'])}?"
                + urlencode({"message": str(exc), "level": "error"})
            )

    def _save_context(self, project_id: str) -> None:
        form, _files = self._parse_post()
        self.server.store.update_project_context(
            project_id=project_id,
            job_posting=_first(form, "job_posting"),
            work_context=_first(form, "work_context"),
        )
        self._redirect_project(project_id, "Job posting saved")

    def _process_resumes(self, project_id: str) -> None:
        form, _files = self._parse_post()
        document_ids = form.get("document_id", [])
        if not document_ids:
            self._redirect_project(project_id, "Select at least one resume", "error")
            return

        screened_count = 0
        processing_count = 0
        pending_count = 0
        failed_count = 0
        pii_review_ids = []
        processing_ids = []
        for document_id in document_ids:
            try:
                project = self.server.store.load_project(project_id)
                document = self.server.store.get_document(project, document_id)
                status = document.get("status")
                if status == "uploaded":
                    self._screen_one_resume(project_id, document_id)
                    screened_count += 1
                    pii_review_ids.append(document_id)
                elif status == "redacted":
                    self.server.store.update_document(
                        project_id,
                        document_id,
                        status="processing",
                        processing_started_at=utc_now(),
                        processing_error="",
                    )
                    processing_ids.append(document_id)
                    processing_count += 1
                elif status == "pii_review":
                    pending_count += 1
                    pii_review_ids.append(document_id)
                elif status == "processing":
                    pending_count += 1
                else:
                    raise ValueError(
                        "Only uploaded or redacted resumes can be processed"
                    )
            except Exception as exc:
                failed_count += 1
                logger.exception("Resume processing failed: %s", document_id)
                self.server.store.update_document(
                    project_id,
                    document_id,
                    processing_error=str(exc),
                )
        if processing_ids:
            threading.Thread(
                target=self._process_resume_batch,
                args=(project_id, processing_ids),
                daemon=True,
            ).start()

        level = "error" if failed_count and not processing_count else "info"
        if (
            len(pii_review_ids) == 1
            and not processing_count
            and not failed_count
        ):
            self._redirect(
                f"/projects/{quote(project_id)}/documents/"
                f"{quote(pii_review_ids[0])}/pii"
            )
            return
        message = (
            f"PII review {screened_count}; processing {processing_count}; "
            f"pending {pending_count}; failed {failed_count}"
        )
        self._redirect_project(project_id, message, level)

    def _auto_redact_resumes(self, project_id: str) -> None:
        form, _files = self._parse_post()
        document_ids = form.get("document_id", [])
        if not document_ids:
            self._redirect_project(project_id, "Select at least one resume", "error")
            return

        redacted_count = 0
        skipped_count = 0
        failed_count = 0
        for document_id in document_ids:
            try:
                project = self.server.store.load_project(project_id)
                document = self.server.store.get_document(project, document_id)
                status = document.get("status")
                if status == "uploaded":
                    self._screen_one_resume(project_id, document_id)
                    project = self.server.store.load_project(project_id)
                    document = self.server.store.get_document(project, document_id)
                    status = document.get("status")
                if status == "pii_review":
                    self._apply_document_redactions(
                        project_id=project_id,
                        document_id=document_id,
                        redactions=document.get("pii_detections") or [],
                    )
                    redacted_count += 1
                elif status == "redacted":
                    skipped_count += 1
                else:
                    raise ValueError(
                        "Only uploaded or PII-review resumes can auto-redact"
                    )
            except Exception as exc:
                failed_count += 1
                logger.exception("Auto-redaction failed: %s", document_id)
                self.server.store.update_document(
                    project_id,
                    document_id,
                    pii_error=str(exc),
                )

        level = "error" if failed_count and not redacted_count else "info"
        message = (
            f"Auto-redacted {redacted_count}; skipped {skipped_count}; "
            f"failed {failed_count}"
        )
        self._redirect_project(project_id, message, level)

    def _review_resumes(self, project_id: str) -> None:
        form, _files = self._parse_post()
        document_ids = form.get("document_id", [])
        project = self.server.store.load_project(project_id)
        if not project.get("job_posting", "").strip():
            self._redirect_project(
                project_id,
                "Save a job posting before review",
                "error",
            )
            return
        if not document_ids:
            self._redirect_project(
                project_id,
                "Select at least one processed resume",
                "error",
            )
            return

        reviewing_count = 0
        failed_count = 0
        reviewing_ids = []
        for document_id in document_ids:
            try:
                document = self.server.store.get_document(
                    self.server.store.load_project(project_id),
                    document_id,
                )
                if document.get("status") != "processed":
                    raise ValueError("Only processed resumes can be reviewed")
                self.server.store.update_document(
                    project_id,
                    document_id,
                    status="reviewing",
                    review_started_at=utc_now(),
                    review_error="",
                )
                reviewing_ids.append(document_id)
                reviewing_count += 1
            except Exception as exc:
                failed_count += 1
                logger.exception("Resume review failed: %s", document_id)
                self.server.store.update_document(
                    project_id,
                    document_id,
                    review_error=str(exc),
                )
        if reviewing_ids:
            threading.Thread(
                target=self._review_resume_batch,
                args=(project_id, reviewing_ids),
                daemon=True,
            ).start()
        level = "error" if failed_count and not reviewing_count else "info"
        message = f"Reviewing {reviewing_count}; failed {failed_count}"
        self._redirect_project(project_id, message, level)

    def _split_package(self, project_id: str, package_id: str) -> None:
        form, _files = self._parse_post()
        project = self.server.store.load_project(project_id)
        package = self.server.store.get_package(project, package_id)
        split_rows = _split_rows_from_form(form)
        if not split_rows:
            self._redirect(
                f"/projects/{quote(project_id)}/packages/{quote(package_id)}?"
                + urlencode(
                    {
                        "message": "Select at least one split row",
                        "level": "error",
                    }
                )
            )
            return

        source_pdf = self.server.store.resolve_project_path(
            project_id,
            package["source_pdf_path"],
        )
        output_dir = (
            self.server.store.project_dir(project_id)
            / "packages"
            / package_id
            / "split-output"
        )
        try:
            split_pdfs = split_resume_package(
                source_pdf=source_pdf,
                rows=split_rows,
                output_dir=output_dir,
                index_pages=int(package.get("index_pages") or 1),
            )
            uploads = [
                UploadedFile(
                    filename=split_pdf.filename,
                    content=split_pdf.path.read_bytes(),
                )
                for split_pdf in split_pdfs
            ]
            created = self.server.store.add_uploads(project_id, uploads)
            for document, split_pdf in zip(created, split_pdfs, strict=False):
                self.server.store.update_document(
                    project_id,
                    document["id"],
                    source_package_id=package_id,
                    source_package_filename=package.get("original_filename", ""),
                    source_package_range={
                        "start_page": split_pdf.start_page,
                        "end_page": split_pdf.end_page,
                    },
                    candidate_id=split_pdf.candidate_id,
                    candidate_name_hint=split_pdf.candidate_name,
                )
            delete_paths([output_dir])
            self.server.store.update_package(
                project_id,
                package_id,
                status="split",
                split_at=utc_now(),
                candidate_rows=split_rows,
                created_document_ids=[document["id"] for document in created],
                error="",
            )
            message = (
                f"Created {len(created)} uploaded resume(s); "
                "send them through PII review next"
            )
            self._redirect_project(project_id, message)
        except Exception as exc:
            logger.exception("Package split failed: %s", package_id)
            self.server.store.update_package(project_id, package_id, error=str(exc))
            self._redirect(
                f"/projects/{quote(project_id)}/packages/{quote(package_id)}?"
                + urlencode({"message": str(exc), "level": "error"})
            )

    def _process_resume_batch(
        self,
        project_id: str,
        document_ids: list[str],
    ) -> None:
        service = ResumeLLMService(self.server.config, self.server.ssl_setup)
        for document_id in document_ids:
            try:
                self._process_one_resume(project_id, document_id, service)
            except Exception as exc:
                logger.exception("Background resume processing failed: %s", document_id)
                self.server.store.update_document(
                    project_id,
                    document_id,
                    status="redacted",
                    processing_error=str(exc),
                )

    def _review_resume_batch(
        self,
        project_id: str,
        document_ids: list[str],
    ) -> None:
        service = ResumeLLMService(self.server.config, self.server.ssl_setup)
        for document_id in document_ids:
            try:
                self._review_one_resume(project_id, document_id, service)
            except Exception as exc:
                logger.exception("Background resume review failed: %s", document_id)
                self.server.store.update_document(
                    project_id,
                    document_id,
                    status="processed",
                    review_error=str(exc),
                )

    def _screen_one_resume(self, project_id: str, document_id: str) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if document.get("status") not in {"uploaded", "pii_review"}:
            raise ValueError("Only uploaded resumes can enter PII review")

        project_root = self.server.store.project_dir(project_id)
        pdf_path = self.server.store.resolve_project_path(
            project_id,
            document["pdf_path"],
        )
        upload_date = document.get("upload_date") or "unknown-date"
        pages_dir = (
            project_root
            / "documents"
            / upload_date
            / "pii-pages"
            / document_id
        )
        page_paths = render_pdf_pages(pdf_path=pdf_path, output_dir=pages_dir)
        pii_result = detect_pii_boxes(
            pdf_path,
            candidate_names=_candidate_name_hints(document),
        )
        self.server.store.update_document(
            project_id,
            document_id,
            status="pii_review",
            pii_review_at=utc_now(),
            pii_page_image_paths=[
                relative_path(path, project_root)
                for path in page_paths
            ],
            pii_detections=pii_result["detections"],
            pii_error="",
            processing_error="",
        )

    def _finalize_redactions(self, project_id: str, document_id: str) -> None:
        form, _files = self._parse_post()
        raw_redactions = _first(form, "redactions_json")
        raw_detections = _first(form, "detections_json")
        selection_dirty = _first(form, "selection_dirty") == "true"
        try:
            redactions = json.loads(raw_redactions or "[]")
            detections = json.loads(raw_detections or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid redaction payload") from exc
        if not isinstance(redactions, list) or not isinstance(detections, list):
            raise ValueError("Invalid redaction payload")
        if not selection_dirty and len(redactions) < len(detections):
            redactions = detections

        self._apply_document_redactions(
            project_id=project_id,
            document_id=document_id,
            redactions=redactions,
        )
        self._redirect_project(project_id, "Redactions finalized")

    def _apply_document_redactions(
        self,
        project_id: str,
        document_id: str,
        redactions: list[dict[str, Any]],
    ) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if document.get("status") != "pii_review":
            raise ValueError("Resume is not in PII review")

        project_root = self.server.store.project_dir(project_id)
        source_pdf = self.server.store.resolve_project_path(
            project_id,
            document["pdf_path"],
        )
        upload_date = document.get("upload_date") or "unknown-date"
        redacted_pdf = (
            project_root
            / "documents"
            / upload_date
            / "redacted"
            / f"{document_id}.pdf"
        )
        apply_redactions(source_pdf, redacted_pdf, redactions)
        redacted_pages_dir = (
            project_root
            / "documents"
            / upload_date
            / "redacted-pages"
            / document_id
        )
        redacted_pages = render_pdf_pages(
            pdf_path=redacted_pdf,
            output_dir=redacted_pages_dir,
        )

        delete_paths(
            [
                source_pdf,
                project_root
                / "documents"
                / upload_date
                / "pii-pages"
                / document_id,
            ]
        )
        redacted_pdf_rel = relative_path(redacted_pdf, project_root)
        self.server.store.update_document(
            project_id,
            document_id,
            status="redacted",
            redacted_at=utc_now(),
            pdf_path=redacted_pdf_rel,
            redacted_pdf_path=redacted_pdf_rel,
            redacted_page_image_paths=[
                relative_path(path, project_root)
                for path in redacted_pages
            ],
            pii_page_image_paths=[],
            redactions=_sanitize_redactions(redactions),
            original_pdf_deleted=True,
            original_filename=_privacy_filename(document),
            stored_filename=f"{document_id}.pdf",
            candidate_name_hint="",
            pii_error="",
            processing_error="",
        )

    def _process_one_resume(
        self,
        project_id: str,
        document_id: str,
        service: ResumeLLMService,
    ) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if document.get("status") != "processing":
            raise ValueError("Only processing resumes can be sent to the LLM")

        project_root = self.server.store.project_dir(project_id)
        page_paths = [
            self.server.store.resolve_project_path(project_id, relative)
            for relative in document.get("redacted_page_image_paths", [])
        ]
        if not page_paths:
            redacted_pdf_path = self.server.store.resolve_project_path(
                project_id,
                document["redacted_pdf_path"],
            )
            upload_date = document.get("upload_date") or "unknown-date"
            pages_dir = (
                project_root
                / "documents"
                / upload_date
                / "redacted-pages"
                / document_id
            )
            page_paths = render_pdf_pages(
                pdf_path=redacted_pdf_path,
                output_dir=pages_dir,
            )
        upload_date = document.get("upload_date") or "unknown-date"
        result = service.process_resume_pages(
            page_paths=page_paths,
            original_filename=_privacy_filename(document),
        )

        markdown_rel = f"documents/{upload_date}/processed/{document_id}.md"
        self.server.store.write_project_text(project_id, markdown_rel, result.markdown)
        prompt_versions = dict(document.get("prompt_versions") or {})
        prompt_versions.update(result.prompt_versions)
        self.server.store.update_document(
            project_id,
            document_id,
            status="processed",
            processed_at=utc_now(),
            page_image_paths=[
                relative_path(path, project_root)
                for path in page_paths
            ],
            markdown_path=markdown_rel,
            metadata=result.metadata,
            processing_error="",
            prompt_versions=prompt_versions,
        )

    def _review_one_resume(
        self,
        project_id: str,
        document_id: str,
        service: ResumeLLMService,
    ) -> None:
        project = self.server.store.load_project(project_id)
        document = self.server.store.get_document(project, document_id)
        if document.get("status") != "reviewing":
            raise ValueError("Only reviewing resumes can be reviewed")

        markdown_path = self.server.store.resolve_project_path(
            project_id,
            document["markdown_path"],
        )
        resume_markdown = markdown_path.read_text(encoding="utf-8")
        review = service.review_resume(
            resume_markdown=resume_markdown,
            metadata=document.get("metadata", {}),
            job_posting=project.get("job_posting", ""),
            work_context=project.get("work_context", ""),
        )
        upload_date = document.get("upload_date") or "unknown-date"
        review_md_rel = f"documents/{upload_date}/reviewed/{document_id}.md"
        review_json_rel = f"documents/{upload_date}/reviewed/{document_id}.json"
        self.server.store.write_project_text(
            project_id,
            review_md_rel,
            review.report_markdown,
        )
        self.server.store.write_project_json(
            project_id,
            review_json_rel,
            review.review_payload,
        )
        prompt_versions = dict(document.get("prompt_versions") or {})
        prompt_versions.update(review.prompt_versions)
        self.server.store.update_document(
            project_id,
            document_id,
            status="reviewed",
            reviewed_at=utc_now(),
            review_markdown_path=review_md_rel,
            review_json_path=review_json_rel,
            scores=review.scores,
            review_error="",
            prompt_versions=prompt_versions,
        )

    def _parse_post(self) -> tuple[dict[str, list[str]], list[UploadedFile]]:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if content_type.startswith("multipart/form-data"):
            return _parse_multipart(content_type, body)
        fields = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        return fields, []

    def _redirect_project(
        self,
        project_id: str,
        message: str,
        level: str = "info",
    ) -> None:
        query = urlencode({"message": message, "level": level})
        self._redirect(f"/projects/{quote(project_id)}?{query}")

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _send_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _not_found(self) -> None:
        self._send_error_page(HTTPStatus.NOT_FOUND, "Not found")

    def _send_error_page(self, status: HTTPStatus, message: str) -> None:
        body = f"""
        <nav class="top-nav"><a href="/">Projects</a></nav>
        <section class="panel">
          <h1>{status.value} {status.phrase}</h1>
          <p>{_h(message)}</p>
        </section>
        """
        self._send_html(_page(status.phrase, body), status)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    env_path: str = ".env",
    data_dir: str = "data",
    open_browser: bool = True,
    reset_data: bool = False,
) -> None:
    """Start the local hiring assistant web server."""
    config = load_config(env_path)
    setup_logging(config.log_level, output_logs=config.output_logs)
    ssl_setup = setup_ssl(config)
    if not ssl_setup.success:
        raise RuntimeError(ssl_setup.error or "SSL setup failed")

    store = ProjectStore(data_dir)
    if reset_data:
        store.reset_all()
    store.ensure_ready()
    server = HiringAssistantHTTPServer(
        (host, port),
        HiringAssistantHandler,
        store=store,
        config=config,
        ssl_setup=ssl_setup,
    )
    url = f"http://{host}:{port}"
    logger.info("Hiring assistant running at %s", url)
    print(f"Hiring assistant running at {url}")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping hiring assistant")
    finally:
        server.server_close()


def main() -> int:
    """CLI entrypoint for launching the local web app."""
    parser = argparse.ArgumentParser(description="Run the hiring assistant web app.")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--reset-data", action="store_true")
    args = parser.parse_args()
    env_path = Path(args.env)
    if env_path.exists():
        load_env_file(env_path)
    open_browser = _truthy_env("APP_OPEN_BROWSER", True) and not args.no_open_browser
    run_server(
        host=args.host or os.getenv("APP_HOST", "127.0.0.1"),
        port=args.port or int(os.getenv("APP_PORT", "8000")),
        env_path=args.env,
        data_dir=args.data_dir or os.getenv("APP_DATA_DIR", "data"),
        open_browser=open_browser,
        reset_data=args.reset_data
        or _truthy_env("APP_RESET_DATA_ON_STARTUP", False),
    )
    return 0


def _parse_multipart(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, list[str]], list[UploadedFile]]:
    header = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    fields: dict[str, list[str]] = {}
    files: list[UploadedFile] = []
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            if payload:
                files.append(UploadedFile(filename=filename, content=payload))
            continue
        value = payload.decode(part.get_content_charset() or "utf-8")
        fields.setdefault(name, []).append(value)
    return fields, files


def _banner(
    title: str,
    subtitle: str,
    stats: str,
    actions: str = "",
) -> str:
    subtitle_html = f"<p>{_h(subtitle)}</p>" if subtitle else ""
    return f"""
    <header class="title-banner">
      <div class="banner-copy">
        <span class="eyebrow">Hiring Assistant</span>
        <h1>{_h(title)}</h1>
        {subtitle_html}
      </div>
      <div class="banner-side">
        <div class="stat-row">{stats}</div>
        {actions}
      </div>
    </header>
    """


def _reset_form() -> str:
    confirm_text = (
        "Delete all local projects, packages, resumes, and reviews?"
    )
    return f"""
    <form method="post" action="/reset" class="reset-form"
          onsubmit="return confirm('{_h(confirm_text)}');">
      <button type="submit" class="danger-button">Reset local data</button>
    </form>
    """


def _project_sidebar(
    projects: list[dict[str, Any]],
    active_project_id: str,
) -> str:
    items = []
    for project in projects:
        active = " active" if project.get("id") == active_project_id else ""
        href = f"/projects/{quote(project['id'])}"
        count = project.get("document_count", 0)
        updated = _format_ts(project.get("updated_at", ""))
        items.append(
            f"<a class=\"project-link{active}\" href=\"{href}\">"
            f"<span>{_h(project['name'])}</span>"
            f"<small>{count} resumes</small>"
            f"<small>{_h(updated)}</small>"
            "</a>"
        )
    if not items:
        items.append("<p class=\"muted sidebar-empty\">No projects yet.</p>")
    tools = ""
    if active_project_id:
        tools = f"""
        <div class="sidebar-tools">
          <a class="side-tool" href="/projects/{quote(active_project_id)}">
            Resume tables
          </a>
          <a class="side-tool" href="/projects/{quote(active_project_id)}/packages">
            Package splitter
          </a>
        </div>
        """
    return f"""
    <aside class="left-pane">
      <div class="sidebar-section">
        <h2>Projects</h2>
        <form method="post" action="/projects" class="sidebar-create">
          <label>
            New project
            <input name="name" type="text" required placeholder="GG08 posting">
          </label>
          <button type="submit">Create</button>
        </form>
      </div>
      <nav class="project-list" aria-label="Projects">
        {''.join(items)}
      </nav>
      {tools}
    </aside>
    """


def _package_table_row(project_id: str, package: dict[str, Any]) -> str:
    href = f"/projects/{quote(project_id)}/packages/{quote(package['id'])}"
    row_count = len(package.get("candidate_rows") or [])
    error = package.get("error") or ""
    status = package.get("status", "draft")
    status_label = status if not error else f"{status}: {error}"
    return (
        "<tr>"
        f"<td>{_h(package.get('original_filename', 'package.pdf'))}</td>"
        f"<td>{_h(status_label)}</td>"
        f"<td>{_h(package.get('page_count', ''))}</td>"
        f"<td>{_h(package.get('index_pages', ''))}</td>"
        f"<td>{row_count}</td>"
        f"<td>{_h(_format_ts(package.get('created_at', '')))}</td>"
        f"<td><a href=\"{href}\">Open</a></td>"
        "</tr>"
    )


def _fallback_split_row(package: dict[str, Any]) -> dict[str, Any]:
    page_count = max(1, int(package.get("page_count") or 1))
    index_pages = max(1, int(package.get("index_pages") or 1))
    start_page = min(page_count, index_pages + 1)
    return {
        "candidate_name": "Candidate 1",
        "candidate_id": "",
        "start_page": start_page,
        "end_page": page_count,
        "source_text": "Manual range",
        "confidence": "manual",
    }


def _split_row_html(
    row: dict[str, Any],
    index: int,
    package: dict[str, Any],
) -> str:
    page_count = max(1, int(package.get("page_count") or 1))
    min_start = max(1, int(package.get("index_pages") or 1) + 1)
    start_page = int(row.get("start_page") or min_start)
    end_page = int(row.get("end_page") or page_count)
    source = row.get("source_text") or row.get("confidence") or "Manual"
    return f"""
    <tr>
      <td>
        <input type="hidden" name="row_index" value="{index}">
        <input type="checkbox" name="include_row" value="{index}" checked>
      </td>
      <td>
        <input name="candidate_name" type="text"
               value="{_h(row.get('candidate_name', ''))}">
      </td>
      <td>
        <input name="candidate_id" type="text"
               value="{_h(row.get('candidate_id', ''))}">
      </td>
      <td>
        <input name="start_page" type="number" min="{min_start}"
               max="{page_count}" value="{start_page}">
      </td>
      <td>
        <input name="end_page" type="number" min="{min_start}"
               max="{page_count}" value="{end_page}">
      </td>
      <td class="source-cell">{_h(source)}</td>
    </tr>
    """


def _package_split_script(package: dict[str, Any]) -> str:
    page_count = max(1, int(package.get("page_count") or 1))
    min_start = max(1, int(package.get("index_pages") or 1) + 1)
    return f"""
    <script>
      (() => {{
        const tbody = document.querySelector("[data-split-rows]");
        const addButton = document.querySelector("[data-add-split-row]");
        if (!tbody || !addButton) return;
        let nextIndex = tbody.querySelectorAll("tr").length;
        const rowHtml = index => [
          "<tr>",
          "<td>",
          "<input type=\\"hidden\\" name=\\"row_index\\" "
            + "value=\\"" + index + "\\">",
          "<input type=\\"checkbox\\" name=\\"include_row\\" "
            + "value=\\"" + index + "\\" checked>",
          "</td>",
          "<td><input name=\\"candidate_name\\" type=\\"text\\" value=\\"\\"></td>",
          "<td><input name=\\"candidate_id\\" type=\\"text\\" value=\\"\\"></td>",
          "<td><input name=\\"start_page\\" type=\\"number\\" "
            + "min=\\"{min_start}\\" max=\\"{page_count}\\" "
            + "value=\\"{min_start}\\"></td>",
          "<td><input name=\\"end_page\\" type=\\"number\\" "
            + "min=\\"{min_start}\\" max=\\"{page_count}\\" "
            + "value=\\"{page_count}\\"></td>",
          "<td class=\\"source-cell\\">Manual</td>",
          "</tr>"
        ].join("");
        addButton.addEventListener("click", () => {{
          tbody.insertAdjacentHTML("beforeend", rowHtml(nextIndex));
          nextIndex += 1;
        }});
      }})();
    </script>
    """


def _split_rows_from_form(form: dict[str, list[str]]) -> list[dict[str, Any]]:
    row_indices = form.get("row_index", [])
    included = set(form.get("include_row", []))
    rows = []
    for position, row_index in enumerate(row_indices):
        if row_index not in included:
            continue
        start_page = _parse_positive_int(
            _list_at(form.get("start_page", []), position),
            0,
        )
        end_page = _parse_positive_int(
            _list_at(form.get("end_page", []), position),
            0,
        )
        rows.append(
            {
                "candidate_name": _list_at(
                    form.get("candidate_name", []),
                    position,
                ).strip(),
                "candidate_id": _list_at(
                    form.get("candidate_id", []),
                    position,
                ).strip(),
                "start_page": start_page,
                "end_page": end_page,
                "source_text": "Manual edit",
                "confidence": "manual",
            }
        )
    return rows


def _list_at(values: list[str], index: int, default: str = "") -> str:
    return values[index] if index < len(values) else default


def _uploaded_table(project_id: str, documents: list[dict[str, Any]]) -> str:
    rows = []
    for document in documents:
        error = document.get("pii_error") or document.get("processing_error", "")
        status = document.get("status", "uploaded")
        checkbox = (
            _checkbox(document["id"])
            if status in {"uploaded", "pii_review", "redacted"}
            else ""
        )
        pii_href = _document_view_href(project_id, document["id"], "pii")
        if status == "redacted":
            pii_action = "Finalized"
        elif status == "processing":
            pii_action = "Finalized"
        else:
            pii_action = f"<a href=\"{pii_href}\">PII review</a>"
        rows.append(
            "<tr>"
            f"<td>{checkbox}</td>"
            f"<td>{_file_link(project_id, document)}</td>"
            f"<td>{_status_badge(status)}</td>"
            f"<td>{pii_action}</td>"
            f"<td>{_h(_format_ts(document.get('uploaded_at', '')))}</td>"
            f"<td class=\"error-cell\">{_h(error)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(_empty_row(6, "No unprocessed resumes."))
    selectable = [
        doc
        for doc in documents
        if doc.get("status") in {"uploaded", "pii_review", "redacted"}
    ]
    disabled = " disabled" if not selectable else ""
    return f"""
    <form method="post" action="/projects/{quote(project_id)}/process"
          class="selection-form">
      <table>
        <thead>
          <tr>
            <th>{_select_all_checkbox()}</th><th>File</th><th>Status</th><th>PII</th>
            <th>Uploaded</th><th>Error</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div class="table-action">
        {_selection_buttons()}
        <button type="submit"{disabled}
                formaction="/projects/{quote(project_id)}/auto-redact">
          Auto-redact selected
        </button>
        <button type="submit"{disabled}>Continue selected</button>
      </div>
    </form>
    """


def _processed_table(project_id: str, documents: list[dict[str, Any]]) -> str:
    rows = []
    for document in documents:
        metadata = document.get("metadata") or {}
        error = document.get("review_error", "")
        status = document.get("status", "processed")
        checkbox = _checkbox(document["id"]) if status == "processed" else ""
        markdown_href = _document_view_href(project_id, document["id"], "markdown")
        rows.append(
            "<tr>"
            f"<td>{checkbox}</td>"
            f"<td>{_candidate_label(project_id, document)}</td>"
            f"<td>{_status_badge(status)}</td>"
            f"<td>{_h(metadata.get('current_or_recent_title', ''))}</td>"
            f"<td>{_h(metadata.get('current_or_recent_employer', ''))}</td>"
            f"<td>{_h(metadata.get('years_of_experience_estimate', ''))}</td>"
            f"<td><a href=\"{markdown_href}\">Markdown</a></td>"
            f"<td class=\"error-cell\">{_h(error)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(_empty_row(8, "No processed resumes."))
    selectable = [doc for doc in documents if doc.get("status") == "processed"]
    disabled = " disabled" if not selectable else ""
    return f"""
    <form method="post" action="/projects/{quote(project_id)}/review"
          class="selection-form">
      <table>
        <thead>
          <tr>
            <th>{_select_all_checkbox()}</th><th>Candidate</th><th>Status</th>
            <th>Recent role</th><th>Employer</th><th>Experience</th>
            <th>LLM Markdown</th><th>Error</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div class="table-action">
        {_selection_buttons()}
        <button type="submit"{disabled}>Review selected</button>
      </div>
    </form>
    """


def _reviewed_table(project_id: str, documents: list[dict[str, Any]]) -> str:
    rows = []
    for document in documents:
        scores = document.get("scores") or {}
        review_href = _document_view_href(project_id, document["id"], "review")
        markdown_href = _document_view_href(project_id, document["id"], "markdown")
        rows.append(
            "<tr>"
            f"<td>{_candidate_label(project_id, document)}</td>"
            f"<td>{_h(_score(scores.get('aggregate')))}</td>"
            f"<td>{_h(_score(scores.get('holistic')))}</td>"
            f"<td>{_h(scores.get('recommendation', ''))}</td>"
            f"<td><a href=\"{review_href}\">Review</a></td>"
            f"<td><a href=\"{markdown_href}\">Markdown</a></td>"
            "</tr>"
        )
    if not rows:
        rows.append(_empty_row(6, "No reviewed resumes."))
    return f"""
    <table>
      <thead>
        <tr>
          <th>Candidate</th><th>Aggregate</th><th>Holistic</th>
          <th>Recommendation</th><th>Report</th><th>Markdown</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _empty_row(columns: int, message: str) -> str:
    return f"<tr><td colspan=\"{columns}\" class=\"empty-cell\">{_h(message)}</td></tr>"


def _asset_href(project_id: str, relative_path_value: str) -> str:
    return (
        f"/projects/{quote(project_id)}/assets?"
        + urlencode({"path": relative_path_value})
    )


def _redaction_box_html(box: dict[str, Any]) -> str:
    left = float(box.get("x0", 0.0)) * 100
    top = float(box.get("y0", 0.0)) * 100
    width = (float(box.get("x1", 0.0)) - float(box.get("x0", 0.0))) * 100
    height = (float(box.get("y1", 0.0)) - float(box.get("y0", 0.0))) * 100
    label = str(box.get("label") or "PII")
    return (
        "<button type=\"button\" class=\"redaction-box selected\" "
        f"style=\"left:{left:.4f}%;top:{top:.4f}%;"
        f"width:{width:.4f}%;height:{height:.4f}%\" "
        f"data-page-index=\"{int(box.get('page_index', 0))}\" "
        f"data-x0=\"{float(box.get('x0', 0.0)):.6f}\" "
        f"data-y0=\"{float(box.get('y0', 0.0)):.6f}\" "
        f"data-x1=\"{float(box.get('x1', 0.0)):.6f}\" "
        f"data-y1=\"{float(box.get('y1', 0.0)):.6f}\" "
        f"data-label=\"{_h(label)}\"><span>{_h(label)}</span></button>"
    )


def _redaction_script() -> str:
    return """
    <script>
      (() => {
        const form = document.querySelector(".redaction-form");
        if (!form) return;
        const hidden = form.querySelector("input[name='redactions_json']");
        const dirty = form.querySelector("input[name='selection_dirty']");
        const clamp = value => Math.max(0, Math.min(1, value));

        document.addEventListener("click", event => {
          const box = event.target.closest(".redaction-box");
          if (!box) return;
          event.preventDefault();
          dirty.value = "true";
          box.classList.toggle("selected");
        });

        document.querySelectorAll(".redaction-overlay").forEach(overlay => {
          let activeBox = null;
          let startX = 0;
          let startY = 0;

          overlay.addEventListener("pointerdown", event => {
            if (event.target.closest(".redaction-box")) return;
            const rect = overlay.getBoundingClientRect();
            startX = clamp((event.clientX - rect.left) / rect.width);
            startY = clamp((event.clientY - rect.top) / rect.height);
            activeBox = document.createElement("button");
            activeBox.type = "button";
            activeBox.className = "redaction-box manual selected";
            activeBox.dataset.pageIndex = overlay.parentElement.dataset.pageIndex;
            activeBox.dataset.label = "Manual";
            activeBox.innerHTML = "<span>Manual</span>";
            overlay.appendChild(activeBox);
            dirty.value = "true";
            overlay.setPointerCapture(event.pointerId);
          });

          overlay.addEventListener("pointermove", event => {
            if (!activeBox) return;
            const rect = overlay.getBoundingClientRect();
            const currentX = clamp((event.clientX - rect.left) / rect.width);
            const currentY = clamp((event.clientY - rect.top) / rect.height);
            const x0 = Math.min(startX, currentX);
            const y0 = Math.min(startY, currentY);
            const x1 = Math.max(startX, currentX);
            const y1 = Math.max(startY, currentY);
            Object.assign(activeBox.dataset, { x0, y0, x1, y1 });
            activeBox.style.left = `${x0 * 100}%`;
            activeBox.style.top = `${y0 * 100}%`;
            activeBox.style.width = `${(x1 - x0) * 100}%`;
            activeBox.style.height = `${(y1 - y0) * 100}%`;
          });

          overlay.addEventListener("pointerup", event => {
            if (!activeBox) return;
            const width = Number(activeBox.dataset.x1) - Number(activeBox.dataset.x0);
            const height = Number(activeBox.dataset.y1) - Number(activeBox.dataset.y0);
            if (width < 0.004 || height < 0.004) {
              activeBox.remove();
            }
            activeBox = null;
            overlay.releasePointerCapture(event.pointerId);
          });
        });

        form.addEventListener("submit", () => {
          const boxes = [...document.querySelectorAll(".redaction-box.selected")]
            .map(box => ({
              page_index: Number(box.dataset.pageIndex),
              x0: Number(box.dataset.x0),
              y0: Number(box.dataset.y0),
              x1: Number(box.dataset.x1),
              y1: Number(box.dataset.y1),
              label: box.dataset.label || "PII",
              source: box.classList.contains("manual") ? "manual" : "detected"
            }))
            .filter(box => Number.isFinite(box.x0) && Number.isFinite(box.y0));
          hidden.value = JSON.stringify(boxes);
        });
      })();
    </script>
    """


def _auto_refresh_script(active_count: int) -> str:
    if active_count <= 0:
        return ""
    return """
    <script>
      window.setTimeout(() => {
        window.location.reload();
      }, 5000);
    </script>
    """


def _interaction_script() -> str:
    return """
    <script>
      (() => {
        document.addEventListener("click", event => {
          const button = event.target.closest("[data-select-action]");
          if (!button) return;
          const form = button.closest("form");
          if (!form) return;
          const checked = button.dataset.selectAction === "all";
          form.querySelectorAll("input[name='document_id']").forEach(input => {
            input.checked = checked;
          });
          const master = form.querySelector("[data-select-master]");
          if (master) master.checked = checked;
        });

        document.addEventListener("change", event => {
          const master = event.target.closest("[data-select-master]");
          if (!master) return;
          const form = master.closest("form");
          if (!form) return;
          form.querySelectorAll("input[name='document_id']").forEach(input => {
            input.checked = master.checked;
          });
        });

        document.addEventListener("submit", event => {
          const form = event.target.closest("form.selection-form");
          if (!form) return;
          const submit = event.submitter || form.querySelector("button[type='submit']");
          if (submit) {
            submit.disabled = true;
            submit.textContent = "Working...";
          }
        });
      })();
    </script>
    """


def _sanitize_redactions(redactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for redaction in redactions:
        try:
            sanitized.append(
                {
                    "page_index": int(redaction.get("page_index", 0)),
                    "x0": float(redaction.get("x0", 0.0)),
                    "y0": float(redaction.get("y0", 0.0)),
                    "x1": float(redaction.get("x1", 0.0)),
                    "y1": float(redaction.get("y1", 0.0)),
                    "label": str(redaction.get("label") or "PII"),
                    "source": str(redaction.get("source") or "detected"),
                }
            )
        except (TypeError, ValueError):
            continue
    return sanitized


def _candidate_label(project_id: str, document: dict[str, Any]) -> str:
    return (
        f"<a href=\"/projects/{quote(project_id)}/files/{quote(document['id'])}\">"
        f"{_h(_document_label(document))}</a>"
    )


def _file_link(project_id: str, document: dict[str, Any]) -> str:
    return (
        f"<a href=\"/projects/{quote(project_id)}/files/{quote(document['id'])}\">"
        f"{_h(_document_label(document))}</a>"
    )


def _checkbox(document_id: str) -> str:
    return (
        "<input type=\"checkbox\" name=\"document_id\" "
        f"value=\"{_h(document_id)}\">"
    )


def _select_all_checkbox() -> str:
    return "<input type=\"checkbox\" data-select-master aria-label=\"Select all\">"


def _selection_buttons() -> str:
    return """
    <button type="button" class="secondary-button compact-button"
            data-select-action="all">Select all</button>
    <button type="button" class="secondary-button compact-button"
            data-select-action="none">Deselect all</button>
    """


def _status_badge(status: str) -> str:
    class_name = "status-pill"
    if status in {"processing", "reviewing"}:
        class_name += " status-loading"
    return f"<span class=\"{class_name}\">{_h(_status_label(status))}</span>"


def _document_label(document: dict[str, Any]) -> str:
    candidate_id = str(document.get("candidate_id") or "").strip()
    if candidate_id:
        return f"Candidate {candidate_id}"
    return f"Document {str(document.get('id', ''))[:8]}"


def _privacy_filename(document: dict[str, Any]) -> str:
    return f"{_document_label(document).lower().replace(' ', '-')}.pdf"


def _candidate_name_hints(document: dict[str, Any]) -> list[str]:
    hints = []
    for key in ("candidate_name_hint",):
        value = str(document.get(key) or "").strip()
        if value:
            hints.append(value)
    return hints


def _document_view_href(project_id: str, document_id: str, view: str) -> str:
    return (
        f"/projects/{quote(project_id)}/documents/"
        f"{quote(document_id)}/{quote(view)}"
    )


def _status_label(status: str) -> str:
    labels = {
        "uploaded": "Needs PII review",
        "pii_review": "PII review",
        "redacted": "Redacted",
        "processing": "Processing markdown",
        "processed": "Processed",
        "reviewing": "Reviewing",
        "reviewed": "Reviewed",
    }
    return labels.get(status, status)


def _external_link(value: str) -> str:
    if not value:
        return ""
    href = value if value.startswith(("http://", "https://")) else f"https://{value}"
    return f"<a href=\"{_h(href)}\">{_h(value)}</a>"


def _markdown_to_html(markdown: str) -> str:
    """Render a conservative markdown subset as escaped HTML."""
    parts: list[str] = []
    paragraph: list[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        if not paragraph:
            return
        content = "<br>".join(_inline_markdown(line) for line in paragraph)
        parts.append(f"<p>{content}</p>")
        paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            parts.append(f"</{list_tag}>")
            list_tag = ""

    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = min(len(heading.group(1)), 4)
            parts.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            target_tag = "ul" if unordered else "ol"
            if list_tag != target_tag:
                close_list()
                parts.append(f"<{target_tag}>")
                list_tag = target_tag
            item = unordered.group(1) if unordered else ordered.group(1)
            parts.append(f"<li>{_inline_markdown(item)}</li>")
            continue

        close_list()
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(parts)


def _inline_markdown(text: str) -> str:
    """Render inline markdown after escaping raw text."""
    escaped = html.escape(text, quote=True)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return _auto_link(escaped)


def _auto_link(escaped: str) -> str:
    """Link safe http(s) URLs in already-escaped text."""

    def replace(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = ""
        while url and url[-1] in ".,);:":
            trailing = url[-1] + trailing
            url = url[:-1]
        return f"<a href=\"{url}\">{url}</a>{trailing}"

    return re.sub(r"https?://[^\s<]+", replace, escaped)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f5f7;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --line: #d9dee7;
      --text: #1d2530;
      --muted: #667085;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --banner: #17212b;
      --banner-muted: #b7c3cf;
      --error: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      width: min(1680px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 16px 0 36px;
    }}
    h1, h2, h3 {{ margin: 0 0 14px; line-height: 1.2; letter-spacing: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; }}
    h3 {{ font-size: 16px; margin-top: 18px; }}
    p {{ margin: 0 0 12px; }}
    a {{ color: var(--accent-dark); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .app-shell {{
      display: grid;
      gap: 16px;
    }}
    .title-banner {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      min-height: 118px;
      padding: 22px 24px;
      color: #ffffff;
      background: var(--banner);
      border-radius: 8px;
      border: 1px solid #223244;
    }}
    .banner-copy {{
      min-width: 240px;
    }}
    .banner-copy h1 {{
      margin-bottom: 8px;
    }}
    .banner-copy p {{
      color: var(--banner-muted);
      margin: 0;
    }}
    .eyebrow {{
      display: block;
      margin-bottom: 8px;
      color: #70d4c7;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .banner-side {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 18px;
      flex-wrap: wrap;
    }}
    .banner-upload {{
      display: flex;
      align-items: flex-end;
      gap: 10px;
      padding: 10px;
      background: rgb(255 255 255 / 8%);
      border: 1px solid rgb(255 255 255 / 15%);
      border-radius: 8px;
    }}
    .banner-upload label {{
      width: min(280px, 34vw);
      margin: 0;
      color: var(--banner-muted);
    }}
    .banner-upload input[type="file"] {{
      margin-top: 5px;
      background: #ffffff;
      color: var(--text);
    }}
    .banner-link {{
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      padding: 10px 13px;
      color: #ffffff;
      background: rgb(255 255 255 / 8%);
      border: 1px solid rgb(255 255 255 / 18%);
      border-radius: 6px;
      font-weight: 800;
    }}
    .banner-link:hover {{
      color: #ffffff;
      border-color: rgb(255 255 255 / 36%);
      text-decoration: none;
    }}
    .app-layout {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: start;
    }}
    .left-pane,
    .right-pane,
    .table-panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .left-pane,
    .right-pane {{
      padding: 16px;
      position: sticky;
      top: 16px;
      max-height: calc(100vh - 32px);
      overflow: auto;
    }}
    .muted {{ color: var(--muted); }}
    .sidebar-section h2,
    .right-pane h2 {{
      margin-bottom: 12px;
    }}
    .sidebar-create {{
      display: grid;
      gap: 8px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .project-list {{
      display: grid;
      gap: 8px;
      padding-top: 14px;
    }}
    .sidebar-tools {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    .side-tool {{
      display: flex;
      align-items: center;
      min-height: 38px;
      padding: 9px 10px;
      color: var(--accent-dark);
      background: #ecfdf9;
      border: 1px solid #a7e3d8;
      border-radius: 6px;
      font-weight: 800;
    }}
    .side-tool:hover {{
      text-decoration: none;
      border-color: var(--accent);
    }}
    .project-link {{
      display: grid;
      gap: 3px;
      padding: 10px;
      color: var(--text);
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .project-link:hover {{
      text-decoration: none;
      border-color: #9bc6bf;
    }}
    .project-link.active {{
      border-color: var(--accent);
      background: #ecfdf9;
    }}
    .project-link span {{
      font-weight: 700;
    }}
    .project-link small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .sidebar-empty {{
      margin: 0;
      padding: 10px 0;
    }}
    .main-pane {{
      min-width: 0;
    }}
    .table-stack {{
      display: grid;
      gap: 16px;
    }}
    .table-panel {{
      padding: 14px;
      overflow: hidden;
    }}
    .table-panel-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .table-panel-header h2 {{
      margin-bottom: 4px;
    }}
    .table-panel-header p {{
      margin: 0;
      font-size: 13px;
    }}
    .count-pill {{
      min-width: 34px;
      height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: var(--accent-dark);
      background: #ecfdf9;
      border: 1px solid #a7e3d8;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 800;
    }}
    .stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .stat {{
      min-width: 96px;
      border: 1px solid rgb(255 255 255 / 18%);
      border-radius: 8px;
      padding: 10px 12px;
      color: #ffffff;
      background: rgb(255 255 255 / 8%);
    }}
    .stat strong {{ display: block; font-size: 22px; }}
    .stat span {{
      color: var(--banner-muted);
      font-size: 12px;
    }}
    form {{ margin: 0; }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    input[type="text"], input[type="file"], input[type="number"], textarea {{
      display: block;
      width: 100%;
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
    }}
    textarea {{ resize: vertical; }}
    button {{
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.45;
    }}
    .secondary-button {{
      color: var(--accent-dark);
      background: #ecfdf9;
      border: 1px solid #a7e3d8;
    }}
    .secondary-button:hover {{
      background: #d9f7f0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin: 0;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #eef2f6;
      color: #344054;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .table-action {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
    }}
    .compact-button {{
      padding: 8px 10px;
      font-size: 12px;
    }}
    .empty-cell {{
      color: var(--muted);
      background: var(--surface-soft);
      text-align: center;
      padding: 20px 12px;
    }}
    .context-form {{
      display: grid;
      gap: 10px;
    }}
    .package-upload-form {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 140px auto;
      gap: 12px;
      align-items: end;
    }}
    .package-upload-form label {{
      margin-bottom: 0;
    }}
    .side-note {{
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    .side-note strong {{
      display: block;
      margin-bottom: 6px;
      color: var(--text);
    }}
    .side-note p {{
      margin: 0;
    }}
    .reset-form {{
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    .danger-button {{
      background: #b42318;
    }}
    .danger-button:hover {{
      background: #8f1d14;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 26px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface-soft);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .status-loading {{
      color: var(--accent-dark);
      border-color: #a7e3d8;
      background: #ecfdf9;
    }}
    .status-loading::before {{
      content: "";
      width: 9px;
      height: 9px;
      border: 2px solid #a7e3d8;
      border-top-color: var(--accent-dark);
      border-radius: 999px;
      animation: spin 0.9s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .document-shell {{
      width: min(1040px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }}
    .top-nav a {{
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      padding: 8px 12px;
      color: var(--accent-dark);
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      font-weight: 700;
    }}
    .top-nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .top-nav a:hover {{
      text-decoration: none;
      border-color: #9bc6bf;
    }}
    .document-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .document-header {{
      padding: 24px 28px;
      color: #ffffff;
      background: var(--banner);
    }}
    .document-header h1 {{
      margin-bottom: 8px;
    }}
    .document-header p {{
      margin: 0;
      color: var(--banner-muted);
    }}
    .markdown-body {{
      padding: 26px 30px 34px;
      line-height: 1.55;
      font-size: 15px;
    }}
    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3,
    .markdown-body h4 {{
      color: var(--text);
      margin: 24px 0 10px;
    }}
    .markdown-body h1:first-child,
    .markdown-body h2:first-child {{
      margin-top: 0;
    }}
    .markdown-body h1 {{
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
      font-size: 26px;
    }}
    .markdown-body h2 {{
      padding-bottom: 7px;
      border-bottom: 1px solid var(--line);
      font-size: 20px;
    }}
    .markdown-body h3 {{
      font-size: 17px;
    }}
    .markdown-body p {{
      margin: 0 0 12px;
    }}
    .markdown-body ul,
    .markdown-body ol {{
      margin: 0 0 14px 22px;
      padding: 0;
    }}
    .markdown-body li {{
      margin: 6px 0;
      padding-left: 2px;
    }}
    .markdown-body strong {{
      font-weight: 800;
    }}
    .markdown-body code {{
      padding: 2px 5px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }}
    .redaction-form {{
      background: var(--surface-soft);
    }}
    .redaction-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }}
    .redaction-toolbar strong {{
      display: block;
      margin-bottom: 3px;
    }}
    .redaction-pages {{
      display: grid;
      gap: 18px;
      padding: 18px;
    }}
    .redaction-page {{
      position: relative;
      width: min(100%, 920px);
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      user-select: none;
    }}
    .redaction-page img {{
      display: block;
      width: 100%;
      height: auto;
      pointer-events: none;
    }}
    .redaction-overlay {{
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }}
    .redaction-box {{
      position: absolute;
      z-index: 2;
      min-width: 8px;
      min-height: 8px;
      padding: 0;
      color: #ffffff;
      background: #000000;
      border: 2px solid #000000;
      border-radius: 2px;
      cursor: pointer;
    }}
    .redaction-box span {{
      position: absolute;
      left: -2px;
      top: -24px;
      display: none;
      padding: 2px 5px;
      color: #ffffff;
      background: #b42318;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .redaction-box:hover span {{
      display: block;
    }}
    .redaction-box:not(.selected) {{
      background: transparent;
      border-color: #98a2b3;
      border-style: dashed;
    }}
    .redaction-box.manual {{
      background: #000000;
      border-color: #000000;
    }}
    .redaction-box.manual span {{
      background: var(--accent);
    }}
    .redaction-box.manual:not(.selected) {{
      background: transparent;
      border-color: var(--accent);
    }}
    .package-shell {{
      width: min(1260px, 100%);
    }}
    .split-table-wrap {{
      padding: 18px;
      overflow-x: auto;
      background: var(--surface-soft);
    }}
    .split-table input[type="text"],
    .split-table input[type="number"] {{
      min-width: 120px;
      margin: 0;
      padding: 8px 9px;
    }}
    .split-table input[name="candidate_name"] {{
      min-width: 220px;
    }}
    .source-cell {{
      min-width: 220px;
      max-width: 420px;
      color: var(--muted);
      font-size: 13px;
    }}
    .split-action {{
      padding: 0 18px 18px;
      margin-top: 0;
      background: var(--surface-soft);
    }}
    .flash {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      background: var(--surface);
      padding: 12px 14px;
      margin-bottom: 14px;
    }}
    .flash.error {{
      border-left-color: var(--error);
      color: var(--error);
    }}
    .error-cell {{ color: var(--error); max-width: 320px; }}
    @media (max-width: 900px) {{
      main {{ width: min(100vw - 24px, 1680px); padding-top: 12px; }}
      .title-banner,
      .banner-side,
      .banner-upload {{
        display: grid;
        justify-content: stretch;
      }}
      .banner-upload label {{
        width: 100%;
      }}
      .app-layout {{ grid-template-columns: 1fr; }}
      .left-pane,
      .right-pane {{
        position: static;
        max-height: none;
      }}
      .package-upload-form {{
        grid-template-columns: 1fr;
      }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>{body}</main>
  {_interaction_script()}
</body>
</html>
"""


def _flash(query: dict[str, list[str]]) -> str:
    message = _first(query, "message")
    if not message:
        return ""
    level = _first(query, "level") or "info"
    class_name = "flash error" if level == "error" else "flash"
    return f"<div class=\"{class_name}\">{_h(message)}</div>"


def _stat(label: str, value: int) -> str:
    return f"<div class=\"stat\"><strong>{value}</strong><span>{_h(label)}</span></div>"


def _path_segments(path: str) -> list[str]:
    return [unquote(segment) for segment in path.strip("/").split("/") if segment]


def _first(values: dict[str, list[str]], key: str) -> str:
    raw = values.get(key, [""])
    return raw[0] if raw else ""


def _format_ts(value: str) -> str:
    if not value:
        return ""
    return value.replace("T", " ").replace("+00:00", " UTC")


def _score(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())

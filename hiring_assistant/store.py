"""File-backed project and resume storage for the hiring assistant."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DOCUMENT_STATUSES = {
    "uploaded",
    "pii_review",
    "redacted",
    "processing",
    "processed",
    "reviewing",
    "reviewed",
}


def utc_now() -> str:
    """Return an ISO timestamp with UTC timezone."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_date_slug() -> str:
    """Return the local upload date slug used for document folders."""
    return datetime.now().strftime("%Y-%m-%d")


def slugify(value: str, fallback: str = "project") -> str:
    """Return a URL and filesystem-safe slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned or fallback


def safe_filename(filename: str) -> str:
    """Return a conservative filename while preserving a useful stem."""
    path = Path(filename)
    stem = slugify(path.stem, fallback="resume")
    suffix = path.suffix.lower() if path.suffix else ".pdf"
    if suffix != ".pdf":
        suffix = ".pdf"
    return f"{stem}{suffix}"


def relative_path(path: Path, root: Path) -> str:
    """Return a POSIX relative path for JSON storage."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().relative_to(root.resolve()).as_posix()


@dataclass(frozen=True)
class UploadedFile:
    """One uploaded PDF payload."""

    filename: str
    content: bytes


class ProjectStore:
    """Manage projects and their local document manifests."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.index_path = self.projects_dir / "index.json"
        self.lock = threading.RLock()

    def ensure_ready(self) -> None:
        """Create the local data directories and index file."""
        with self.lock:
            self.projects_dir.mkdir(parents=True, exist_ok=True)
            if not self.index_path.exists():
                self._write_json(self.index_path, {"projects": []})

    def reset_all(self) -> None:
        """Delete all local project data and recreate an empty store."""
        with self.lock:
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            self.ensure_ready()

    def list_projects(self) -> list[dict[str, Any]]:
        """Return projects sorted newest first."""
        with self.lock:
            self.ensure_ready()
            index = self._read_json(self.index_path, {"projects": []})
            return sorted(
                index.get("projects", []),
                key=lambda item: item.get("updated_at", item.get("created_at", "")),
                reverse=True,
            )

    def create_project(self, name: str) -> dict[str, Any]:
        """Create a new project and return its manifest."""
        with self.lock:
            self.ensure_ready()
            now = utc_now()
            project_id = self._new_project_id(name)
            project_dir = self.project_dir(project_id)
            project_dir.mkdir(parents=True, exist_ok=False)
            for child in ("documents", "packages", "reviews"):
                (project_dir / child).mkdir(parents=True, exist_ok=True)

            project = {
                "id": project_id,
                "name": name.strip() or "Untitled project",
                "job_posting": "",
                "work_context": "",
                "created_at": now,
                "updated_at": now,
                "documents": [],
                "packages": [],
            }
            self.save_project(project)
            self._upsert_index_project(project)
            return project

    def load_project(self, project_id: str) -> dict[str, Any]:
        """Load one project manifest."""
        with self.lock:
            self.ensure_ready()
            project_path = self.project_path(project_id)
            if not project_path.exists():
                raise KeyError(f"Project not found: {project_id}")
            return self._read_json(project_path, {})

    def save_project(self, project: dict[str, Any]) -> None:
        """Persist one project manifest and update the index."""
        with self.lock:
            project["updated_at"] = utc_now()
            project_path = self.project_path(project["id"])
            project_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(project_path, project)
            self._upsert_index_project(project)

    def update_project_context(
        self,
        project_id: str,
        job_posting: str,
        work_context: str,
    ) -> dict[str, Any]:
        """Save the job posting and optional work context."""
        with self.lock:
            project = self.load_project(project_id)
            project["job_posting"] = job_posting.strip()
            project["work_context"] = work_context.strip()
            self.save_project(project)
            return project

    def add_uploads(
        self,
        project_id: str,
        uploaded_files: list[UploadedFile],
    ) -> list[dict[str, Any]]:
        """Store uploaded PDFs and append document records to the project."""
        with self.lock:
            project = self.load_project(project_id)
            upload_date = current_date_slug()
            uploads_dir = (
                self.project_dir(project_id)
                / "documents"
                / upload_date
                / "uploaded"
            )
            uploads_dir.mkdir(parents=True, exist_ok=True)

            created: list[dict[str, Any]] = []
            for uploaded_file in uploaded_files:
                if not uploaded_file.filename.lower().endswith(".pdf"):
                    continue
                document_id = uuid.uuid4().hex
                digest = hashlib.sha256(uploaded_file.content).hexdigest()
                stored_name = f"{document_id}-{safe_filename(uploaded_file.filename)}"
                target_path = uploads_dir / stored_name
                target_path.write_bytes(uploaded_file.content)
                duplicate_of = self._find_duplicate(project, digest)
                document = {
                    "id": document_id,
                    "status": "uploaded",
                    "original_filename": uploaded_file.filename,
                    "stored_filename": stored_name,
                    "file_sha256": digest,
                    "duplicate_of": duplicate_of,
                    "upload_date": upload_date,
                    "uploaded_at": utc_now(),
                    "processed_at": "",
                    "pii_review_at": "",
                    "redacted_at": "",
                    "reviewed_at": "",
                    "pdf_path": relative_path(
                        target_path,
                        self.project_dir(project_id),
                    ),
                    "redacted_pdf_path": "",
                    "pii_page_image_paths": [],
                    "redacted_page_image_paths": [],
                    "page_image_paths": [],
                    "markdown_path": "",
                    "review_markdown_path": "",
                    "review_json_path": "",
                    "metadata": {},
                    "scores": {},
                    "pii_detections": [],
                    "redactions": [],
                    "original_pdf_deleted": False,
                    "processing_error": "",
                    "pii_error": "",
                    "review_error": "",
                    "prompt_versions": {},
                }
                project["documents"].append(document)
                created.append(document)

            if created:
                self.save_project(project)
            return created

    def create_package_upload(
        self,
        project_id: str,
        uploaded_file: UploadedFile,
        index_pages: int,
    ) -> dict[str, Any]:
        """Store one combined resume package PDF for later splitting."""
        with self.lock:
            project = self.load_project(project_id)
            if not uploaded_file.filename.lower().endswith(".pdf"):
                raise ValueError("Package upload must be a PDF")
            package_id = uuid.uuid4().hex
            package_dir = self.project_dir(project_id) / "packages" / package_id
            package_dir.mkdir(parents=True, exist_ok=False)
            stored_name = f"source-{safe_filename(uploaded_file.filename)}"
            target_path = package_dir / stored_name
            target_path.write_bytes(uploaded_file.content)
            now = utc_now()
            package = {
                "id": package_id,
                "status": "draft",
                "original_filename": uploaded_file.filename,
                "stored_filename": stored_name,
                "source_pdf_path": relative_path(
                    target_path,
                    self.project_dir(project_id),
                ),
                "index_pages": max(1, int(index_pages or 1)),
                "page_count": 0,
                "candidate_rows": [],
                "warnings": [],
                "created_document_ids": [],
                "created_at": now,
                "analyzed_at": "",
                "split_at": "",
                "error": "",
            }
            project.setdefault("packages", []).append(package)
            self.save_project(project)
            return package

    def update_package(
        self,
        project_id: str,
        package_id: str,
        **updates: Any,
    ) -> dict[str, Any]:
        """Patch one package record and save the project."""
        with self.lock:
            project = self.load_project(project_id)
            package = self.get_package(project, package_id)
            package.update(updates)
            self.save_project(project)
            return package

    def get_package(
        self,
        project: dict[str, Any],
        package_id: str,
    ) -> dict[str, Any]:
        """Return a package from a loaded project manifest."""
        for package in project.get("packages", []):
            if package.get("id") == package_id:
                return package
        raise KeyError(f"Package not found: {package_id}")

    def update_document(
        self,
        project_id: str,
        document_id: str,
        **updates: Any,
    ) -> dict[str, Any]:
        """Patch one document record and save the project."""
        with self.lock:
            project = self.load_project(project_id)
            document = self.get_document(project, document_id)
            if "status" in updates and updates["status"] not in DOCUMENT_STATUSES:
                raise ValueError(f"Invalid status: {updates['status']}")
            document.update(updates)
            self.save_project(project)
            return document

    def get_document(
        self,
        project: dict[str, Any],
        document_id: str,
    ) -> dict[str, Any]:
        """Return a document from a loaded project manifest."""
        for document in project.get("documents", []):
            if document.get("id") == document_id:
                return document
        raise KeyError(f"Document not found: {document_id}")

    def resolve_project_path(self, project_id: str, relative: str) -> Path:
        """Resolve a stored relative project path."""
        path = (self.project_dir(project_id) / relative).resolve()
        root = self.project_dir(project_id).resolve()
        if root not in path.parents and path != root:
            raise ValueError("Path escapes project directory")
        return path

    def write_project_text(self, project_id: str, relative: str, text: str) -> Path:
        """Write UTF-8 text under a project directory."""
        with self.lock:
            path = self.resolve_project_path(project_id, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return path

    def write_project_json(
        self,
        project_id: str,
        relative: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write JSON under a project directory."""
        with self.lock:
            path = self.resolve_project_path(project_id, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(path, payload)
            return path

    def project_dir(self, project_id: str) -> Path:
        """Return the project directory."""
        return self.projects_dir / project_id

    def project_path(self, project_id: str) -> Path:
        """Return the project manifest path."""
        return self.project_dir(project_id) / "project.json"

    def _new_project_id(self, name: str) -> str:
        base = slugify(name, fallback="project")
        candidate = base
        counter = 2
        while self.project_path(candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _find_duplicate(self, project: dict[str, Any], digest: str) -> str:
        for document in project.get("documents", []):
            if document.get("file_sha256") == digest:
                return str(document.get("id", ""))
        return ""

    def _upsert_index_project(self, project: dict[str, Any]) -> None:
        self.ensure_ready()
        index = self._read_json(self.index_path, {"projects": []})
        summary = {
            "id": project["id"],
            "name": project["name"],
            "created_at": project.get("created_at", ""),
            "updated_at": project.get("updated_at", ""),
            "document_count": len(project.get("documents", [])),
        }
        projects = [
            item
            for item in index.get("projects", [])
            if item.get("id") != project["id"]
        ]
        projects.append(summary)
        index["projects"] = projects
        self._write_json(self.index_path, index)

    def _read_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default.copy()
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        shutil.move(temp_path, path)

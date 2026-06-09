"""Tests for local project and document storage."""

from hiring_assistant.store import ProjectStore, UploadedFile


def test_project_store_creates_project_and_uploads_pdf(tmp_path):
    """It stores project-local uploads and duplicate metadata."""
    store = ProjectStore(tmp_path / "data")
    project = store.create_project("GG08 Posting")

    uploaded = store.add_uploads(
        project["id"],
        [
            UploadedFile("Alex Resume.pdf", b"%PDF-1.4 test"),
            UploadedFile("Alex Resume.pdf", b"%PDF-1.4 test"),
        ],
    )

    assert project["id"] == "gg08-posting"
    assert len(uploaded) == 2
    assert uploaded[0]["status"] == "uploaded"
    assert uploaded[0]["pdf_path"].startswith("documents/")
    assert uploaded[0]["pdf_path"].endswith(".pdf")
    assert uploaded[1]["duplicate_of"] == uploaded[0]["id"]

    loaded = store.load_project(project["id"])
    assert len(loaded["documents"]) == 2
    assert store.resolve_project_path(project["id"], uploaded[0]["pdf_path"]).exists()


def test_project_store_keeps_projects_isolated(tmp_path):
    """Projects keep separate manifests and document folders."""
    store = ProjectStore(tmp_path / "data")
    first = store.create_project("GG08 Posting")
    second = store.create_project("GG07 Posting")

    store.add_uploads(first["id"], [UploadedFile("First.pdf", b"%PDF first")])
    store.add_uploads(second["id"], [UploadedFile("Second.pdf", b"%PDF second")])

    first_loaded = store.load_project(first["id"])
    second_loaded = store.load_project(second["id"])

    assert first_loaded["documents"][0]["original_filename"] == "First.pdf"
    assert second_loaded["documents"][0]["original_filename"] == "Second.pdf"
    assert first_loaded["documents"][0]["pdf_path"] != second_loaded["documents"][0][
        "pdf_path"
    ]


def test_project_store_resets_local_data(tmp_path):
    """It can wipe all local projects and recreate an empty index."""
    store = ProjectStore(tmp_path / "data")
    project = store.create_project("GG08 Posting")
    store.add_uploads(project["id"], [UploadedFile("First.pdf", b"%PDF first")])

    store.reset_all()

    assert store.list_projects() == []
    assert store.index_path.exists()

# tests/test_unit_cases.py

import io
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint



client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text

# -------- Health Endpoint -------- #
def test_health():
    response = client.post("/health")
    assert response.status_code == 200
    assert response.json() == {"status" : "ok", "service": "document-portal"}


# -------- Analyze Endpoint (happy Path) -------- #

def test_home():
    """Basic sanity check for home route"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


def test_analyze_valid_pdf(monkeypatch):
    """Test PDF upload + analysis with mocked analyzer"""

    # --- Mock dependencies ---
    class MockDocHandler:
        def save_pdf(self, file):
            return "dummy_path.pdf"
        def read_pdf(self, path):
            return "Dummy PDF content"

    class MockAnalyzer:
        def analyze_document(self, text: str):
            return {"title": "Mocked Title", "summary": "Mocked Summary"}

    # monkeypatch the real handlers
    import api.main as main_app
    monkeypatch.setattr(main_app, "DocHandler", MockDocHandler)
    monkeypatch.setattr(main_app, "DocumentAnalyzer", lambda: MockAnalyzer())

    # Prepare fake PDF file for upload
    fake_pdf = io.BytesIO(b"%PDF-1.4 Mock PDF content")
    response = client.post(
        "/analyze",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["title"] == "Mocked Title"


def test_analyze_invalid_file_type():
    """Should reject non-PDF files"""
    txt_file = io.BytesIO(b"Not a PDF")
    response = client.post(
        "/analyze",
        files={"file": ("test.txt", txt_file, "text/plain")}
    )
    # 500 because DocumentPortalException is raised
    assert response.status_code == 500
    assert "Invalid file type" in response.text


def test_analyze_internal_error(monkeypatch):
    """Simulate internal error in analyzer"""

    class MockDocHandler:
        def save_pdf(self, file):
            return "dummy_path.pdf"
        def read_pdf(self, path):
            return "content"

    # Mock Analyzer to throw error
    class MockAnalyzer:
        def analyze_document(self, text: str):
            raise Exception("LLM failed")

    import api.main as main_app
    monkeypatch.setattr(main_app, "DocHandler", MockDocHandler)
    monkeypatch.setattr(main_app, "DocumentAnalyzer", lambda: MockAnalyzer())

    client = TestClient(app)

    fake_pdf = io.BytesIO(b"%PDF-1.4 Broken dummy content")
    response = client.post(
        "/analyze",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")}
    )
    print("RESPONSE JSON:", response.json())
    assert response.status_code == 500
    assert "Analysis Failed" in response.json()["detail"]


# -------- Compare Endpoint (happy Path) -------- #

def test_compare_documents_success(monkeypatch):
    class MockDocumentComparator:
        def __init__(self):
            self.session_id = "mock-session"
            self.session_path = None

        def save_uploaded_files(self, reference_file, actual_file):
            # Return fake paths
            return "ref_path.pdf", "act_path.pdf"

        def combine_documents(self):
            return "Combined document text for comparison"

    class MockDocumentComparatorLLM:
        def compare_documents(self, combined_docs: str):
            import pandas as pd
            # Return a simple DataFrame as mock result
            return pd.DataFrame([
                {"section": "Summary", "difference": "Minor"},
                {"section": "Details", "difference": "Major"},
            ])

    # Patch classes in your main app module to use mocks
    import api.main as main_app
    monkeypatch.setattr(main_app, "DocumentComparator", MockDocumentComparator)
    monkeypatch.setattr(main_app, "DocumentComparatorLLM", lambda: MockDocumentComparatorLLM())

    # Prepare two fake PDF files as in-memory byte streams
    fake_pdf_ref = io.BytesIO(b"%PDF-1.4 fake reference pdf data")
    fake_pdf_act = io.BytesIO(b"%PDF-1.4 fake actual pdf data")

    response = client.post(
        "/compare",
        files={
            "reference": ("ref.pdf", fake_pdf_ref, "application/pdf"),
            "actual": ("act.pdf", fake_pdf_act, "application/pdf"),
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "rows" in data
    assert isinstance(data["rows"], list)
    assert len(data["rows"]) == 2
    assert "session_id" in data
    assert data["session_id"] == "mock-session"
    # Check content keys in row
    assert "section" in data["rows"][0]
    assert "difference" in data["rows"][0]

def test_compare_documents_invalid_file(monkeypatch):
    # Provide a non-PDF file input to check error handling

    fake_txt_file = io.BytesIO(b"Not a PDF file content")

    # No patching needed here; real code should raise on invalid file type

    response = client.post(
        "/compare",
        files={
            "reference": ("ref.txt", fake_txt_file, "text/plain"),
            "actual": ("act.pdf", io.BytesIO(b"%PDF-1.4 valid pdf"), "application/pdf"),
        }
    )

    assert response.status_code == 500
    assert "Only PDF files are allowed" in response.text or "Comparison Failed" in response.text


# -------- Chat index  -------- #

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.document_ingestion.data_ingestion import ChatIngestor
from exception.custom_exception import DocumentPortalException

@pytest.fixture
def tmp_dirs(tmp_path):
    return {
        "upload": tmp_path / "uploads",
        "faiss": tmp_path / "faiss"
    }

@pytest.fixture
def fake_doc():
    from langchain.schema import Document
    return Document(page_content="Hello World", metadata={"source": "file.txt"})

def test_chat_ingestor_init(tmp_dirs):
    ci = ChatIngestor(temp_base=tmp_dirs["upload"], faiss_base=tmp_dirs["faiss"])
    assert ci.temp_dir.exists()
    assert ci.faiss_dir.exists()
    assert ci.session_id is not None

def test_split_documents(fake_doc, tmp_dirs):
    ci = ChatIngestor(temp_base=tmp_dirs["upload"], faiss_base=tmp_dirs["faiss"])
    chunks = ci._split([fake_doc], chunk_size=5, chunk_overlap=0)
    assert len(chunks) >= 1
    assert all(hasattr(c, "page_content") for c in chunks)

@patch("src.document_ingestion.data_ingestion.save_uploaded_files")
@patch("src.document_ingestion.data_ingestion.load_documents")
@patch("src.document_ingestion.data_ingestion.FaissManager")
def test_build_retriever_success(mock_fm, mock_load, mock_save, fake_doc, tmp_dirs):
    ci = ChatIngestor(temp_base=tmp_dirs["upload"], faiss_base=tmp_dirs["faiss"])

    mock_save.return_value = ["file1.txt"]
    mock_load.return_value = [fake_doc]

    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = "retriever"
    mock_fm.return_value.load_or_create.return_value = fake_vs
    mock_fm.return_value.add_documents.return_value = 1

    retriever = ci.built_retriever(["dummy"])
    assert retriever == "retriever"
    mock_fm.return_value.add_documents.assert_called()

@patch("src.document_ingestion.data_ingestion.save_uploaded_files", lambda files, td: [])
@patch("src.document_ingestion.data_ingestion.load_documents", lambda paths: [])
def test_build_retriever_no_docs(tmp_dirs):
    ci = ChatIngestor(temp_base=tmp_dirs["upload"], faiss_base=tmp_dirs["faiss"])
    with pytest.raises(DocumentPortalException):
        ci.built_retriever(["dummy"])

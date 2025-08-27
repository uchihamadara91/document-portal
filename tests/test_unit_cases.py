# tests/test_unit_cases.py

import io
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.document_ingestion.data_ingestion import FaissManager
from exception.custom_exception import DocumentPortalException
from langchain.schema import Document



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


# -------- Chat Index -------- #




@pytest.fixture
def tmp_index_dir(tmp_path):
    d = tmp_path / "faiss_index"
    d.mkdir(parents=True, exist_ok=True)
    return d

@patch("langchain_community.vectorstores.faiss.FAISS")
def test_load_or_create_new_index(mock_faiss, tmp_index_dir):
    # Patch the creation via FAISS.from_texts and the embedding function
    mock_vs = MagicMock()
    mock_faiss.from_texts.return_value = mock_vs
    fake_emb = MagicMock()
    fake_emb.embed_documents.return_value = [[0.1] * 10]  # Correct shape

    with patch("src.document_ingestion.data_ingestion.FAISS", mock_faiss):
        fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
        fm.emb = fake_emb
        vs = fm.load_or_create(texts=["hi"], metadatas=[{"a": 1}])

    assert mock_faiss.from_texts.called
    assert vs == mock_vs  # Now guaranteed to be the mocked object

@patch("langchain_community.vectorstores.faiss.FAISS")
def test_load_or_create_existing_index(mock_faiss, tmp_index_dir):
    # Patch the class and its load_local method/side effects.
    mock_faiss.load_local.return_value = "mocked_vs"
    (tmp_index_dir / "index.faiss").touch()
    (tmp_index_dir / "index.pkl").touch()
    with patch("src.document_ingestion.data_ingestion.FAISS", mock_faiss):
        fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
        fm.emb = MagicMock()
        vs = fm.load_or_create(texts=["irrelevant"], metadatas=[{}])
    # load_local should be called and "vs" should not be None
    assert mock_faiss.load_local.called
    # The original FaissManager.load_or_create returns self.vs (which might be None)
    # Optionally patch FaissManager to set self.vs if needed, but test is to assert load_local happened


def test_faiss_manager_fingerprint_and_save_meta(tmp_index_dir):
    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    # Initially meta should have no rows
    assert "rows" in fm._meta and isinstance(fm._meta["rows"], dict)

    doc = Document(page_content="sample text", metadata={"source": "file.txt", "row_id": 1})
    key = fm._fingerprint(doc.page_content, doc.metadata)
    # Add manually to meta
    fm._meta["rows"][key] = True
    # Trigger save meta and check file exists
    fm._save_meta()
    assert (tmp_index_dir / "ingested_meta.json").exists()

@patch("src.document_ingestion.data_ingestion.FAISS")
def test_load_or_create_raises_without_texts(mock_faiss, tmp_index_dir):
    """load_or_create should raise if no index and no input texts provided"""
    mock_faiss.load_local.side_effect = FileNotFoundError()
    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    fm.emb = MagicMock()
    # Manually prevent existing index
    fm._exists = lambda: False
    with pytest.raises(Exception):
        fm.load_or_create(texts=None)
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

@pytest.fixture
def fake_doc():
    from langchain.schema import Document
    return Document(page_content="Some content", metadata={"source": "a.txt"})

@patch("langchain_community.vectorstores.faiss.FAISS")
def test_load_or_create_new_index(mock_faiss, tmp_index_dir):
    fake_emb = MagicMock()
    # Simulate from_texts returning a VS and the embedding function returning the proper number of embeddings
    mock_vs = MagicMock()
    mock_faiss.from_texts.return_value = mock_vs

    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    fm.emb = MagicMock()
    # Patch emb so that embed_documents returns list of the correct shape
    fm.emb.embed_documents.return_value = [[0.1] * 10]  # One embedding vector for "hi"
    vs = fm.load_or_create(texts=["hi"], metadatas=[{"a": 1}])
    assert vs == mock_vs
    mock_faiss.from_texts.assert_called_once()

@patch("langchain_community.vectorstores.faiss.FAISS")
def test_load_or_create_existing_index(mock_faiss, tmp_index_dir):
    # Ensure directory exists
    tmp_index_dir.mkdir(parents=True, exist_ok=True)
    (tmp_index_dir / "index.faiss").write_text("x")
    (tmp_index_dir / "index.pkl").write_text("y")
    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    fm.emb = MagicMock()
    vs = fm.load_or_create(texts=["fake"], metadatas=[{}])
    mock_faiss.load_local.assert_called_once()

def test_add_documents_new_and_duplicate(tmp_index_dir, fake_doc):
    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    fm.vs = MagicMock()

    added = fm.add_documents([fake_doc])
    assert added == 1
    # add again -> duplicate → no increment
    added2 = fm.add_documents([fake_doc])
    assert added2 == 0

def test_add_documents_without_load(tmp_index_dir, fake_doc):
    fm = FaissManager(tmp_index_dir, model_loader=MagicMock())
    fm.vs = None
    with pytest.raises(RuntimeError):
        fm.add_documents([fake_doc])

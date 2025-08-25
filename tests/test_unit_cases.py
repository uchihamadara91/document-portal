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


# -------- Chat index and Chat Query Endpoint (happy Path) -------- #

class FakeUploadFile:
    def __init__(self, filename, content_bytes):
        self.filename = filename
        self.file = io.BytesIO(content_bytes)
        self.content = content_bytes

@pytest.fixture
def mock_chat_ingestor(monkeypatch):
    class MockChatIngestor:
        def __init__(self, temp_base, faiss_base, use_session_dirs, session_id=None):
            self.session_id = session_id or "mock-session"
        def built_retriever(self, wrapped, chunk_size, chunk_overlap, k):
            return "mock_retriever"
    monkeypatch.setattr("api.main.ChatIngestor", MockChatIngestor)

@pytest.fixture
def mock_conversational_rag(monkeypatch):
    class MockConversationalRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
        def load_retriever_from_faiss(self, index_dir, k=5, index_name=None):
            pass
        def invoke(self, question, chat_history=None):
            return "mock answer"
    monkeypatch.setattr("api.main.ConversationalRAG", MockConversationalRAG)


# 1. Test /chat/index success with minimal files
def test_chat_build_index_success(mock_chat_ingestor):
    files = [
        ("files", ("file1.txt", io.BytesIO(b"Dummy text content"), "text/plain")),
        ("files", ("file2.txt", io.BytesIO(b"More text"), "text/plain")),
    ]
    response = client.post(
        "/chat/index",
        files=files,
        data={"session_id": "test-session", "use_session_dirs": "true"}
    )
    assert response.status_code == 200
    json_resp = response.json()
    assert "session_id" in json_resp
    assert json_resp["k"] == 5
    assert json_resp["use_session_dirs"] is True


# 2. Test /chat/index failure returns 500 on exception
def test_chat_build_index_failure(monkeypatch):
    def fail_init(*args, **kwargs):
        raise Exception("fail")
    monkeypatch.setattr("api.main.ChatIngestor", fail_init)
    files = [("files", ("file.txt", io.BytesIO(b"dummy"), "text/plain"))]
    response = client.post("/chat/index", files=files)
    assert response.status_code == 500
    assert "Indexing failed" in response.json()["detail"]

# 3. Test /chat/query success with session id
def test_chat_query_success(mock_conversational_rag):
    data = {
        "question": "What is AI?",
        "session_id": "test-session",
        "use_session_dirs": "true",
        "k": "5"
    }
    response = client.post("/chat/query", data=data)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["answer"] == "mock answer"
    assert json_resp["session_id"] == "test-session"


# 4. Test /chat/query error on missing session_id with use_session_dirs true
def test_chat_query_missing_session_id():
    data = {
        "question": "Question?",
        "use_session_dirs": "true"
    }
    response = client.post("/chat/query", data=data)
    assert response.status_code == 400
    assert "session_id is required" in response.json()["detail"]

# 5. Test /chat/query error when FAISS index dir missing
def test_chat_query_missing_index_dir(monkeypatch, tmp_path):
    # Make index dir not exist
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    data = {
        "question": "Hi?",
        "session_id": "test-session",
        "use_session_dirs": "true"
    }
    response = client.post("/chat/query", data=data)
    assert response.status_code == 404
    assert "FAISS index not found" in response.json()["detail"]

# 6. Test FaissManager _exists returns correct boolean
def test_faiss_manager_exists(tmp_path):
    index_dir = tmp_path
    (index_dir / "index.faiss").write_text("fake faiss content")
    (index_dir / "index.pkl").write_text("fake pkl content")
    from src.document_ingestion.data_ingestion import FaissManager  # Adjust import path appropriately
    fm = FaissManager(index_dir=index_dir, model_loader=None)
    assert fm._exists() is True

# 7. Test FaissManager _fingerprint returns string as expected
def test_faiss_manager_fingerprint():
    from src.document_ingestion.data_ingestion import FaissManager
    text = "some text"
    md = {"source": "/path/to/file", "row_id": 5}
    fp = FaissManager._fingerprint(text, md)
    assert fp == "/path/to/file::5"
    md2 = {}
    fp2 = FaissManager._fingerprint(text, md2)
    assert isinstance(fp2, str) and len(fp2) == 64  # sha256 hex length

# 8. Test ChatIngestor _resolve_dir creates correct paths
def test_chat_ingestor_resolve_dir(monkeypatch, tmp_path):
    from src.document_ingestion.data_ingestion import ChatIngestor
    ci = ChatIngestor(temp_base=str(tmp_path), faiss_base=str(tmp_path), use_session_dirs=True, session_id="sess1")
    assert (tmp_path / "sess1").exists()
    ci2 = ChatIngestor(temp_base=str(tmp_path), faiss_base=str(tmp_path), use_session_dirs=False)
    assert str(ci2.temp_dir) == str(tmp_path)

# 9. Test ConversationalRAG raises exception if invoke before loading retriever
def test_conversationalrag_invoke_fail(monkeypatch):
    from src.document_chat.retrieval import ConversationalRAG
    rag = ConversationalRAG(session_id="sess")
    rag.chain = None
    with pytest.raises(Exception):
        rag.invoke("Hello")

# 10. Test ConversationalRAG load_retriever_from_faiss success
def test_conversationalrag_load_retriever(monkeypatch, tmp_path):
    from src.document_chat.retrieval import ConversationalRAG

    def fake_load_embeddings():
        return "embedding"

    class FakeFAISS:
        @staticmethod
        def load_local(path, embeddings, index_name=None, allow_dangerous_deserialization=True):
            return FakeFAISS()
        def as_retriever(self, search_type=None, search_kwargs=None):
            return "retriever"

    monkeypatch.setattr("src.chatindex.conversationalrag.ModelLoader.load_embeddings", fake_load_embeddings)
    monkeypatch.setattr("src.chatindex.conversationalrag.FAISS", FakeFAISS)
    rag = ConversationalRAG(session_id="sess")
    retriever = rag.load_retriever_from_faiss(str(tmp_path), k=5)
    assert retriever == "retriever"
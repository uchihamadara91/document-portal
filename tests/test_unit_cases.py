# tests/test_unit_cases.py

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

"""
Here we simulate uploading a PDF. Since real PDF parsing requires file I/O
You can mock it or upload a dummy file
"""

import io

def test_analyze_valid_pdf(monkeypatch):
    
    #Fake PDF Content
    file_content = io.BytesIO(b"%PDF-1.4 fake pdf content")


    # Monkeypatch analyzer to return fixed result
    from src.document_analyzer.data_analysis import DocumentAnalyzer
    monkeypatch.setattr(DocumentAnalyzer, "analyze_document",lambda self, text: {"summary": "fake summary"})

    response = client.post("/analyze", files={"file": ("test.pdf", file_content, "application/pdf")})
    assert response.status_code == 200
    assert response.json()["summary"] == "fake summary"


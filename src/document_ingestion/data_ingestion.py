from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Optional, Dict, Any

import fitz
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

from utils.file_io import _session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


#FAISS Manager (load-or-create)

class FaissManager:
    def __init__(self, index_dir : Path,model_loader: Optional[ModelLoader]):
        self.log = CustomLogger().get_logger()

        try:
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            self.meta_path = self.index_dir / "ingested_meta.json"
            self._meta: Dict[str, Any] = {"rows": {}}

            if self.meta_path.exists():
                try:
                    self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except Exception:
                    self._meta = {"rows": {}}

            self.model_loader = model_loader or ModelLoader()
            self.emb = self.model_loader.load_embeddings()
            self.vs: Optional[FAISS] = None

        except Exception as e:
            self.log.error(f"Error initializing FaissManager: {e}")
            raise DocumentPortalException("Failed to initialize FAISS Manager", e) from e
        
    def _exists(self) ->bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl".exists()
                                                              )

    @staticmethod
    def _fingerprint():
        pass

    def _save_meta(self):
        pass

    def add_documents(self):
        pass

    def load_or_create(self):
        pass









class ChatIngestor:
    pass
class DocHandler:
    pass
class DocumentComparator:
    pass

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterable
from langchain.schema import Document
import json
from utils.model_loader import ModelLoader
from langchain.vectorstores import FAISS
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents
import fitz

class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader]):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            self.meta_path = self.index_dir / "ingested_meta.json"
            self._meta: Dict[str,Any] = {"rows": {}}

            if self.meta_path.exists():
                try:
                    self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

                except Exception:
                    self._meta = {"rows": {}}

            self.model_loader = model_loader or ModelLoader()
            self.emb = self.model_loader.load_embeddings()
            self.vs: Optional[FAISS] = None
            self.log.info("FaissManager Initialized", index_dir=self.index_dir)

        except Exception as e:
            self.log.error(f"FaissManager Initialization Error {e}")
            raise DocumentPortalException("Faiss Manager Initialization Fialed", e) from e


    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl")
    
    @staticmethod
    def _fingerprint(text:str,md: Dict[str, Any]) ->str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")

        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding='utf-8')
    
    def add_documents(self, docs:List[Document]):
        if self.vs is None:
            raise RuntimeError("VS is empty , run load_or_create() before add_docments()")
        
        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        return len(new_docs)

        
    def load_or_create(self, texts: Optional[List[str]], metadatas: Optional[List[Dict]]):
        if self._exists():
            FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True

            )
            return self.vs
        
        if not texts:
            raise DocumentPortalException("No FAISS index &NO data to create FAISS Index")
        
        self.vs = FAISS.from_texts(texts=texts, embeddings=self.emb, metadatas=metadatas or [] )
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
class ChatIngestor:
    def __init__(self,
                 temp_base: str = "data",
                 faiss_base: str = "faiss_index",
                 use_session_dirs: bool = True,
                 session_id: Optional[str] = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)
            self.log.info(
                "ChatIngestor Initialized",
                session_id=self.session_id,
                temp_base=str(self.temp_base),
                faiss_base=str(self.faiss_base),
                sessionized=self.use_session_dirs
            )
        except Exception as e:
            self.log.error(f"Error initializing ChatIngestor", error=str(e))
            raise DocumentPortalException("Failed to initialize ChatIngestor", e) from e


    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base


    def _split(self, 
               docs:List[Document],
               chunk_size:int = 1000,
               chunk_overlap = 200
               ) ->List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        self.log.info("Documents Split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriever(self, 
                        uploaded_files :Iterable,
                        *,
                        chunk_size: int=1000,
                        chunk_overlap: int=200,
                        k: int =5):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)

            if not docs:
                raise ValueError("No valid documents")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            except:
                vs = fm.load_or_create(texts=texts, metadatas=metas) # add data in vector store

            added = fm.add_documents(chunks) # in data inventory

            self.log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            return vs.as_retriever(search_type='similarity', search_kwargs={"K":k})
        
        except Exception as e:
            self.log.error(f"Error building retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e
        


class DocHandler:
    """PDF Save + read for analysis"""

    def __init__(self, data_dir:Optional[str] = None,session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE-PATH", os.path.join(os.getcwd(),"data","document_analyzer"))
        self.session_id = session_id or generate_session_id()
        self.session_path = os.path.join(self.data_dir, self.session_path)
        os.makedirs(self.session_path, exist_ok=True)
        self.log.info("DocHandler Initialized",data_dir=self.data_dir, session_path=self.session_path)

    def save_pdf(self, uploaded_file: Document)->str:
        try:
            filename = os.path.basename(uploaded_file)
            if filename.lower().endswith(".pdf"):
                raise ValueError("Only PDFs for now!!!")
            save_path = os.path.join(self.session_path , self.filename)

            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())

                else:
                    f.write(uploaded_file.getbuffer())
            self.log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)

        except Exception as e:
            self.log.error("Failed to save PDF", error=str(e),session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}",e) from e

    def read_pdf(self, pdf_path: str) ->str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as f:
                for page_num in range(f.page_count):
                    page = f.load_page(page_num)
                    text_chunks.append(f"\n---- Page {page_num +1} --- \n  {page.get_text()}")
            text = "\n".join(text_chunks)
            self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text

        except Exception as e:
            self.log.error("Failed to read PDF", error=str(e),session_id=self.session_id)
            raise DocumentPortalException(f"Failed to read PDF: {pdf_path}",e) from e





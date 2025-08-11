import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException

class DocumentIngestion:
    def __init__(self, base_dir:str="data\\document_compare"):
        
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True) 
    
    def delete_existing_files(self):
        """
        Deletes existing files at specific paths
        """
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink() # Deletes the file
                        self.log.info("File Deleted", path=str(file))

                self.log.info("Directory Cleaned", directory=str(self.base_dir))

        except Exception as e:
            self.log.error(f"Error deleting existing files: {e}")
            raise DocumentPortalException("An error occured while deleting existing files", sys)


    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves uploaded files to a specific directory
        """
        try:
            self.delete_existing_files()
            self.log.info("Existing files deleted")

            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Upload Only PDF files")
            
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files saved", reference=str(ref_path), actual=str(act_path))
            return ref_path, act_path

        except Exception as e:
            self.log.error(f"Error saving uploaded files: {e}")
            raise DocumentPortalException("An error occured while saving uploaded files", sys)
            


    def read_pdf(self, pdf_path: Path)->str:
        """
        Reads a pdf file and extracts text from each page
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                
                all_text = []

                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text() #type: ignore
                    if text.strip():
                        all_text.append(f"\n --- Page {page_num + 1} --- \n{text}")
                self.log.info("PDF Read Successfully", file=str(pdf_path), pages=len(all_text))
                return "\n".join(all_text)     
        
        except Exception as e:
            self.log.error(f"Error reading PDF: {e}")
            raise DocumentPortalException("An error occured while reading PDF files", sys)        

    def combine_documents(self)->str:
        """  
        Combines documents to be compared into a single document
        """
        try:
            content_dict = {}
            doc_parts = []

            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

                for filename, content in content_dict.items():
                    doc_parts.append(f"Document: {filename}\n{content}")
                
                combined_text = "\n\n".join(doc_parts)
                self.log.info("Documents Combined", count=len(doc_parts))
                return combined_text
                                   
        except Exception as e:
            self.log.error(f"Error combining documents: {e}")
            raise DocumentPortalException("An error occured while combining documents", sys)

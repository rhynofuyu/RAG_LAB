from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import fitz
import re


class DocumentProcessor:
    def __init__(self):
        pass

    def get_pdf_chunks(self, pdf_docs):
        chunks = []
        metadatas = []
        
        for pdf in pdf_docs:
            file_name = getattr(pdf, 'name', 'document.pdf')
            
            pdf.seek(0)
            pdf_bytes = pdf.read()
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                text = page.get_text("text")
                
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  
                text = re.sub(r'\.([A-Z])', r'. \1', text)       
                text = re.sub(r'\s+', ' ', text)                  
                text = text.strip()
                
                if text:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    page_chunks = text_splitter.split_text(text)
                    
                    for chunk_idx, chunk in enumerate(page_chunks):
                        chunks.append(chunk)
                        metadatas.append({
                            "page_number": page_num + 1,
                            "chunk_index": chunk_idx + 1,
                            "file_name": file_name
                        })
            
            doc.close()
        
        return chunks, metadatas

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

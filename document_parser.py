import fitz
import io
import base64
import os
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import pandas as pd
import re
from typing import List, Dict, Any, Tuple
import streamlit as st

class DocumentParser:
    def __init__(self):
        self.vlm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
    
    def extract_content_from_pdfs(self, pdf_docs) -> Tuple[List[str], List[Dict[str, Any]]]:
        all_chunks = []
        all_metadatas = []
        
        for pdf in pdf_docs:
            file_name = getattr(pdf, 'name', 'document.pdf')
            pdf.seek(0)
            pdf_bytes = pdf.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                text_content = self._extract_text_content(page, file_name, page_num)
                table_content = self._extract_table_content(page, file_name, page_num)
                image_content = self._extract_image_content(page, file_name, page_num)
                
                all_chunks.extend(text_content['chunks'])
                all_metadatas.extend(text_content['metadatas'])
                
                all_chunks.extend(table_content['chunks'])
                all_metadatas.extend(table_content['metadatas'])
                
                all_chunks.extend(image_content['chunks'])
                all_metadatas.extend(image_content['metadatas'])
            
            doc.close()
        
        return all_chunks, all_metadatas
    
    def _extract_text_content(self, page, file_name: str, page_num: int) -> Dict[str, List]:
        text = page.get_text("text")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        chunks = []
        metadatas = []
        
        if text:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            text_chunks = text_splitter.split_text(text)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "page_number": page_num + 1,
                    "chunk_index": chunk_idx + 1,
                    "file_name": file_name,
                    "content_type": "text",
                    "source": f"{file_name}_page_{page_num + 1}_text_chunk_{chunk_idx + 1}"
                })
        
        return {"chunks": chunks, "metadatas": metadatas}
    
    def _extract_table_content(self, page, file_name: str, page_num: int) -> Dict[str, List]:
        chunks = []
        metadatas = []
        
        try:
            tables = page.find_tables()
            for table_idx, table in enumerate(tables):
                table_data = table.extract()
                if table_data and len(table_data) > 1:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    table_text = f"Table {table_idx + 1}:\n{df.to_string(index=False)}"
                    
                    chunks.append(table_text)
                    metadatas.append({
                        "page_number": page_num + 1,
                        "chunk_index": table_idx + 1,
                        "file_name": file_name,
                        "content_type": "table",
                        "table_rows": len(df),
                        "table_columns": len(df.columns),
                        "source": f"{file_name}_page_{page_num + 1}_table_{table_idx + 1}"
                    })
        except Exception as e:
            pass
        
        return {"chunks": chunks, "metadatas": metadatas}
    
    def _extract_image_content(self, page, file_name: str, page_num: int) -> Dict[str, List]:
        chunks = []
        metadatas = []
        
        try:
            image_list = page.get_images()
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        
                        description = self._analyze_image_with_vlm(img_base64)
                        
                        if description:
                            image_text = f"Image {img_idx + 1} description: {description}"
                            chunks.append(image_text)
                            metadatas.append({
                                "page_number": page_num + 1,
                                "chunk_index": img_idx + 1,
                                "file_name": file_name,
                                "content_type": "image",
                                "image_format": "png",
                                "source": f"{file_name}_page_{page_num + 1}_image_{img_idx + 1}"
                            })
                    
                    pix = None
                except Exception as e:
                    continue
        except Exception as e:
            pass
        
        return {"chunks": chunks, "metadatas": metadatas}
    
    def _analyze_image_with_vlm(self, image_base64: str) -> str:
        try:
            prompt = """Analyze this image and provide a detailed description including:
1. What objects, people, or elements are visible
2. Any text content that appears in the image
3. The context or purpose of the image
4. Any data, charts, diagrams, or technical information shown
5. Colors, layout, and visual structure

Provide a comprehensive description that would be useful for someone who cannot see the image."""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            )
            
            response = self.vlm.invoke([message])
            return response.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

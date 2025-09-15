import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient


class VectorStoreManager:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "rag-lab-index"
        
    def get_vectorstore(self, text_chunks, metadatas=None):
        pinecone_client = PineconeClient(api_key=self.pinecone_api_key)

        if self.index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=self.index_name,
                dimension=768, 
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if metadatas:
            vectorstore = PineconeVectorStore.from_texts(
                texts=text_chunks, 
                embedding=embeddings, 
                metadatas=metadatas,
                index_name=self.index_name
            )
        else:
            vectorstore = PineconeVectorStore.from_texts(
                texts=text_chunks, 
                embedding=embeddings, 
                index_name=self.index_name
            )
        return vectorstore

    def get_conversation_chain(self, vectorstore):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        retriever = vectorstore.as_retriever()
        return llm, retriever

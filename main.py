import streamlit as st
import os
from dotenv import load_dotenv

from database.db_manager import DatabaseManager
from document_processing.processor import DocumentProcessor
from vector_store.manager import VectorStoreManager
from query_processing.processor import QueryProcessor
from ui.components import UIComponents
from config.settings import Config


def initialize_session_state():
    """Initialize all session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = Config.DEFAULT_TOP_K
    if "use_multi_query" not in st.session_state:
        st.session_state.use_multi_query = True
    if "num_query_variations" not in st.session_state:
        st.session_state.num_query_variations = Config.DEFAULT_QUERY_VARIATIONS
    if "top_k_per_query" not in st.session_state:
        st.session_state.top_k_per_query = Config.DEFAULT_TOP_K_PER_QUERY


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Lab", page_icon=":books:")
    
    # Initialize components
    db_manager = DatabaseManager(Config.DATABASE_PATH)
    doc_processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    query_processor = QueryProcessor()
    ui_components = UIComponents(db_manager, doc_processor, vector_manager, query_processor)
    
    # Initialize session state
    initialize_session_state()

    # Handle authentication
    if not st.session_state.authenticated:
        ui_components.login_form()
        return

    # Main application
    ui_components.user_sidebar()
    ui_components.chat_interface()


if __name__ == '__main__':
    main()

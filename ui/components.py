import streamlit as st
from langchain.schema import HumanMessage, AIMessage


class UIComponents:
    def __init__(self, db_manager, doc_processor, vector_manager, query_processor):
        self.db_manager = db_manager
        self.doc_processor = doc_processor
        self.vector_manager = vector_manager
        self.query_processor = query_processor

    def user_sidebar(self):
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state.username}**")
            
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            if st.button("Clear Chat History"):
                self.db_manager.clear_user_chat_history(st.session_state.username)
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
            
            st.divider()
            
            st.subheader("Retriever Settings")
            st.session_state.use_multi_query = st.checkbox(
                "Enable Multi-Query RAG",
                value=st.session_state.get('use_multi_query', True),
                help="Automatically generate multiple query variations to improve search results"
            )
            
            if st.session_state.use_multi_query:
                st.session_state.num_query_variations = st.slider(
                    "Number of query variations:",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.get('num_query_variations', 3),
                    step=1,
                    help="Number of query variations that will be automatically generated"
                )
                
                st.session_state.top_k_per_query = st.slider(
                    "Documents per query:",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.get('top_k_per_query', 3),
                    step=1,
                    help="Number of documents to retrieve for each query variation"
                )
            
            st.session_state.top_k = st.slider(
                "Final documents to use:",
                min_value=1,
                max_value=10,
                value=st.session_state.get('top_k', 4),
                step=1,
                help="Total number of final documents to use for answering"
            )
            
            st.divider()
            st.subheader("Your Document")
            pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        text_chunks, metadatas = self.doc_processor.get_pdf_chunks(pdf_docs)
                        vectorstore = self.vector_manager.get_vectorstore(text_chunks, metadatas)
                        st.session_state.llm, st.session_state.retriever = self.vector_manager.get_conversation_chain(vectorstore)
                    st.success("Processing complete. You can now ask questions.")
                else:
                    st.warning("Please upload at least one PDF file.")

    def login_form(self):
        st.header("RAG Lab - Please Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            login_button = st.form_submit_button("Login")
            register_button = st.form_submit_button("Register")

            if login_button:
                if self.db_manager.verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            if register_button:
                if username and password:
                    if self.db_manager.register_user(username, password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Please enter username and password")

    def chat_interface(self):
        st.header("RAG LAB")

        if not st.session_state.chat_history:
            st.session_state.chat_history = self.db_manager.load_chat_history_with_window(st.session_state.username, 5)

        for message in st.session_state.chat_history:
            with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
                st.write(message.content)

        if user_question := st.chat_input("Ask a question about your document..."):
            if st.session_state.retriever:
                with st.chat_message("human"):
                    st.write(user_question)

                with st.spinner("Thinking..."):
                    settings = {
                        'use_multi_query': st.session_state.get('use_multi_query', True),
                        'num_query_variations': st.session_state.get('num_query_variations', 3),
                        'top_k_per_query': st.session_state.get('top_k_per_query', 3),
                        'top_k': st.session_state.get('top_k', 4)
                    }
                    ai_answer = self.query_processor.handle_user_question(
                        user_question, 
                        st.session_state.llm, 
                        st.session_state.retriever, 
                        st.session_state.chat_history,
                        settings
                    )

                with st.chat_message("ai"):
                    st.write(ai_answer)

                self.db_manager.save_message_to_db(st.session_state.username, user_question, 'user')
                self.db_manager.save_message_to_db(st.session_state.username, ai_answer, 'ai')

                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=ai_answer))
            else:
                st.warning("Please upload and process a PDF document first.")

import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import asyncio
import nest_asyncio
import sqlite3
import hashlib
from datetime import datetime

nest_asyncio.apply()

def init_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  message_content TEXT,
                  message_type TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (username) REFERENCES users (username))''')
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False
    


def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None


def change_password(username, old_password, new_password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        if verify_user(username, old_password):
            c.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(new_password), username))
            conn.commit()
            return True
        else:
            return False
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()


def save_message_to_db(username, message_content, message_type):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO chat_history (username, message_content, message_type, timestamp) VALUES (?, ?, ?, ?)",
              (username, message_content, message_type, timestamp))
    conn.commit()
    conn.close()

def load_chat_history_from_db(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT message_content, message_type FROM chat_history WHERE username = ? ORDER BY timestamp",
              (username,))
    results = c.fetchall()
    conn.close()
    return results

def clear_user_chat_history(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def login_form():
    with st.sidebar:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    load_chat_history_for_user(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with col2:
            if st.button("Register"):
                if username and password:
                    if register_user(username, password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Please enter username and password")

def load_chat_history_for_user(username):
    from langchain.schema import HumanMessage, AIMessage
    
    db_history = load_chat_history_from_db(username)
    if db_history:
        chat_messages = []
        for message_content, message_type in db_history:
            if message_type == 'user':
                chat_messages.append(HumanMessage(content=message_content))
            else:
                chat_messages.append(AIMessage(content=message_content))
        st.session_state.chat_history = chat_messages
    else:
        st.session_state.chat_history = None

def user_sidebar():
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.username}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.conversation = None
                st.session_state.chat_history = None
                st.rerun()
        with col2:
            if st.button("Clear History"):
                if st.session_state.username:
                    clear_user_chat_history(st.session_state.username)
                    st.session_state.chat_history = None
                    st.success("Chat history cleared!")
                    st.rerun()
        st.divider()
        st.subheader("Change Password")
        old_password = st.text_input("Old Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.button("Change Password"):
            if new_password == confirm_password:
                if change_password(st.session_state.username, old_password, new_password):
                    st.success("Password changed successfully!")
                else:
                    st.error("Failed to change password. Please check your old password.")
            else:
                st.error("New passwords do not match.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    #RecursiveCharacterTextSplitter ["\n\n", "\n", " ", ""]
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vectorstore.as_retriever())
    # Modify the top docs: retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    username = st.session_state.username
    save_message_to_db(username, user_question, 'user')
    
    latest_ai_response = response['chat_history'][-1].content
    save_message_to_db(username, latest_ai_response, 'ai')
    
    for i, message in enumerate(st.session_state.chat_history):
        if (i%2==0):
            st.chat_message("human").write(message.content)
        else:
            st.chat_message("ai").write(message.content)

def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    st.set_page_config(page_title="RAG Lab 04", page_icon=":books:")
    init_database()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if not st.session_state.authenticated:
        st.header("RAG Lab 04 - Please Login")
        st.write("Please login or register to access the application.")
        login_form()
        return
    user_sidebar()
    
    st.header("RAG Lab 04")
    
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if (i%2==0):
                st.chat_message("human").write(message.content)
            else:
                st.chat_message("ai").write(message.content)
    
    if user_question := st.chat_input("Ask a question about your document:"):
        if st.session_state.conversation:
            st.chat_message("human").write(user_question)
            
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            username = st.session_state.username
            save_message_to_db(username, user_question, 'user')
            
            latest_response = st.session_state.chat_history[-1].content
            save_message_to_db(username, latest_response, 'ai')
            
            st.chat_message("ai").write(latest_response)
        else:
            st.warning("Please upload a PDF document to start the conversation.")
    
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete. You can now ask questions about your document.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()

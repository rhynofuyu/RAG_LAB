import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient
import sqlite3
import hashlib
from datetime import datetime
import asyncio
import fitz
import re
import cohere
from document_parser import DocumentParser


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
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

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

def load_chat_history_with_window(username, window_size):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    limit = window_size * 2
    c.execute("""
        SELECT message_content, message_type 
        FROM chat_history 
        WHERE username = ? 
        ORDER BY timestamp DESC
        LIMIT ?
    """, (username, limit))
    results = c.fetchall()
    conn.close()
    
    results.reverse()

    chat_messages = []
    for content, msg_type in results:
        if msg_type == 'user':
            chat_messages.append(HumanMessage(content=content))
        else:
            chat_messages.append(AIMessage(content=content))
    return chat_messages

def clear_user_chat_history(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def get_pdf_chunks(pdf_docs):
    parser = DocumentParser()
    chunks, metadatas = parser.extract_content_from_pdfs(pdf_docs)
    return chunks, metadatas

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, metadatas=None):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    pinecone_client = PineconeClient(api_key=pinecone_api_key)
    index_name = "rag-lab-index"

    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=3072, 
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
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if metadatas:
        vectorstore = PineconeVectorStore.from_texts(
            texts=text_chunks, 
            embedding=embeddings, 
            metadatas=metadatas,
            index_name=index_name
        )
    else:
        vectorstore = PineconeVectorStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vectorstore.as_retriever()
    return llm, retriever

def generate_query_variations(llm, original_question, num_variations=3):
    """
    Generate multiple variations of the original question from different perspectives
    """
    variation_prompt = f"""
You are an AI assistant specialized in creating query variations. Please create {num_variations} different variations of the following question.
Each variation should:
1. Maintain the core meaning of the original question
2. Use different words and sentence structures
3. Approach from different angles or aspects
4. Can be more specific or more general than the original

Original question: "{original_question}"

Please return {num_variations} variations, each on a separate line, numbered from 1 to {num_variations}.
Only return the questions, no additional explanations.
"""

    try:
        response = llm.invoke([HumanMessage(content=variation_prompt)])
        variations = []
        
        lines = response.content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and any(char.isdigit() for char in line[:3]):  
                clean_line = re.sub(r'^\d+[\.\)\-\s]*', '', line).strip()
                if clean_line:
                    variations.append(clean_line)
        
        if not variations:
            variations = [original_question]
        
        return variations
    except Exception as e:
        st.warning(f"Could not generate query variations: {e}")
        return [original_question]

def multi_query_retrieval(retriever, llm, question, num_variations=3, top_k_per_query=3):
    """
    Perform Multi-Query RAG: generate multiple query variations and combine results
    """
    query_variations = generate_query_variations(llm, question, num_variations)
    
    all_queries = [question] + query_variations
    
    doc_scores = {}
    all_retrieved_docs = []
    
    for i, query in enumerate(all_queries):
        try:
            docs = retriever.invoke(query, k=top_k_per_query)
            for j, doc in enumerate(docs):
                file_name = doc.metadata.get('file_name', 'unknown')
                page_num = doc.metadata.get('page_number', 0)
                chunk_idx = doc.metadata.get('chunk_index', 0)
                doc_key = f"{file_name}_{page_num}_{chunk_idx}"
                
                position_score = (top_k_per_query - j) / top_k_per_query
                query_weight = 1.0 if i == 0 else 0.8  
                
                final_score = position_score * query_weight
                if doc_key in doc_scores:
                    if doc_scores[doc_key]['score'] < final_score:
                        doc_scores[doc_key] = {
                            'score': final_score,
                            'doc': doc,
                            'query': query
                        }
                else:
                    doc_scores[doc_key] = {
                        'score': final_score,
                        'doc': doc,
                        'query': query
                    }
        except Exception as e:
            st.warning(f"Error searching with query '{query}': {e}")
            continue
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    final_docs = [item['doc'] for item in sorted_docs[:st.session_state.get('top_k', 4)]]
    return final_docs, all_queries

def handle_user_question(user_question):
    llm = st.session_state.llm
    retriever = st.session_state.retriever

    history_str = "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in st.session_state.chat_history])
    rephrase_prompt_messages = [
        SystemMessage(content="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
        HumanMessage(content=f"Chat History:\n{history_str}\n\nFollow Up Input: {user_question}\nStandalone question:")
    ]
    
    standalone_question_response = llm.invoke(rephrase_prompt_messages)
    standalone_question = standalone_question_response.content

    with st.spinner("Searching for relevant documents..."):
        if st.session_state.get('use_multi_query', True):
            num_variations = st.session_state.get('num_query_variations', 3)
            top_k_per_query = st.session_state.get('top_k_per_query', 3)
            
            retrieved_docs, query_variations = multi_query_retrieval(
                retriever, llm, standalone_question, 
                num_variations=num_variations,
                top_k_per_query=top_k_per_query
            )
            if len(retrieved_docs) < st.session_state.retrieve_k:
                additional_docs = retriever.invoke(standalone_question, k=st.session_state.retrieve_k - len(retrieved_docs))
                retrieved_docs.extend(additional_docs)
            retrieved_docs = retrieved_docs[:st.session_state.retrieve_k]
        else:
            retrieved_docs = retriever.invoke(standalone_question, k=st.session_state.retrieve_k)

    final_docs_for_llm = []
    if st.session_state.get('use_reranker', True) and retrieved_docs:
        with st.spinner("Reranking documents for higher accuracy..."):
            try:
                cohere_api_key = os.getenv("COHERE_API_KEY")
                if not cohere_api_key:
                    st.error("COHERE_API_KEY not found in environment variables.")
                    final_docs_for_llm = retrieved_docs[:st.session_state.top_n_rerank]
                else:
                    co = cohere.Client(cohere_api_key)
                    
                    doc_texts = [doc.page_content for doc in retrieved_docs]
                    
                    rerank_results = co.rerank(
                        model='rerank-english-v3.0',
                        query=standalone_question,
                        documents=doc_texts,
                        top_n=st.session_state.top_n_rerank
                    )
                    
                    for result in rerank_results.results:
                        final_docs_for_llm.append(retrieved_docs[result.index])

            except Exception as e:
                st.warning(f"Reranking failed: {e}. Falling back to initial retrieval.")
                final_docs_for_llm = retrieved_docs[:st.session_state.top_n_rerank]
    else:
        final_docs_for_llm = retrieved_docs[:st.session_state.top_n_rerank]

    context_parts = []
    source_map = {}
    content_type_stats = {}
    
    for i, doc in enumerate(final_docs_for_llm, 1):
        file_name = doc.metadata.get('file_name', 'Unknown Document')
        page_number = doc.metadata.get('page_number', 'N/A')
        content_type = doc.metadata.get('content_type', 'text')
        
        content_type_stats[content_type] = content_type_stats.get(content_type, 0) + 1
        
        content_type_emoji = {
            'text': '📄',
            'table': '📊', 
            'image': '🖼️'
        }
        
        emoji = content_type_emoji.get(content_type, '📄')
        
        context_parts.append(f"Source [{i}] ({content_type}):\n{doc.page_content}")
        source_map[i] = f"{emoji} {file_name}, page {page_number} ({content_type})"

    context_with_sources = "\n\n".join(context_parts)
    
    system_prompt = """
    You are a professional Q&A assistant. Your task is to answer the user's question ONLY BASED ON the provided context.
    The context includes different types of content: text, tables, and image descriptions.
    Follow these rules STRICTLY:
    1. Carefully analyze the provided context, which includes multiple sources numbered as `Source [1]`, `Source [2]`, etc.
    2. Each source is labeled with its content type (text, table, or image).
    3. Synthesize a direct answer to the user's question using information from all relevant content types.
    4. For EVERY piece of information, claim, or data you provide in your answer, you MUST cite the corresponding source where you obtained that information.
    5. Cite by adding the source label, e.g., `[Source 1]`, at the end of the sentence or clause containing that information.
    6. If information is found in multiple sources, cite all of them, e.g., `[Source 1][Source 3]`.
    7. ABSOLUTELY do not fabricate information. If the answer is not in the context, clearly state that you cannot answer based on the provided documents.
    8. Do not list sources at the end of your answer; all citations must be inline within the text.
    9. When referencing tables or images, mention their content type for clarity.
    """

    answer_prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context_with_sources}\n\nQuestion: {standalone_question}\n\nAnswer:")
    ]

    raw_answer = llm.invoke(answer_prompt_messages).content
    
    cited_indices = re.findall(r'\[Source (\d+)\]', raw_answer)
    
    if not cited_indices:
        citations_text = "\n\n**Sources:**\n" + "\n".join(f"- {src}" for src in source_map.values())
        final_answer = raw_answer + citations_text
    else:
        unique_indices = sorted(list(set(int(i) for i in cited_indices)))

        footnote_mapping = {original_index: new_index for new_index, original_index in enumerate(unique_indices, 1)}

        def replace_with_superscript(match):
            original_index = int(match.group(1))
            if original_index in footnote_mapping:
                new_index = footnote_mapping[original_index]
                superscripts = {
                    '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵',
                    '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '10': '¹⁰'
                }
                return superscripts.get(str(new_index), f"^{new_index}")
            return ""

        formatted_answer = re.sub(r'\[Source (\d+)\]', replace_with_superscript, raw_answer).strip()
        
        citations_list = ["\n\n---", "**Sources:**"]
        for original_index in unique_indices:
            new_index = footnote_mapping[original_index]
            source_info = source_map.get(original_index, "Unknown source")
            
            superscripts = {
                '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵',
                '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '10': '¹⁰'
            }
            super_char = superscripts.get(str(new_index), f"^{new_index}")
            
            citations_list.append(f"{super_char} {source_info}")
            
        final_answer = formatted_answer + "\n" + "\n".join(citations_list)

    if st.session_state.get('show_content_types', False) and content_type_stats:
        type_breakdown = ["\n\n**Content Type Breakdown:**"]
        for content_type, count in content_type_stats.items():
            emoji = {'text': '📄', 'table': '📊', 'image': '🖼️'}.get(content_type, '📄')
            type_breakdown.append(f"{emoji} {content_type.title()}: {count} chunks")
        final_answer += "\n" + "\n".join(type_breakdown)

    if st.session_state.get('show_top_chunks', False) and final_docs_for_llm:
        top_chunks = ["\n\n**Top 5 Most Relevant Content Chunks:**"]
        for i, doc in enumerate(final_docs_for_llm[:5], 1):
            file_name = doc.metadata.get('file_name', 'Unknown Document')
            page_number = doc.metadata.get('page_number', 'N/A')
            content_type = doc.metadata.get('content_type', 'text')
            chunk_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            emoji = {'text': '📄', 'table': '📊', 'image': '🖼️'}.get(content_type, '📄')
            top_chunks.append(f"**{i}.** {emoji} {file_name}, page {page_number} ({content_type})")
            top_chunks.append(f"   *{chunk_preview}*")
        
        final_answer += "\n" + "\n".join(top_chunks)
    
    return final_answer

def user_sidebar():
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.button("Clear Chat History"):
            clear_user_chat_history(st.session_state.username)
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
        
        st.divider()
        st.subheader("Reranker Settings")
        st.session_state.use_reranker = st.checkbox(
            "Enable Cohere Reranker",
            value=st.session_state.get('use_reranker', True),
            help="Use Cohere's reranking API to improve document relevance"
        )
        
        if st.session_state.use_reranker:
            st.session_state.retrieve_k = st.slider(
                "Initial retrieval count:",
                min_value=5,
                max_value=50,
                value=st.session_state.get('retrieve_k', 20),
                step=5,
                help="Number of documents to retrieve before reranking"
            )
            
            st.session_state.top_n_rerank = st.slider(
                "Final reranked documents:",
                min_value=1,
                max_value=10,
                value=st.session_state.get('top_n_rerank', 4),
                step=1,
                help="Number of top documents after reranking to use for answering"
            )
        else:
            st.session_state.top_k = st.slider(
                "Documents to retrieve:",
                min_value=1,
                max_value=10,
                value=st.session_state.get('top_k', 4),
                step=1,
                help="Number of documents to retrieve and use for answering"
            )
        
        st.session_state.show_top_chunks = st.checkbox(
            "Show top 5 relevant chunks",
            value=st.session_state.get('show_top_chunks', False),
            help="Display the top 5 most relevant text chunks after each answer"
        )
        
        st.session_state.show_content_types = st.checkbox(
            "Show content type breakdown",
            value=st.session_state.get('show_content_types', False),
            help="Display the types of content (text, table, image) in search results"
        )
        
        st.divider()
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents (extracting text, tables, and analyzing images)..."):
                    text_chunks, metadatas = get_pdf_chunks(pdf_docs)
                    vectorstore = get_vectorstore(text_chunks, metadatas)
                    st.session_state.llm, st.session_state.retriever = get_conversation_chain(vectorstore)
                
                content_stats = {}
                for metadata in metadatas:
                    content_type = metadata.get('content_type', 'text')
                    content_stats[content_type] = content_stats.get(content_type, 0) + 1
                
                st.success(f"Processing complete! Extracted: {content_stats}")
            else:
                st.warning("Please upload at least one PDF file.")

def login_form():
    st.header("RAG Lab - Please Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        login_button = st.form_submit_button("Login")
        register_button = st.form_submit_button("Register")

        if login_button:
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        if register_button:
            if username and password:
                if register_user(username, password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please enter username and password")

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Lab", page_icon=":books:")
    init_database()

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
        st.session_state.top_k = 4
    if "use_multi_query" not in st.session_state:
        st.session_state.use_multi_query = True
    if "num_query_variations" not in st.session_state:
        st.session_state.num_query_variations = 3
    if "top_k_per_query" not in st.session_state:
        st.session_state.top_k_per_query = 3
    if "use_reranker" not in st.session_state:
        st.session_state.use_reranker = True
    if "retrieve_k" not in st.session_state:
        st.session_state.retrieve_k = 20
    if "top_n_rerank" not in st.session_state:
        st.session_state.top_n_rerank = 4
    if "show_top_chunks" not in st.session_state:
        st.session_state.show_top_chunks = False

    if not st.session_state.authenticated:
        login_form()
        return

    user_sidebar()
    st.header("RAG LAB")

    if not st.session_state.chat_history:
         st.session_state.chat_history = load_chat_history_with_window(st.session_state.username, 5)

    for message in st.session_state.chat_history:
        with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
            st.write(message.content)

    if user_question := st.chat_input("Ask a question about your document..."):
        if st.session_state.retriever:
            with st.chat_message("human"):
                st.write(user_question)

            with st.spinner("Thinking..."):
                ai_answer = handle_user_question(user_question)

            with st.chat_message("ai"):
                st.write(ai_answer)

            save_message_to_db(st.session_state.username, user_question, 'user')
            save_message_to_db(st.session_state.username, ai_answer, 'ai')

            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=ai_answer))
        else:
            st.warning("Please upload and process a PDF document first.")

if __name__ == '__main__':
    main()
import re
import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage


class QueryProcessor:
    def __init__(self):
        pass

    def generate_query_variations(self, llm, original_question, num_variations=3):
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

    def multi_query_retrieval(self, retriever, llm, question, num_variations=3, top_k_per_query=3, top_k=4):
        query_variations = self.generate_query_variations(llm, question, num_variations)
        
        all_queries = [question] + query_variations
        
        doc_scores = {}
        
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
        final_docs = [item['doc'] for item in sorted_docs[:top_k]]
        return final_docs, all_queries

    def handle_user_question(self, user_question, llm, retriever, chat_history, settings):
        history_str = "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history])
        rephrase_prompt_messages = [
            SystemMessage(content="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
            HumanMessage(content=f"Chat History:\n{history_str}\n\nFollow Up Input: {user_question}\nStandalone question:")
        ]
        
        standalone_question_response = llm.invoke(rephrase_prompt_messages)
        standalone_question = standalone_question_response.content
        
        if settings.get('use_multi_query', True):
            num_variations = settings.get('num_query_variations', 3)
            top_k_per_query = settings.get('top_k_per_query', 3)
            top_k = settings.get('top_k', 4)
            
            retrieved_docs, query_variations = self.multi_query_retrieval(
                retriever, llm, standalone_question, 
                num_variations=num_variations,
                top_k_per_query=top_k_per_query,
                top_k=top_k
            )
        else:
            top_k = settings.get('top_k', 4)
            retrieved_docs = retriever.invoke(standalone_question)[:top_k]

        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        answer_prompt_messages = [
            SystemMessage(content="You are a helpful assistant. Answer the following question based only on the provided context."),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {standalone_question}\n\nAnswer:")
        ]

        final_answer_response = llm.invoke(answer_prompt_messages)
        citations = []
        for i, doc in enumerate(retrieved_docs):
            if hasattr(doc, 'metadata') and doc.metadata:
                file_name = doc.metadata.get('file_name', 'Document')
                page_number = doc.metadata.get('page_number', '?')
                citations.append(f"📄 {file_name}, page {page_number}")
            else:
                citations.append(f"📄 Document {i+1}")

        answer = final_answer_response.content
        if citations:
            citation_text = "\n\n**Sources:**\n" + "\n".join(f"- {citation}" for citation in citations)
            answer += citation_text
        
        return answer

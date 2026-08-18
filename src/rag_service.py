import os
import streamlit as st
from langchain_community.llms import Ollama
from src.query_parser import parse_query

@st.cache_resource
def get_llm():
    """Initializes the Ollama LLM."""
    model_name = "qwen2.5:7b"
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    return Ollama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.1
    )

def generate_answer(query, retrieved_docs):
    llm = get_llm()
    if not llm:
        return "LLM not available to answer this query."
        
    context = "\n\n".join([f"Source: {doc.metadata.get('source_file', 'Unknown')} (Section: {doc.metadata.get('section', 'General')})\n{doc.page_content}" for doc in retrieved_docs])
    
    prompt = f"""
    You are an expert Legal Analyst and Contract Manager.
    Answer the following question based ONLY on the provided contract clauses.
    
    CRITICAL RULES:
    1. ZERO HALLUCINATION: If the requested information is not in the context, clearly state: "I cannot find this information in the provided contract clauses." Do not assume standard legal practices.
    2. BE PRECISE: Quote specific terms, dollar amounts, and days of notice exactly as they appear in the text.
    3. NO PREAMBLE: Answer directly. Do not say "Based on the provided context..."
    
    Question: {query}
    
    Context:
    {context}
    
    Answer:
    """
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def contextualize_query(query: str, history: list) -> str:
    llm = get_llm()
    if not llm or not history:
        return query
        
    history_str = ""
    for turn in history:
        ai_resp = turn.get('a', '')[:2000]
        history_str += f"User: {turn['q']}\nAI: {ai_resp}\n\n"
        
    prompt = f"""
    You are an expert at resolving coreferences in conversational search about legal contracts.
    Given a chat history and the latest user query, rewrite the user query to be a standalone query.
    For example, if the history is about "the Microsoft NDA", and the user asks "What is the termination period?", the standalone query should be "What is the termination period for the Microsoft NDA?".
    Do NOT answer the query. Return ONLY the rewritten standalone query.
    
    Chat History:
    {history_str}
    
    Latest User Query: {query}
    
    Standalone Query:
    """
    try:
        response = llm.invoke(prompt).strip()
        if response.startswith('"') and response.endswith('"'): response = response[1:-1]
        if response.lower().startswith("standalone query:"): response = response.split(":", 1)[1].strip()
        return response
    except Exception:
        return query

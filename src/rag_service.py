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
        temperature=0.1,
        num_ctx=8192
    )

def validate_answer(query, answer, retrieved_docs):
    llm = get_llm()
    if not llm:
        return answer
        
    context = "\n\n".join([
        f"--- Chunk {i+1} ---\nDocument: {doc.metadata.get('source_file', 'Unknown')}\nSection: {doc.metadata.get('section', 'General')}\nPage: {doc.metadata.get('page_number', 'Unknown')}\nContent: {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    prompt = f"""
    You are a strict Answer Validator for a Legal RAG system.
    
    Provided Context:
    {context}
    
    Proposed AI Answer:
    {answer}
    
    TASK: Verify if the Proposed AI Answer is supported by the Provided Context.
    Return ONLY a valid JSON object matching this schema:
    {{
      "supported": true/false,
      "should_abstain": true/false,
      "unsupported_claims": ["claim1", "claim2"]
    }}
    """
    try:
        import json
        raw_validated = llm.invoke(prompt).strip()
        start_idx = raw_validated.find('{')
        end_idx = raw_validated.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = raw_validated[start_idx:end_idx]
            val_result = json.loads(json_str)
        else:
            val_result = {"supported": True, "should_abstain": False, "unsupported_claims": []}
            
        if val_result.get("supported"):
            validated_text = answer
        elif not val_result.get("supported") and val_result.get("should_abstain"):
            validated_text = "I cannot find this information in the provided contract clauses."
        else:
            validated_text = "I cannot find this information in the provided contract clauses."

        if "cannot find" in validated_text.lower() or "could not find" in validated_text.lower() or "not provided" in validated_text.lower():
            validated_text += "\n\nSource:\nDocument: N/A\nSection: N/A\nPage: N/A"
        else:
            validated_text += f"\n\nSource:\nSection: {retrieved_docs[0].metadata.get('section', 'Unknown')}\nPage: {retrieved_docs[0].metadata.get('page_number', 'Unknown')}\nDocument: {retrieved_docs[0].metadata.get('source_file', 'Unknown')}"

        return validated_text
    except Exception as e:
        print(f"Validator Error: {e}")
        return answer

def generate_answer(query, retrieved_docs):
    llm = get_llm()
    if not llm:
        return "LLM not available to answer this query."
        
    # Context Budget Manager
    MAX_CONTEXT_TOKENS = 6000
    SYSTEM_PROMPT_TOKENS = 250
    QUESTION_TOKENS = len(query) // 4
    RESERVED_OUTPUT = 1000
    
    available_budget = MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS - QUESTION_TOKENS - RESERVED_OUTPUT
    
    selected_chunks = []
    current_tokens = 0
    
    print("\n" + "="*50)
    print("CONTEXT BUILDER BUDGET LOG")
    print("="*50)
    
    for i, doc in enumerate(retrieved_docs):
        chunk_text = f"Source: {doc.metadata.get('source_file', 'Unknown')} (Section: {doc.metadata.get('section', 'General')})\n{doc.page_content}\n\n"
        chunk_tokens = len(chunk_text) // 4
        
        if current_tokens + chunk_tokens <= available_budget:
            selected_chunks.append(doc)
            current_tokens += chunk_tokens
            
            print(f"Rank {i+1}")
            print(f"Section: {doc.metadata.get('section', 'Unknown')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page_number', 'Unknown')}")
            print(f"Reranker Score: {doc.metadata.get('rerank_score', 'N/A')}")
            print(f"Tokens: {chunk_tokens}")
            print("-" * 20)
        else:
            break
            
    total_estimated = SYSTEM_PROMPT_TOKENS + QUESTION_TOKENS + current_tokens + RESERVED_OUTPUT
    print(f"Context statistics:")
    print(f"Retrieved chunks: {len(retrieved_docs)}")
    print(f"Selected chunks: {len(selected_chunks)}")
    print(f"System prompt tokens: {SYSTEM_PROMPT_TOKENS}")
    print(f"User question tokens: {QUESTION_TOKENS}")
    print(f"Context tokens: {current_tokens}")
    print(f"Reserved output tokens: {RESERVED_OUTPUT}")
    print(f"Total estimated tokens: {total_estimated}")
    print(f"Configured Ollama context: 8192")
    print("="*50 + "\n")
    
    context = "\n\n".join([f"Source: {doc.metadata.get('source_file', 'Unknown')} (Section: {doc.metadata.get('section', 'General')})\n{doc.page_content}" for doc in selected_chunks])
    
    prompt = f"""
    You are an expert Legal Analyst and Contract Manager.
    Answer the following question based ONLY on the provided contract clauses.
    
    CRITICAL RULES:
    1. ZERO HALLUCINATION: If the requested information is not in the context, clearly state: "I cannot find this information in the provided contract clauses." Do not assume standard legal practices.
    2. BE PRECISE: Quote specific terms, dollar amounts, and days of notice exactly as they appear in the text.
    3. DO NOT INVENT CITATIONS: Do not mention clause numbers or section names unless they are explicitly written in the text. The system will automatically attach the official source metadata.
    4. NO PREAMBLE: Answer directly. Do not say "Based on the provided context..."
    5. ALLOWED INTERPRETATIONS:
       - extracting an answer explicitly stated in the retrieved text
       - paraphrasing explicit contract language
       - mapping legal wording to the user's terminology
       - interpreting "shall not exceed" as a liability cap
       - interpreting "Except for..." as exclusions/carve-outs
    6. NOT ALLOWED:
       - adding facts not present in the context
       - assuming standard legal practice
       - inventing dollar values, dates, parties, obligations, or clauses
       - using external legal knowledge
    
    Use semantic equivalence when answering. The exact words used in the user's question do not need to appear in the contract. If the contract explicitly expresses the same legal concept using different language, answer using that evidence.

    EXAMPLES:

    QUESTION: What is the liability cap?
    CONTEXT: Each party's aggregate liability shall not exceed the total fees paid or payable during the preceding twelve months.
    CORRECT: Each party's aggregate liability is capped at the total fees paid or payable during the preceding twelve months.

    QUESTION: Which liabilities are excluded from the liability cap?
    CONTEXT: Except for liability arising from fraud, willful misconduct, breach of confidentiality, IP infringement, or indemnification...
    CORRECT: The cap does not apply to fraud, willful misconduct, breach of confidentiality, IP infringement/misappropriation, or indemnification obligations.

    QUESTION: What is the Provider's annual revenue?
    CONTEXT: No clause contains Provider annual revenue.
    CORRECT: I cannot find this information in the provided contract clauses.
    
    Question: {query}
    
    Context:
    {context}
    
    Answer:
    """
    
    try:
        response = llm.invoke(prompt)
        validated_response = validate_answer(query, response, selected_chunks)
        return validated_response
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
    {context}
    
    Answer:
    """
    
    try:
        response = llm.invoke(prompt)
        validated_response = validate_answer(query, response, selected_chunks)
        return validated_response
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

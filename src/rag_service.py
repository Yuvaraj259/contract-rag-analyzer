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

def check_answerability(query, context):
    llm = get_llm()
    prompt = f"""
    You are an Answerability Gate for a Legal RAG system.
    Review the following Context and determine if it explicitly contains the answer to the Query.
    
    Context:
    {context}
    
    Query: {query}
    
    Return ONLY a JSON object with this exact schema:
    {{
      "status": "SUPPORTED" or "PARTIALLY_SUPPORTED" or "NOT_FOUND" or "CONFLICTING"
    }}
    """
    try:
        import json
        raw = llm.invoke(prompt).strip()
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(raw[start:end]).get("status", "SUPPORTED")
    except Exception:
        pass
    return "SUPPORTED"

def validate_answer(query, answer, retrieved_docs, metadata_summary=""):
    llm = get_llm()
    if not llm:
        return answer
        
    context = "\n\n".join([
        f"--- Chunk {i+1} ---\nDocument: {doc.metadata.get('source_file', 'Unknown')}\nSection: {doc.metadata.get('section', 'General')}\nPage: {doc.metadata.get('page_number', 'Unknown')}\nLine: {doc.metadata.get('line_number', 'Unknown')}\nContent: {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    prompt = f"""
    You are a strict Answer Validator for a Legal RAG system.
    
    Provided Context:
    {context}
    
    Proposed AI Answer:
    {answer}
    
    TASK: Verify if the Proposed AI Answer is supported by the Provided Context.
    
    RULES:
    1. The Context TEXT is the ultimate ground truth. If the metadata is missing but the text contains the answer, the answer is SUPPORTED.
    2. If the answer accurately reflects facts found in the context text, mark "supported": true.
    3. Do not reject an answer just because it includes a common abbreviation or slight rephrasing of the context.
    4. ARITHMETIC EXCEPTION: If the answer includes a mathematical calculation (like a total sum) that is correctly derived from numbers explicitly present in the Context, it IS supported. Do not flag correct arithmetic as an unsupported claim.
    5. If the answer relies on factual claims completely missing from the context text, mark "supported": false and list the unsupported claims.
    
    Return ONLY a valid JSON object matching this schema:
    {{
      "supported": true/false,
      "unsupported_claims": ["claim1"]
    }}
    """
    try:
        import json
        raw_validated = llm.invoke(prompt).strip()
        start_idx = raw_validated.find('{')
        end_idx = raw_validated.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            val_result = json.loads(raw_validated[start_idx:end_idx])
        else:
            val_result = {"supported": True, "unsupported_claims": []}
            
        if not val_result.get("supported", True):
            unsupported = val_result.get("unsupported_claims", [])
            if unsupported:
                print("\n**System Warning:** The following claims could not be verified by the retrieved text: " + ", ".join(unsupported))
        return answer
    except Exception as e:
        print(f"Validator Error: {e}")
        return answer

def generate_answer(query, retrieved_docs):
    llm = get_llm()
    if not llm:
        return "LLM not available to answer this query."
        
    # Build Context
    context_chunks = []
    for doc in retrieved_docs:
        line_num_str = str(doc.metadata.get('line_number', 'Unknown'))
        numbered_text = doc.page_content
        
        if line_num_str != 'Unknown':
            try:
                start_line = int(line_num_str.split('-')[0])
                lines = doc.page_content.split('\n')
                numbered_lines = [f"[Line {start_line + i}] {line}" for i, line in enumerate(lines)]
                numbered_text = '\n'.join(numbered_lines)
            except ValueError:
                pass
                
        chunk_text = f"Source: Document: {doc.metadata.get('source_file', 'Unknown')}, Document Page: {doc.metadata.get('document_page_number', 'Unknown')}, Section: {doc.metadata.get('section', 'Unknown')}\n{numbered_text}\n"
        context_chunks.append(chunk_text)
        
    context = "\n\n".join(context_chunks)
    
    # Pre-generation Answerability Gate removed to prevent false negatives on large contexts
        
    metadata_summary = ""
    seen_docs = set()
    for doc in retrieved_docs:
        source_file = doc.metadata.get('source_file', 'Unknown')
        if source_file not in seen_docs and source_file != 'Unknown':
            seen_docs.add(source_file)
            parties_str = ", ".join(doc.metadata.get('parties', []))
            metadata_summary += f"Metadata for {source_file}: Title: {doc.metadata.get('contract_title', 'Unknown')}, Date: {doc.metadata.get('effective_date', 'Unknown')}, Parties: {parties_str}\n"

    prompt = f"""You are an expert Legal Analyst. Answer the following question based ONLY on the provided contract context.
    
    {metadata_summary}
    Context:
    {context}
    
    Question: {query}
    
    CRITICAL RULES:
    1. THE TEXT IS AUTHORITATIVE: The Metadata provided is just a brief summary. The actual Context text is the ultimate ground truth.
    2. ZERO HALLUCINATION: Do not invent facts, names, dates, amounts, or clauses.
    3. EXACT CITATIONS: The Provided Context has line numbers in brackets at the start of each line, like [Line 77]. Use these to determine EXACTLY which lines the answer came from.
    4. CITATION FORMAT: If the context contains the answer, you MUST append the exact citation to the VERY END of your answer on a new line. Do not put it in the middle. Use THIS EXACT format:
    Source: Document: [file], Document Page: [y], Lines: [start-end], Section: [z]
    5. NO PREAMBLE: Answer directly. DO NOT repeat the question.
    6. MISSING INFO & PARTIAL ANSWERS: If the exact answer is missing, you MUST still report any highly relevant related information found in the text.
    7. SPECIFICITY OVER GENERALITIES: When asked for features, scope, deliverables, or project phases, extract the EXACT bullet points, technical lists, or specific features (e.g. POS integration, Google Maps API). Do not summarize with vague boilerplate language if specific details are present.
    8. DEFINITION BOUNDARIES: When defining a legal term (e.g. "Commercialization", "Change of Control"), strictly adhere to its inclusions and exclusions as stated in the text. Do not contradict yourself by stating an activity is both included and excluded.
    9. TIMELINE & MILESTONE ACCURACY: Never invent, guess, or shift milestone weeks, dates, or payment terms. Only output the exact timing and amounts literally written in the text. Do not infer "Weeks 7-13" if the text does not explicitly group them that way.
    10. EXACT DEFINITIONS & EXCLUSIONS: If the text contains exclusions (e.g., "excluding X") or distinctions (e.g., distinguishing "Deliverables" from "Work Product"), you MUST explicitly mention them in your answer.
    11. FOCUS: Only answer the specific components requested. Do not mix unrelated topics.
    12. ARITHMETIC EXCEPTION: You are explicitly allowed to perform basic mathematical calculations (like addition) if the user asks for a 'total' or 'combined amount' and the individual numbers are explicitly listed in the text.
    
    Output Format Example 1 (Answer Found):
    Item A is USD 25,000 and Item B is USD 25,000.
    Total: 25,000 + 25,000 = USD 50,000.
    The total cost of the project is USD 50,000.
    
    Source: Document: [Actual Filename], Document Page: 2, Lines: 45-46, Section: FEES

    Output Format Example 2 (Answer Not Found):
    I cannot find this information in the provided contract.
    
    Answer:"""
    
    try:
        with open("data/debug_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            
        response = llm.invoke(prompt).strip()
        
        # Clean up LLM hallucinations where it refuses to answer but still appends a citation
        if "I cannot find this information" in response:
            response = "I cannot find this information in the provided contract."
            
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
    CRITICAL INSTRUCTION: You MUST explicitly include the specific company names (e.g. VAL, Qualigen, ExxonMobil) and the specific Document Name/Type from the chat history into the standalone query! 
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

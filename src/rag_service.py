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
    
    Provided Context and Metadata:
    {metadata_summary}
    
    {context}
    
    Proposed AI Answer:
    {answer}
    
    TASK: Verify if the Proposed AI Answer is supported by the Provided Context and Metadata.
    
    RULES:
    1. If the answer accurately reflects facts found in the context or metadata, mark "supported": true.
    2. Do not reject an answer just because it includes a common abbreviation (like LLC, Inc., or an acronym) or slight rephrasing of the context.
    3. If the answer relies on information completely missing from both the context and metadata, mark "supported": false.
    4. If "supported" is false and the answer is attempting to provide factual information not present, set "should_abstain": true.
    5. Focus on verifying if the entities mentioned exist in the contract as parties. Do not reject answers based on semantic technicalities (e.g., calling an individual a 'company').
    
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
        print("====== VALIDATOR PROMPT ======")
        print(prompt)
        print("====== VALIDATOR RAW RESPONSE ======")
        print(raw_validated)
        print("====================================")
        start_idx = raw_validated.find('{')
        end_idx = raw_validated.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = raw_validated[start_idx:end_idx]
            val_result = json.loads(json_str)
        else:
            val_result = {"supported": True, "should_abstain": False, "unsupported_claims": []}
            
        if val_result.get("supported"):
            validated_text = answer
        if not val_result.get("supported", True):
            # We will no longer hard-block the answer. If the validator is unsure, we just append a warning.
            # This prevents false-positive rejections of perfectly valid answers.
            warning = "\n\n**System Warning:** The following claims could not be firmly verified by the strict validator: " + ", ".join(val_result.get("unsupported_claims", []))
            
            # Append source citation
            first_doc = retrieved_docs[0] if retrieved_docs else None
            if first_doc:
                return answer + warning + f"\n\nSource:\nSection: {first_doc.metadata.get('section', 'Unknown')}\nPage: {first_doc.metadata.get('page_number', 'Unknown')}\nLine: {first_doc.metadata.get('line_number', 'Unknown')}\nDocument: {first_doc.metadata.get('source_file', 'Unknown')}"
            return answer + warning
        
        # Default append source
        if "| Question |" in answer or "|---|---|" in answer:
            # If the answer is a table (multi-question), the LLM provides citations in the table itself.
            validated_text = answer
        else:
            validated_text = answer + f"\n\nSource:\nSection: {retrieved_docs[0].metadata.get('section', 'Unknown')}\nPage: {retrieved_docs[0].metadata.get('page_number', 'Unknown')}\nLine: {retrieved_docs[0].metadata.get('line_number', 'Unknown')}\nDocument: {retrieved_docs[0].metadata.get('source_file', 'Unknown')}"

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
        chunk_text = f"Source: {doc.metadata.get('source_file', 'Unknown')} (Section: {doc.metadata.get('section', 'General')}, Page: {doc.metadata.get('page_number', 'Unknown')}, Line: {doc.metadata.get('line_number', 'Unknown')})\n{doc.page_content}\n\n"
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
    
    # Build a Metadata Summary for all unique documents retrieved
    metadata_summary = ""
    seen_docs = set()
    for chunk in selected_chunks:
        m = chunk.metadata
        source_file = m.get('source_file', 'Unknown')
        if source_file not in seen_docs and source_file != 'Unknown':
            seen_docs.add(source_file)
            parties = m.get('parties', [])
            parties_str = ", ".join(parties) if isinstance(parties, list) else str(parties)
            
            metadata_summary += f"""
    --- CONTRACT METADATA CHEAT SHEET FOR {source_file} ---
    Title: {m.get('contract_title', 'Unknown')}
    Type: {m.get('contract_type', 'Unknown')}
    Effective Date: {m.get('effective_date', 'Unknown')}
    Parties Involved: {parties_str}
    -------------------------------------------------------
    """
    
    context = "\n\n".join([f"Source: {doc.metadata.get('source_file', 'Unknown')} (Section: {doc.metadata.get('section', 'General')}, Page: {doc.metadata.get('page_number', 'Unknown')}, Line: {doc.metadata.get('line_number', 'Unknown')})\n{doc.page_content}" for doc in selected_chunks])
    
    prompt = f"""
    You are an expert Legal Analyst and Contract Manager.
    Answer the following question based ONLY on the provided contract clauses and metadata.
    
    CRITICAL RULES:
    1. ZERO HALLUCINATION: If the requested information is not in the context or metadata, clearly state: "I cannot find this information in the provided contract clauses." Do not assume standard legal practices.
    2. BE PRECISE: Quote specific terms, dollar amounts, and days of notice exactly as they appear in the text.
    3. DO NOT INVENT CITATIONS: Do not mention clause numbers or section names unless they are explicitly written in the text. The system will automatically attach the official source metadata.
    4. NO PREAMBLE: Answer directly. Do not say "Based on the provided context..."
    5. ALLOWED INTERPRETATIONS:
       - extracting an answer explicitly stated in the retrieved text OR the Metadata Cheat Sheet
       - paraphrasing explicit contract language
       - mapping legal wording to the user's terminology
       - interpreting "shall not exceed" as a liability cap
       - interpreting "Except for..." as exclusions/carve-outs
    6. NOT ALLOWED:
       - adding facts not present in the context
       - assuming standard legal practice
       - inventing dollar values, dates, parties, obligations, or clauses
       - using external legal knowledge
    7. NO LOGICAL LEAPS: If the contract states X happens under condition Y, DO NOT assume the opposite happens under other conditions unless explicitly stated. For example, if it says "If A terminates, B gets paid", do not assume "If B terminates, B gets nothing" unless it is explicitly written.
    8. BE DETAILED: Provide comprehensive, nuanced legal answers. If there are multiple conditions, caveats, or exceptions, list them clearly.
    9. USE EXACT QUOTES: Whenever possible, provide the exact wording directly from the contract wrapped in quotation marks. Do not summarize or paraphrase if the original text provides a more accurate answer. Rely heavily on extractive question-answering.
    10. MANDATORY CITATIONS: For every single answer you provide (whether in a paragraph or a table), you MUST explicitly state the Document, Page, and Line number where you found the information.

    Use semantic equivalence when searching for answers, but when presenting the answer, stick closely to the original contract language.
    
    IMPORTANT FORMATTING RULE: 
    If the user's prompt contains multiple questions, you MUST format your final response as a clean Markdown table with exactly three columns: "Question", "Answer", and "Source Citation". In the "Source Citation" column, provide the Document, Section, Page, and Line number where you found the answer based on the provided Context. Do not output anything outside of the table.
    
    EXAMPLE TABLE OUTPUT (for multiple questions):
    | Question | Answer | Source Citation |
    |---|---|---|
    | What is the liability cap? | The liability is capped at the total fees paid during the preceding 12 months. | Document: 1.pdf, Section: LIABILITY, Page: 4, Line: 152 |
    | What is the governing law? | The agreement is governed by the laws of California. | Document: 1.pdf, Section: MISCELLANEOUS, Page: 9, Line: 310 |
    
    EXAMPLE REGULAR OUTPUT (for a single question):
    The liability is capped at the total fees paid during the preceding 12 months (Document: 1.pdf, Page: 4, Line: 152).
    
    Question: {query}
    
    Context:
    {metadata_summary}
    
    {context}
    
    Answer:
    """
    
    try:
        response = llm.invoke(prompt)
        validated_response = validate_answer(query, response, selected_chunks, metadata_summary)
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

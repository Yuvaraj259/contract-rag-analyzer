import json
import re

def detect_operator(query: str) -> str:
    # Only enforce strict boolean logic if the user explicitly capitalizes it
    if " OR " in query:
        return "OR"
    if " AND " in query:
        return "AND"
    return "OR" # Default to OR for better BM25 recall on natural language queries

def classify_query_intent(query: str, llm=None) -> str:
    """
    Uses the LLM to classify the user's query into a specific intent category for contracts.
    """
    if llm is None:
        return "general_search"
        
    router_prompt = f"""
    You are an expert Legal Tech Query Router. Categorize the user's query into exactly one of the following intents:
    - clause_search (e.g., "Find the SLA penalty clause", "What is the limitation of liability?")
    - contract_search (e.g., "Find the Microsoft contract")
    - contract_summary (e.g., "Summarize this agreement", "What are the main points?")
    - risk_analysis (e.g., "Are there any high-risk data privacy issues?")
    - general_search (e.g., any other query)
    
    Query: "{query}"
    
    Return ONLY the category name as a plain string. Do not explain.
    """
    
    try:
        category = llm.invoke(router_prompt).strip().lower()
        valid_categories = ["clause_search", "contract_search", "contract_summary", "risk_analysis", "general_search"]
        for v in valid_categories:
            if v in category:
                return v
        return "general_search"
    except Exception as e:
        print(f"Router failed: {e}")
        return "general_search"

import functools

@functools.lru_cache(maxsize=128)
def extract_search_terms(query: str) -> list:
    """
    Extracts the core search entities from a natural language query.
    """
    try:
        from src.rag_service import get_llm
        llm = get_llm()
        if not llm:
            return [query]
            
        prompt = f"""
    Extract the core legal clauses, terms, or specific entities the user is searching for in this query.
    Return ONLY a JSON list of strings. Do not explain.
    
    Query: "{query}"
    
    Example Output for "Find the SLA penalty and termination clause for Microsoft":
    ["SLA penalty", "termination clause", "Microsoft"]
    """
        resp = llm.invoke(prompt).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.endswith("```"): resp = resp[:-3]
        import json
        extracted = json.loads(resp.strip())
        if isinstance(extracted, list):
            return extracted
    except Exception as e:
        print(f"Extract terms failed: {e}")
    return [query]

def parse_query(query: str, llm=None) -> dict:
    query_type = classify_query_intent(query, llm)
    search_terms = extract_search_terms(query)
    
    return {
        "query_type": query_type,
        "operator": detect_operator(query),
        "search_terms": search_terms
    }

def decompose_query(query: str, llm=None) -> list:
    if not llm:
        return [q.strip() + "?" for q in re.split(r'\?', query) if q.strip()]
    
    prompt = f"""
    You are an expert at breaking down complex legal queries into individual, standalone sub-questions.
    If the query asks multiple independent questions, output each as a separate string in a JSON list.
    If it is a single question, output a list with one string.
    CRITICAL RULE 1: Do NOT split single unified legal concepts (e.g., 'Applicable Laws and Regulations', 'Term and Termination') into multiple questions.
    CRITICAL RULE 2: If the user's query mentions a specific document name, contract title, or context (e.g., 'in the RECIPE DEVELOPMENT AGREEMENT' or 'from document X'), you MUST preserve this document name EXACTLY as written in EVERY sub-question you generate. Do NOT strip it out.
    DO NOT explain. Output ONLY a valid JSON list of strings.
    
    Query: "{query}"
    """
    try:
        resp = llm.invoke(prompt).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.endswith("```"): resp = resp[:-3]
        queries = json.loads(resp.strip())
        if isinstance(queries, list) and len(queries) > 0:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    deduped.append(q)
            return deduped
    except Exception:
        pass
    
    return [q.strip() + "?" for q in re.split(r'\?', query) if q.strip()]

import functools

@functools.lru_cache(maxsize=128)
def expand_query_aliases(query: str) -> str:
    aliases = {
        "effective date": '"effective date" OR "made and effective" OR "commencement date" OR "entered into" OR "executed as of"',
        "parties": '"parties" OR "between" OR "and" OR "party" OR "developer" OR "client" OR "company"',
        "agreement number": '"agreement number" OR "agreement no" OR "contract number" OR "contract no"',
        "duration": '"agreement period" OR "term" OR "maximum period" OR "duration" OR "term of agreement"',
        "quotation": '"quotation" OR "project cost" OR "total price" OR "contract value" OR "commercial" OR "estimation and commercials"',
        "delivery": '"delivery time" OR "total delivery time" OR "project timeline" OR "completion" OR "handover" OR "milestone"',
        "platform": '"platform" OR "iOS" OR "Android" OR "target device" OR "operating system"',
        "scope": '"scope of work" OR "scope" OR "deliverables" OR "requirements" OR "services"',
        "purpose": '"purpose" OR "recitals" OR "engage" OR "services" OR "project"',
        "amount": '"amount payable" OR "paid" OR "payments" OR "fees" OR "compensation"',
        "payable": '"amount payable" OR "paid" OR "payments" OR "fees" OR "compensation"'
    }
    
    query_lower = query.lower()
    for key, alias_str in aliases.items():
        if key in query_lower:
            return f"({query}) OR ({alias_str})"
            
    # Dynamic LLM Expansion
    try:
        from src.rag_service import get_llm
        llm = get_llm()
        if llm:
            prompt = f"""You are a legal terminology expert. Provide 3 to 5 single-word legal synonyms or related terms for the following query. Focus on contractual terms.
Query: "{query}"
Rules:
1. Output ONLY a comma-separated list of words.
2. NO quotes, NO parentheses, NO explanations.
Example output: warrant, represent, covenant, obligation
"""
            resp = llm.invoke(prompt).strip()
            
            # Clean response to prevent Elasticsearch syntax errors
            import re
            clean_resp = re.sub(r'[^a-zA-Z0-9,\s]', '', resp)
            synonyms = [s.strip() for s in clean_resp.split(',') if s.strip()]
            
            if synonyms:
                syn_str = " OR ".join(synonyms)
                return f"({query}) OR ({syn_str})"
    except Exception as e:
        print(f"LLM Query expansion failed: {e}")
        
    return query


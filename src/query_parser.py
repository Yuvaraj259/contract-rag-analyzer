import json
import re

def detect_operator(query: str) -> str:
    q = query.lower()
    if " or " in q:
        return "OR"
    if " and " in q or " both " in q:
        return "AND"
    return "AND" # Default to AND for contract searches

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

def extract_search_terms(query: str, llm=None) -> list:
    """Extracts key legal terms or clauses from the query using the LLM."""
    if llm is None:
        return [query]
        
    ner_prompt = f"""
    Extract the core legal clauses, terms, or specific entities the user is searching for in this query.
    Return ONLY a JSON list of strings. Do not explain.
    
    Query: "{query}"
    
    Example Output for "Find the SLA penalty and termination clause for Microsoft":
    ["SLA penalty", "termination clause", "Microsoft"]
    """
    try:
        resp = llm.invoke(ner_prompt).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.endswith("```"): resp = resp[:-3]
        extracted = json.loads(resp.strip())
        if isinstance(extracted, list):
            return extracted
    except Exception:
        pass
    return [query]

def parse_query(query: str, llm=None) -> dict:
    query_type = classify_query_intent(query, llm)
    search_terms = extract_search_terms(query, llm)
    
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
    DO NOT explain. Output ONLY a valid JSON list of strings.
    
    Query: "{query}"
    """
    try:
        resp = llm.invoke(prompt).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.endswith("```"): resp = resp[:-3]
        queries = json.loads(resp.strip())
        if isinstance(queries, list) and len(queries) > 0:
            return queries
    except Exception:
        pass
    
    return [q.strip() + "?" for q in re.split(r'\?', query) if q.strip()]

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
    return query


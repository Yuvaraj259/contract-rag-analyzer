import json
import os
from datetime import datetime

ANALYTICS_FILE = "data/search_analytics.jsonl"

def log_search(query: str, query_type: str, search_terms: list, total_results: int = 0):
    """
    Logs the contract search analytics to a JSONL file.
    """
    os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "query_type": query_type,
        "search_terms": search_terms,
        "total_results": total_results
    }
    
    try:
        with open(ANALYTICS_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging search analytics: {e}")

import re

def retrieve_context(query, vector_store, k=5, filter_dict=None):
    """
    Legacy retrieval function (keeps existing behavior if needed).
    """
    if not vector_store:
        return []
    
    sub_queries = [q.strip() for q in re.split(r'\?|\band\b', query) if q.strip()]
    if not sub_queries:
        sub_queries = [query]
        
    all_docs = []
    seen_content = set()
    k_per_query = max(1, k // len(sub_queries))
    
    for sub_q in sub_queries:
        # Build Langchain ES filter list
        es_filter = []
        if filter_dict:
            for key, val in filter_dict.items():
                if isinstance(val, list):
                    es_filter.append({"terms": {f"metadata.{key}.keyword": val}})
                else:
                    es_filter.append({"term": {f"metadata.{key}.keyword": val}})
            
        # 1. Vector Search (Filter pushed to database!)
        docs_vector = vector_store.similarity_search(sub_q, k=k_per_query, filter=es_filter if es_filter else None)
        
        # 2. BM25 Keyword Search (Manual)
        docs_keyword = []
        try:
            from src.vector_store import INDEX_NAME
            es_client = vector_store.client
            
            # Base multi_match query
            bm25_query = {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": sub_q,
                                "fields": ["text", "metadata.contract_title^5", "metadata.source_file"]
                            }
                        }
                    ]
                }
            }
            
            # Add wildcard metadata fallback for bad extractions (e.g. underscores in filenames)
            wildcard_q = " OR ".join([f"*{w}*" for w in sub_q.split() if len(w) > 3])
            if wildcard_q:
                bm25_query["bool"]["should"].append({
                    "query_string": {
                        "query": wildcard_q,
                        "fields": ["metadata.contract_title^50", "metadata.source_file^50"]
                    }
                })
            
            # Apply filter to manual BM25 query if needed
            if filter_dict:
                body = {
                    "query": {
                        "bool": {
                            "must": [bm25_query],
                            "filter": [{"terms" if isinstance(v, list) else "term": {f"metadata.{k}.keyword": v}} for k, v in filter_dict.items()]
                        }
                    },
                    "size": k_per_query
                }
            else:
                body = {
                    "query": bm25_query,
                    "size": k_per_query * 3
                }
                
            res = es_client.search(index=INDEX_NAME, body=body)
            from langchain_core.documents import Document
            for hit in res.get("hits", {}).get("hits", []):
                docs_keyword.append(Document(
                    page_content=hit["_source"].get("text", ""),
                    metadata=hit["_source"].get("metadata", {})
                ))
        except Exception as e:
            print(f"BM25 fallback failed: {e}")

        # Combine and deduplicate (Put Keyword matches first so they aren't truncated by the vector limit!)
        for doc in docs_keyword + docs_vector:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_docs.append(doc)
                
    # Trim down to requested K after filtering
    return all_docs[:k]



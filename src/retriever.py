import re
import streamlit as st

def fetch_neighbors(docs, vector_store):
    if not docs:
        return []
    
    try:
        es_client = vector_store.client
        from src.vector_store import INDEX_NAME
        from langchain_core.documents import Document
        
        expanded_docs = []
        seen_ids = set()
        search_queries = []
        
        for doc in docs:
            source_file = doc.metadata.get("source_file")
            chunk_index = doc.metadata.get("chunk_index")
            
            if source_file and chunk_index is not None:
                chunk_id_curr = f"{source_file}_{chunk_index}"
                if chunk_id_curr not in seen_ids:
                    expanded_docs.append(doc)
                    seen_ids.add(chunk_id_curr)
                
                for offset in [-1, 1]:
                    neighbor_idx = chunk_index + offset
                    if neighbor_idx >= 0:
                        neighbor_id = f"{source_file}_{neighbor_idx}"
                        if neighbor_id not in seen_ids:
                            seen_ids.add(neighbor_id)
                            search_queries.append({
                                "bool": {
                                    "must": [
                                        {"term": {"metadata.source_file.keyword": source_file}},
                                        {"term": {"metadata.chunk_index": neighbor_idx}}
                                    ]
                                }
                            })
            else:
                expanded_docs.append(doc)
                
        if search_queries:
            body = {
                "query": {
                    "bool": {
                        "should": search_queries
                    }
                },
                "size": len(search_queries)
            }
            res = es_client.search(index=INDEX_NAME, body=body)
            for hit in res.get("hits", {}).get("hits", []):
                expanded_docs.append(Document(
                    page_content=hit["_source"].get("text", ""),
                    metadata=hit["_source"].get("metadata", {})
                ))
                
        # Sort so that contiguous chunks appear in order
        def sort_key(d):
            return (d.metadata.get("source_file", ""), d.metadata.get("chunk_index", 0))
            
        expanded_docs.sort(key=sort_key)
        return expanded_docs
        
    except Exception as e:
        print(f"Failed to fetch neighbors: {e}")
        return docs


@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_docs(query, docs, top_n=5):
    if not docs:
        return []
    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        
        q_lower = query.lower()
        boosted_scores = []
        for doc, base_score in zip(docs, scores):
            score = base_score
            sec = doc.metadata.get("section", "").lower()
            if sec:
                # Exact match
                if sec in q_lower:
                    score += 5.0
                # Semantic match fallbacks
                elif "scope" in q_lower and "scope" in sec:
                    score += 5.0
                elif ("price" in q_lower or "quotation" in q_lower or "cost" in q_lower or "amount" in q_lower) and ("fee" in sec or "payment" in sec or "commercial" in sec or "pricing" in sec):
                    score += 5.0
                elif ("parties" in q_lower or "between" in q_lower) and ("intro" in sec or "parties" in sec or "between" in sec):
                    score += 5.0
                elif ("duration" in q_lower or "term" in q_lower) and ("term" in sec or "period" in sec):
                    score += 5.0
                elif ("delivery" in q_lower or "milestone" in q_lower) and ("milestone" in sec or "delivery" in sec):
                    score += 5.0
                    
            boosted_scores.append((doc, score))
            
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in boosted_scores[:top_n]]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return docs[:top_n]

def retrieve_context(query, vector_store, k=5, filter_dict=None, enable_reranking=True):
    """
    Retrieves context for a single decomposed query, using both semantic search and BM25 fallback.
    """
    if not vector_store:
        return []
    
    from src.query_parser import expand_query_aliases
    
    # We now assume the query passed here is a single sub-question.
    all_docs = []
    seen_content = set()
    
    # 1. Vector Search (Filter pushed to database!)
    es_filter = []
    if filter_dict:
        for key, val in filter_dict.items():
            if isinstance(val, list):
                es_filter.append({"terms": {f"metadata.{key}.keyword": val}})
            else:
                es_filter.append({"term": {f"metadata.{key}.keyword": val}})
                
    docs_vector = vector_store.similarity_search(query, k=k, filter=es_filter if es_filter else None)
    
    # 2. BM25 Keyword Search (Manual)
    docs_keyword = []
    try:
        from src.vector_store import INDEX_NAME
        es_client = vector_store.client
        
        # Expand query for BM25 lexical match
        expanded_query = expand_query_aliases(query)
        
        # Determine Section Boosts
        q_lower = query.lower()
        section_boosts = []
        if "scope" in q_lower:
            section_boosts.append({"match": {"metadata.section": {"query": "SCOPE OF WORK", "boost": 3}}})
        if "duration" in q_lower or "period" in q_lower or "term" in q_lower:
            section_boosts.append({"match": {"metadata.section": {"query": "TERM", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "TERM OF AGREEMENT", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "TERM AND TERMINATION", "boost": 3}}})
        if "quotation" in q_lower or "price" in q_lower or "cost" in q_lower or "amount" in q_lower or "payable" in q_lower:
            section_boosts.append({"match": {"metadata.section": {"query": "PAYMENT TERMS", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "PAYMENT", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "FEES", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "ESTIMATION AND COMMERCIALS", "boost": 3}}})
        if "parties" in q_lower or "between" in q_lower:
            section_boosts.append({"match": {"metadata.section": {"query": "INTRO", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "PARTIES", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "BETWEEN:", "boost": 3}}})
        if "delivery" in q_lower or "time" in q_lower:
            section_boosts.append({"match": {"metadata.section": {"query": "MILESTONE", "boost": 3}}})
            section_boosts.append({"match": {"metadata.section": {"query": "DELIVERY", "boost": 3}}})
        
        # Base multi_match query
        bm25_query = {
            "bool": {
                "should": [
                    {
                        "query_string": {
                            "query": expanded_query,
                            "fields": ["text^2", "metadata.contract_title^5", "metadata.source_file", "metadata.section^3"]
                        }
                    }
                ] + section_boosts
            }
        }
        
        # Apply filter to manual BM25 query if needed
        if filter_dict:
            body = {
                "query": {
                    "bool": {
                        "must": [bm25_query],
                        "filter": [{"terms" if isinstance(v, list) else "term": {f"metadata.{k}.keyword": v}} for k, v in filter_dict.items()]
                    }
                },
                "size": k
            }
        else:
            body = {
                "query": bm25_query,
                "size": k * 2
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

    # Combine and deduplicate
    for doc in docs_keyword + docs_vector:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            all_docs.append(doc)
                
    if enable_reranking:
        # Rerank the combined results
        reranked_docs = rerank_docs(query, all_docs, top_n=k)
        return fetch_neighbors(reranked_docs, vector_store)
    else:
        return fetch_neighbors(all_docs[:k], vector_store)



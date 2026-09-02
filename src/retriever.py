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
        
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_n]]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return docs[:top_n]

def retrieve_context(query, vector_store, k=5, filter_dict=None, enable_reranking=True, operator="AND"):
    """
    Retrieves context for a single decomposed query, using both semantic search and BM25 fallback.
    """
    if not vector_store:
        return []
    
    from src.query_parser import expand_query_aliases
    
    # We now assume the query passed here is a single sub-question.
    all_docs = []
    seen_content = set()
    
    fetch_k = k * 4 if enable_reranking else k
    
    # 1. Vector Search (Filter pushed to database!)
    es_filter = []
    if filter_dict:
        for key, val in filter_dict.items():
            if isinstance(val, list):
                es_filter.append({"terms": {f"metadata.{key}.keyword": val}})
            else:
                es_filter.append({"term": {f"metadata.{key}.keyword": val}})
                
    docs_vector = vector_store.similarity_search(
        query, 
        k=fetch_k, 
        fetch_k=fetch_k * 3, # Ensure num_candidates is strictly greater than k
        filter=es_filter if es_filter else None
    )
    
    # 2. BM25 Keyword Search (Manual)
    docs_keyword = []
    try:
        from src.vector_store import INDEX_NAME
        es_client = vector_store.client
        
        # Expand query for BM25 lexical match
        expanded_query = expand_query_aliases(query)
        
        # Determine Section Boosts (Removed hardcoded section heuristics for better generalization)
        section_boosts = []

            
        import re
        quoted_terms = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
        for qt in quoted_terms:
            if qt.strip():
                section_boosts.append({"match_phrase": {"text": {"query": qt.strip(), "boost": 10}}})
        
        # Base multi_match query
        bm25_query = {
            "bool": {
                "should": [
                    {
                        "simple_query_string": {
                            "query": expanded_query,
                            "default_operator": operator,
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
                "size": fetch_k
            }
        else:
            body = {
                "query": bm25_query,
                "size": fetch_k
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
        return reranked_docs
    else:
        return all_docs[:k]


def get_group_context(queries, vector_store, top_k=1):
    """
    Evaluates a batch of queries to find a confident group-level document context.
    Returns (target_docs, is_confident)
    """
    if not vector_store:
        return [], False
        
    doc_scores = {}
    explicit_docs_found = set()
    
    for q in queries:
        # Prevent group context if multiple distinct documents are explicitly named
        q_docs, is_conf = check_explicit_intent(q, vector_store)
        if is_conf and q_docs:
            for d in q_docs:
                explicit_docs_found.add(d)
                
        retrieved = retrieve_context(q, vector_store, k=4, filter_dict=None, enable_reranking=True)
        seen_in_query = set()
        for i, doc in enumerate(retrieved):
            source = doc.metadata.get("source_file")
            if source and source not in seen_in_query:
                score = max(4 - i, 1)
                doc_scores[source] = doc_scores.get(source, 0) + score
                seen_in_query.add(source)
                
    if len(explicit_docs_found) > 1:
        # User explicitly asked about multiple different contracts in this batch
        return [], False
                
    if not doc_scores:
        return [], False
        
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_docs) == 1:
        return [sorted_docs[0][0]], True
        
    top_score = sorted_docs[0][1]
    runner_up = sorted_docs[1][1]
    
    # Confident if top doc dominates the batch (avg rank 1 across almost all queries)
    # Using 3.5 multiplier ensures it must be consistently highly ranked on ALMOST EVERY query to be a Safe Batch Context.
    is_confident = (top_score >= len(queries) * 3.5) and (top_score >= runner_up + 4)
    return [sorted_docs[0][0]], is_confident

def get_question_context(query, vector_store):
    """
    Evaluates a single query to find an explicit document match via full Global Search.
    Returns (target_docs, is_confident)
    """
    if not vector_store:
        return [], False
        
    retrieved = retrieve_context(query, vector_store, k=4, filter_dict=None, enable_reranking=True)
    if not retrieved:
        return [], False
        
    # Since we don't have access to raw Cross-Encoder scores here, we check chunk density.
    # If a document owns 2 or more of the top 3 most relevant chunks, it's a confident match.
    doc_counts = {}
    for doc in retrieved[:3]:
        source = doc.metadata.get("source_file")
        if source:
            doc_counts[source] = doc_counts.get(source, 0) + 1
            
    sorted_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_docs:
        return [], False
        
    top_doc, count = sorted_docs[0]
    is_confident = (count >= 2)
    return [top_doc], is_confident

def check_explicit_intent(query, vector_store):
    """
    LLM-based Entity Resolution and Document Binding.
    Uses PostgreSQL metadata to identify if a query explicitly refers to specific documents.
    """
    if not vector_store:
        return [], False
        
    try:
        from src.rag_service import get_llm
        llm = get_llm()
        if not llm:
            return [], False
            
        from src.vector_store import get_indexed_documents
        from src.db_service import get_contract_info
        
        indexed_files = get_indexed_documents(vector_store)
        if not indexed_files:
            return [], False
            
        contract_info = get_contract_info(indexed_files)
        
        info_str = "Available Contracts:\n"
        for info in contract_info:
            info_str += f"- Filename: {info['file']} | Title: {info['title']} | Parties: {', '.join(info['parties'])}\n"
            
        prompt = f"""
        You are an expert Legal Document Router. Your task is to perform Entity Resolution and Document Binding.
        Look at the user's query and the list of available contracts. Determine if the query specifically refers to one or more of these contracts by name, title, or party.
        
        {info_str}
        
        Query: "{query}"
        
        CRITICAL RULES:
        1. If the query clearly asks about a specific contract (e.g. mentions "PhaseBio" and PhaseBio is a party in 5.pdf), return a JSON list containing the exact filename (e.g. ["5.pdf"]).
        2. If the query compares two contracts, return both (e.g. ["1.pdf", "5.pdf"]).
        3. If the query is generic (e.g. "What is the penalty?") and does NOT name a specific contract or party, return an empty list [].
        4. Do NOT explain. Return ONLY a valid JSON list of filenames.
        """
        
        resp = llm.invoke(prompt).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.endswith("```"): resp = resp[:-3]
        
        import json
        target_files = json.loads(resp.strip())
        
        if isinstance(target_files, list) and len(target_files) > 0:
            valid_files = [f for f in target_files if f in indexed_files]
            if valid_files:
                return valid_files, True
                
    except Exception as e:
        print(f"Explicit intent check failed: {e}")
        
    return [], False

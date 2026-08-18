import os
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from elasticsearch import Elasticsearch

# Defaulting to localhost, assuming you'll run it locally via Docker
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX_NAME = "contract_rag_index"

def get_es_client():
    return Elasticsearch(ES_URL)

def build_vector_store(chunks, embeddings_model):
    """Builds a new Elasticsearch vector store from chunks."""
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    
    # Enable Hybrid Search (BM25 Keyword + Dense Vector)
    # RRF (Reciprocal Rank Fusion) combines the lexical and semantic scores
    vector_store = ElasticsearchStore.from_documents(
        docs,
        embeddings_model,
        es_url=ES_URL,
        index_name=INDEX_NAME,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy()
    )
    return vector_store

def add_to_vector_store(chunks, embeddings_model, vector_store=None):
    """Adds chunks to the Elasticsearch index."""
    if vector_store is None:
        es_client = get_es_client()
        if not es_client.indices.exists(index=INDEX_NAME):
            return build_vector_store(chunks, embeddings_model)
        else:
            vector_store = load_vector_store(embeddings_model)
    
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    try:
        vector_store.add_documents(docs)
    except Exception as e:
        if hasattr(e, 'errors'):
            print("ELASTICSEARCH REJECTION DETAILS:")
            for err in e.errors[:5]: # Print first 5 errors
                print(err)
            import streamlit as st
            st.error(f"Elasticsearch rejected some documents. Check your terminal for details. First error: {e.errors[0]}")
        raise e
    return vector_store

def save_vector_store(vector_store):
    """No-op for Elasticsearch as it automatically syncs to the server."""
    pass

def load_vector_store(embeddings_model):
    """Connects to the existing Elasticsearch index."""
    try:
        es_client = get_es_client()
        if es_client.indices.exists(index=INDEX_NAME):
            return ElasticsearchStore(
                es_url=ES_URL,
                index_name=INDEX_NAME,
                embedding=embeddings_model,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy()
            )
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
    return None

def get_indexed_documents(vector_store):
    """Returns a list of unique source files currently in the index using aggregation."""
    if not vector_store:
        return []
    
    try:
        es_client = get_es_client()
        if not es_client.indices.exists(index=INDEX_NAME):
            return []
            
        # Use Elasticsearch aggregation to get unique source_file metadata
        query = {
            "size": 0,
            "aggs": {
                "unique_sources": {
                    "terms": {
                        "field": "metadata.source_file.keyword",
                        "size": 1000
                    }
                }
            }
        }
        response = es_client.search(index=INDEX_NAME, body=query)
        buckets = response.get("aggregations", {}).get("unique_sources", {}).get("buckets", [])
        return [bucket["key"] for bucket in buckets]
    except Exception as e:
        print(f"Error getting indexed documents from ES: {e}")
        return []

def clear_vector_store():
    """Deletes the Elasticsearch index."""
    try:
        es_client = get_es_client()
        if es_client.indices.exists(index=INDEX_NAME):
            es_client.indices.delete(index=INDEX_NAME)
    except Exception as e:
        print(f"Error clearing vector store: {e}")

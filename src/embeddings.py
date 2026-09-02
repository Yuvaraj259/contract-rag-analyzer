from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model():
    """Returns the HuggingFace embeddings model."""
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # BGE models perform best with normalized embeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

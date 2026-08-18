import os
from dotenv import load_dotenv

from src.document_loader import load_document
from src.text_cleaner import clean_text
from src.contract_parser import extract_metadata
from src.chunking import chunk_document
from src.embeddings import get_embeddings_model
from src.vector_store import load_vector_store, add_to_vector_store, save_vector_store, get_indexed_documents

load_dotenv()

def process_folder():
    contracts_dir = os.path.join("data", "contracts")
    if not os.path.exists(contracts_dir):
        print(f"Directory {contracts_dir} does not exist.")
        return

    embeddings_model = get_embeddings_model()
    vector_store = load_vector_store(embeddings_model)
    indexed = get_indexed_documents(vector_store) if vector_store else []
    
    all_chunks = []
    
    for filename in os.listdir(contracts_dir):
        file_path = os.path.join(contracts_dir, filename)
        if os.path.isfile(file_path):
            if filename in indexed:
                print(f"Skipping {filename}, already indexed.")
                continue
                
            print(f"Processing {filename}...")
            try:
                raw_text = load_document(file_path)
                cleaned_text = clean_text(raw_text)
                metadata = extract_metadata(cleaned_text, filename)
                chunks = chunk_document(cleaned_text, metadata)
                all_chunks.extend(chunks)
                print(f"Successfully processed {filename} into {len(chunks)} chunks.")
            except Exception as e:
                print(f"Error processing {filename}: {e}. (Note: .doc files are not supported natively without extra tools. Convert them to .docx or .pdf)")

    if all_chunks:
        print(f"Indexing {len(all_chunks)} chunks to FAISS...")
        vector_store = add_to_vector_store(all_chunks, embeddings_model, vector_store)
        save_vector_store(vector_store)
        print("Indexing complete!")
    else:
        print("No new valid documents were processed.")

if __name__ == "__main__":
    process_folder()

# Resume Search and Analysis Using RAG

This is a Retrieval-Augmented Generation application that can read, index, search, and analyse candidate resumes. It uses FAISS as a vector store, `sentence-transformers` for embeddings, and an LLM to answer queries about the indexed resumes.

## Features
- **Document Processing**: Extracts text from PDF and DOCX files.
- **Chunking**: Splits resumes logically into sections (Summary, Skills, Experience, Projects, Education) instead of purely character-based chunking.
- **RAG QA**: Uses LangChain + Vector database to answer context-aware queries about candidates.
- **Source Grounding**: Retrieves the candidate's name and original file to prevent hallucination.
- **Candidate Comparison**: Compares candidates based on skills, experience, etc.

## Setup Instructions
1. Clone the repository and navigate to `resume-rag`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```


4. Place your sample `.pdf` or `.docx` resumes in the `data/resumes/` folder.
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Architecture Summary
- **Loaders**: PyMuPDF (`fitz`) for PDFs and `python-docx` for word documents.
- **Chunking**: Heuristic-based regex split into major resume sections.
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` for generating embeddings.
- **Vector Store**: FAISS local index for fast similarity search.
- **LLM**: LangChain integrated with Google's Gemini / another local model for grounded answers.

## Known Limitations & Improvements
- *OCR for Scanned PDFs*: Currently, it only processes textual PDFs. Adding `pytesseract` can help extract text from scanned images.
- *Information Extraction*: A dedicated LLM extraction pass to format resume to JSON would improve structured querying (e.g. strict experience filtering).
- *Reranking*: Implementing Cross-Encoder reranking could improve accuracy when dealing with thousands of candidates.

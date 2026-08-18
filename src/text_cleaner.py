import re

def clean_text(text):
    """Cleans up the extracted text by removing unnecessary spaces and formatting."""
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove multiple newlines (reduce to max 2 newlines)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove non-ascii characters that might mess up processing (optional)
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()

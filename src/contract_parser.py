import re

# Load small english model for NER if available, else basic fallback
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    nlp = None

def extract_contract_title(text, filename=""):
    """Extracts a readable title for the contract."""
    if filename:
        import os
        base = os.path.basename(filename)
        name = os.path.splitext(base)[0]
        # Remove underscores and dashes
        name = re.sub(r'[\-_]', ' ', name).strip()
        # Clean up multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()
        if name and len(name) > 2:
            return name[:100].title()
            
    # Fallback to the first substantial line in the document
    lines = text.strip().split('\n')
    for line in lines[:10]:
        line_clean = line.strip()
        if line_clean and 5 < len(line_clean) < 100:
            return line_clean.title()
            
    return "Unknown Contract"

def extract_effective_date(text):
    """Attempts to find the Effective Date of the agreement."""
    # Look for common date patterns following "Effective Date" or "Dated as of"
    date_regex = r"(?:effective date|dated as of|dated)[^\n\w]{0,30}?((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    
    # Search in the first 2000 characters (usually where the intro is)
    match = re.search(date_regex, text[:2000], re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unknown"

def extract_parties(text):
    """Attempts to extract the organizations involved in the contract."""
    parties = set()
    
    # Heuristic 1: Regex for "by and between [Party 1] and [Party 2]"
    between_regex = r"by and between\s+([A-Z][a-zA-Z0-9\s,&]+?)(?:\s+\(|,)\s*and\s+([A-Z][a-zA-Z0-9\s,&]+?)(?:\s+\(|,)"
    match = re.search(between_regex, text[:2000], re.IGNORECASE)
    if match:
        parties.add(match.group(1).strip())
        parties.add(match.group(2).strip())
    
    # Heuristic 2: Use SpaCy NER to find ORGs in the first 1000 characters
    if not parties and nlp:
        doc = nlp(text[:1000]) 
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_name = ent.text.strip().replace('\n', ' ')
                parties.add(org_name)
                
    # Clean up obvious false positives
    ignore_list = ["the", "company", "client", "vendor", "customer", "contractor", "licensor", "licensee", "party", "parties", "agreement"]
    clean_parties = [p for p in parties if len(p) > 2 and p.lower() not in ignore_list]
    
    # Return up to the top 2 parties
    return clean_parties[:2]

def extract_metadata(text, filename):
    """
    Main entry point for extracting contract-level metadata.
    This replaces the resume-centric extraction logic.
    """
    return {
        "contract_title": extract_contract_title(text, filename),
        "effective_date": extract_effective_date(text),
        "parties": extract_parties(text),
        "source_file": filename
    }

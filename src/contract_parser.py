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
    # Look for common date patterns following "Effective Date", "Dated as of", "effective from", etc.
    date_regex = r"(?:effective date|dated as of|dated|effective from)[^\n\w]{0,30}?((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:day of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)[,\s]+\d{4})"
    
    # Search in the first 3000 characters
    match = re.search(date_regex, text[:3000], re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_parties(text):
    """Attempts to extract the organizations involved in the contract."""
    parties = set()
    
    # Priority 1: Multi-line or single-line BETWEEN ... AND ...
    between_patterns = [
        r"(?:by and between|between|this agreement is between)[\s:]+([A-Z][a-zA-Z0-9\s,&.\-]+?)(?:\s+\(|,).+?(?:and)[\s:]+([A-Z][a-zA-Z0-9\s,&.\-]+?)(?:\s+\(|,)",
        r"BETWEEN:[\s]*([^\n(]+).*?AND:[\s]*([^\n(]+)"
    ]
    
    for pattern in between_patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE | re.DOTALL)
        if match:
            parties.add(match.group(1).strip())
            parties.add(match.group(2).strip())
            break
            
    # Priority 2: Use SpaCy NER to find ORGs in the first 1500 characters, BUT ignore known non-party text
    if not parties and nlp:
        # Strip out common reference text that trips up NER
        safe_text = re.sub(r'(?i)(united nations centre for trade facilitation|electronic business|un/cefact|applicable laws)', '', text[:1500])
        doc = nlp(safe_text) 
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_name = ent.text.strip().replace('\n', ' ')
                parties.add(org_name)
                
    # Clean up obvious false positives
    ignore_list = ["the", "company", "client", "vendor", "customer", "contractor", "licensor", "licensee", "party", "parties", "agreement"]
    clean_parties = [p for p in parties if len(p) > 2 and p.lower() not in ignore_list]
    
    # Return up to the top 2 parties
    return clean_parties[:2]

def extract_contract_type(text, title):
    """Categorizes the structure/type of the contract."""
    combined = (title + " " + text[:1000]).lower()
    if "non-disclosure" in combined or "nda" in combined:
        return "Non-Disclosure Agreement (NDA)"
    if "master service" in combined or "msa" in combined:
        return "Master Services Agreement (MSA)"
    if "end user license" in combined or "eula" in combined:
        return "End User License Agreement (EULA)"
    if "service level" in combined or "sla" in combined:
        return "Service Level Agreement (SLA)"
    if "employment" in combined:
        return "Employment Agreement"
    return "General Commercial Contract"

def extract_metadata(text, filename):
    """
    Main entry point for extracting contract-level metadata.
    This replaces the resume-centric extraction logic.
    """
    title = extract_contract_title(text, filename)
    return {
        "contract_title": title,
        "contract_type": extract_contract_type(text, title),
        "effective_date": extract_effective_date(text),
        "parties": extract_parties(text),
        "source_file": filename
    }

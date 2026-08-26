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
        safe_text = re.sub(r'(?i)(united nations centre for trade facilitation|electronic business|un/cefact|netherlands law)', '', text[:1500])
        doc = nlp(safe_text) 
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_name = ent.text.strip().replace('\n', ' ')
                parties.add(org_name)
                
    # Clean up obvious false positives
    ignore_list = ["the", "company", "client", "vendor", "customer", "contractor", "licensor", "licensee", "party", "parties", "agreement"]
    clean_parties = [p for p in parties if len(p) > 2 and p.lower() not in ignore_list]
    
    # Return up to the top 5 parties
    return clean_parties[:5]

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
    Main entry point for extracting contract-level metadata using an LLM.
    """
    try:
        from src.rag_service import get_llm
        import json
        llm = get_llm()
        if llm:
            prompt = f"""You are an expert legal assistant. Extract the metadata from the following contract text.
            If a field cannot be determined, return "Unknown".
            Return ONLY a valid JSON object matching this schema exactly:
            {{
                "contract_title": "The exact formal title of the agreement",
                "parties": ["Party 1", "Party 2"],
                "effective_date": "The effective date of the agreement"
            }}
            
            Contract Text (First 3000 characters):
            {text[:3000]}
            """
            
            response = llm.invoke(prompt).strip()
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                meta = json.loads(response[start_idx:end_idx])
                return {
                    "contract_title": meta.get("contract_title", "Unknown"),
                    "contract_type": extract_contract_type(text, meta.get("contract_title", "")),
                    "effective_date": meta.get("effective_date", "Unknown"),
                    "parties": meta.get("parties", []),
                    "source_file": filename
                }
    except Exception as e:
        print(f"LLM Metadata Extraction Failed for {filename}: {e}")
        
    # Fallback to simple extraction
    title = extract_contract_title(text, filename)
    return {
        "contract_title": title,
        "contract_type": extract_contract_type(text, title),
        "effective_date": extract_effective_date(text),
        "parties": extract_parties(text),
        "source_file": filename
    }

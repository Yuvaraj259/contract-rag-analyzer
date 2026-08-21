import re
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_sections(text):
    """
    Heuristically splits a document (like a software contract or resume) into major sections based on common headings.
    """
    section_headers = [
        # Software Contract Headers
        r"ARTICLE\s+[IVX\d]+", r"SECTION\s+\d+(\.\d+)*",
        r"LICENSE GRANT", r"SCOPE OF LICENSE", r"GRANT OF LICENSE",
        r"INTELLECTUAL PROPERTY", r"OWNERSHIP", r"PROPRIETARY RIGHTS",
        r"SERVICE LEVEL AGREEMENT", r"SLA", r"UPTIME",
        r"DATA PRIVACY", r"SECURITY", r"DATA PROTECTION", r"CONFIDENTIALITY", r"CONFIDENTIAL INFORMATION",
        r"SUPPORT AND MAINTENANCE", r"MAINTENANCE",
        r"SOURCE CODE ESCROW", r"ESCROW",
        r"LIMITATION OF LIABILITY", r"LIABILITY", r"INDEMNIFICATION", r"INDEMNITY",
        r"TERM AND TERMINATION", r"TERMINATION", r"TERM",
        r"PAYMENT", r"FEES", r"TAXES", r"PRICING",
        r"WARRANTIES", r"REPRESENTATIONS", r"DISCLAIMER",
        r"GOVERNING LAW", r"JURISDICTION", r"DISPUTE RESOLUTION", r"ARBITRATION",
        r"FORCE MAJEURE", r"MISCELLANEOUS", r"GENERAL", r"ENTIRE AGREEMENT",
        
        # Legacy Resume Headers
        "PROFILE", "SUMMARY", "OBJECTIVE",
        "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "WORK HISTORY",
        "EDUCATION", "ACADEMIC BACKGROUND",
        "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
        "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS",
        "CERTIFICATIONS", "AWARDS", "PUBLICATIONS"
    ]
    
    # Regex to match headers (e.g. "EXPERIENCE" or "WORK EXPERIENCE" on its own line)
    # Allows for some leading/trailing whitespace or special chars like ":"
    header_pattern = re.compile(
        r'^\s*(' + '|'.join(section_headers) + r')\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    matches = list(header_pattern.finditer(text))
    
    sections = {}
    if not matches:
        # If no headers found, treat the whole document as one section
        sections["GENERAL"] = text
        return sections
        
    # Extract sections between headers
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        section_name = matches[i].group(1).upper()
        section_content = text[start:end].strip()
        
        # In case of duplicate headers, append
        if section_name in sections:
            sections[section_name] += "\n\n" + section_content
        else:
            sections[section_name] = section_content
            
    # Also capture whatever was before the first header (usually contact info/summary)
    intro_content = text[:matches[0].start()].strip()
    if intro_content:
        sections["INTRO"] = intro_content
        
    return sections

def extract_section_numbers(section_name):
    """Extracts section_number and clause_number if present."""
    sec_num = "Unknown"
    clause_num = "Unknown"
    
    m_art = re.search(r'ARTICLE\s+([IVX\d]+)', section_name, re.IGNORECASE)
    if m_art:
        sec_num = m_art.group(1)
        
    m_sec = re.search(r'(?:SECTION\s+)?(\d+(?:\.\d+)*)', section_name, re.IGNORECASE)
    if m_sec:
        if '.' in m_sec.group(1):
            parts = m_sec.group(1).split('.')
            sec_num = parts[0]
            clause_num = parts[1]
        else:
            sec_num = m_sec.group(1)
            
    return sec_num, clause_num

def chunk_document(text, metadata):
    """
    Splits the document into section-based chunks. If a section is too large,
    falls back to RecursiveCharacterTextSplitter for that section.
    Returns a list of dicts: {"text": chunk_text, "metadata": metadata}
    """
    sections = split_into_sections(text)
    
    # Fallback splitter for very large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = []
    chunk_index = 0
    parent_section = "None"
    
    for section_name, section_content in sections.items():
        if not section_content:
            continue
            
        sec_num, clause_num = extract_section_numbers(section_name)
        if "ARTICLE" in section_name:
            parent_section = section_name
            
        # Determine page and line number by finding where this section starts in the original text
        page_number = "1"
        line_number = "Unknown"
        idx = text.find(section_content[:50])
        if idx != -1:
            page_matches = list(re.finditer(r'---\s*PAGE\s+(\d+)\s*---', text[:idx]))
            if page_matches:
                page_number = page_matches[-1].group(1)
            
            line_number = str(text.count('\n', 0, idx) + 1)
                
        # Clean the page markers out of the final chunk text so it doesn't confuse the LLM
        clean_section_content = re.sub(r'\n?---\s*PAGE\s+\d+\s*---\n?', '\n', section_content).strip()
        
        # Add section name context to the chunk
        context_prefix = f"[{section_name}] "
        
        if len(clean_section_content) > 1000:
            # Sub-chunk if too large
            sub_chunks = text_splitter.split_text(clean_section_content)
            for sc in sub_chunks:
                chunk_meta = metadata.copy()
                chunk_meta["section"] = section_name
                chunk_meta["section_number"] = sec_num
                chunk_meta["clause_number"] = clause_num
                chunk_meta["page_number"] = page_number
                chunk_meta["line_number"] = line_number
                chunk_meta["parent_section"] = parent_section
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["chunk_id"] = f"{metadata.get('document_id', 'doc_unknown')}_chunk_{chunk_index}"
                chunks.append({
                    "text": context_prefix + sc,
                    "metadata": chunk_meta
                })
                chunk_index += 1
        else:
            chunk_meta = metadata.copy()
            chunk_meta["section"] = section_name
            chunk_meta["section_number"] = sec_num
            chunk_meta["clause_number"] = clause_num
            chunk_meta["page_number"] = page_number
            chunk_meta["line_number"] = line_number
            chunk_meta["parent_section"] = parent_section
            chunk_meta["chunk_index"] = chunk_index
            chunk_meta["chunk_id"] = f"{metadata.get('document_id', 'doc_unknown')}_chunk_{chunk_index}"
            chunks.append({
                "text": context_prefix + clean_section_content,
                "metadata": chunk_meta
            })
            chunk_index += 1
            
    return chunks

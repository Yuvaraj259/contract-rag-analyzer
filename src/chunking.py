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
    for section_name, section_content in sections.items():
        if not section_content:
            continue
            
        # Add section name context to the chunk
        context_prefix = f"[{section_name}] "
        
        if len(section_content) > 1000:
            # Sub-chunk if too large
            sub_chunks = text_splitter.split_text(section_content)
            for sc in sub_chunks:
                chunk_meta = metadata.copy()
                chunk_meta["section"] = section_name
                chunks.append({
                    "text": context_prefix + sc,
                    "metadata": chunk_meta
                })
        else:
            chunk_meta = metadata.copy()
            chunk_meta["section"] = section_name
            chunks.append({
                "text": context_prefix + section_content,
                "metadata": chunk_meta
            })
            
    return chunks

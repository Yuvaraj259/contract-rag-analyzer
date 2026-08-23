import re
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_sections(text):
    """
    Heuristically splits a document (like a software contract or resume) into major sections based on common headings.
    """
    section_headers = [
        # Introductory headers
        r"BETWEEN", r"AND", r"RECITALS", r"PREAMBLE", r"PARTIES", r"BACKGROUND",
        
        # Software Contract Headers
        r"ARTICLE\s+[IVX\d]+", r"SECTION\s+\d+(\.\d+)*",
        r"LICENSE GRANT", r"SCOPE OF LICENSE", r"GRANT OF LICENSE",
        r"INTELLECTUAL PROPERTY", r"OWNERSHIP", r"PROPRIETARY RIGHTS",
        r"SERVICE LEVEL AGREEMENT", r"SLA", r"UPTIME",
        r"DATA PRIVACY", r"SECURITY", r"DATA PROTECTION", r"CONFIDENTIALITY", r"CONFIDENTIAL INFORMATION",
        r"SUPPORT AND MAINTENANCE", r"MAINTENANCE",
        r"SOURCE CODE ESCROW", r"ESCROW",
        r"LIMITATION OF LIABILITY", r"LIABILITY", r"INDEMNIFICATION", r"INDEMNITY",
        r"TERM AND TERMINATION", r"TERMINATION", r"TERM", r"TERM OF AGREEMENT", r"AGREEMENT PERIOD", r"COMMENCEMENT DATE",
        r"PAYMENT TERMS", r"PAYMENT", r"FEES", r"TAXES", r"PRICING", r"ESTIMATION AND COMMERCIALS",
        r"SCOPE OF WORK", r"SCOPE OF DELIVERABLES", r"ENGAGEMENT PROCESS(?: & MILESTONE)?", r"CHANGE ORDERS", r"ASSUMPTIONS AND DEPENDENCIES",
        r"WARRANTIES", r"REPRESENTATIONS", r"DISCLAIMER",
        r"GOVERNING LAW", r"JURISDICTION", r"DISPUTE RESOLUTION", r"ARBITRATION",
        r"FORCE MAJEURE", r"MISCELLANEOUS", r"GENERAL", r"ENTIRE AGREEMENT", r"ENTIRE AGREEMENT AND AMENDMENTS",
        r"LANGUAGE", r"NOTICE", r"EFFECT OF HEADINGS", r"BINDING EFFECT",
        
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
    
    sections = []
    if not matches:
        # If no headers found, treat the whole document as one section
        sections.append({"name": "GENERAL", "content": text, "start_idx": 0})
        return sections
        
    # Extract sections between headers
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        section_name = matches[i].group(1).upper()
        section_content = text[start:end].strip()
        
        sections.append({
            "name": section_name,
            "content": section_content,
            "start_idx": start
        })
            
    # Also capture whatever was before the first header (usually contact info/summary)
    intro_content = text[:matches[0].start()].strip()
    if intro_content:
        sections.insert(0, {"name": "INTRO", "content": intro_content, "start_idx": 0})
        
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
    
    for sec_data in sections:
        section_name = sec_data["name"]
        section_content = sec_data["content"]
        idx = sec_data["start_idx"]
        
        if not section_content:
            continue
            
        sec_num, clause_num = extract_section_numbers(section_name)
        if "ARTICLE" in section_name:
            parent_section = section_name
            
        # Determine page and line number by finding where this section starts in the original text
        pdf_page_number = "1"
        line_number = "Unknown"
        doc_page_number = "Unknown"
        
        if idx != -1:
            page_matches = list(re.finditer(r'---\s*PAGE\s+(\d+)\s*---', text[:idx]))
            if page_matches:
                pdf_page_number = page_matches[-1].group(1)
                
            doc_page_matches = list(re.finditer(r'Page\s+(\d+)\s+of\s+\d+', text[:idx], re.IGNORECASE))
            if doc_page_matches:
                doc_page_number = doc_page_matches[-1].group(1)
            
            line_start = text.count('\n', 0, idx) + 1
            line_end = line_start + section_content.count('\n')
            if line_start == line_end:
                line_number = str(line_start)
            else:
                line_number = f"{line_start}-{line_end}"
            
        # Try to find a printed document page number like "Page 2 of 7" in the section as fallback
        if doc_page_number == "Unknown":
            doc_page_match = re.search(r'Page\s+(\d+)\s+of\s+\d+', section_content, re.IGNORECASE)
            if doc_page_match:
                doc_page_number = doc_page_match.group(1)
                
        # Clean the page markers out of the final chunk text so it doesn't confuse the LLM
        clean_section_content = re.sub(r'\n?---\s*PAGE\s+\d+\s*---\n?', '\n', section_content).strip()
        
        # Add section name context to the chunk
        context_prefix = f"[{section_name}] "
        
        if len(clean_section_content) > 1000:
            # Sub-chunk if too large, use raw section_content to map exact line numbers
            sub_chunks = text_splitter.split_text(section_content)
            for sc in sub_chunks:
                sc_idx = text.find(sc[:100], idx)
                if sc_idx == -1:
                    sc_idx = text.find(sc[:50], idx)
                    
                sc_pdf_page = pdf_page_number
                sc_line_num = line_number
                
                if sc_idx != -1:
                    page_matches = list(re.finditer(r'---\s*PAGE\s+(\d+)\s*---', text[:sc_idx]))
                    if page_matches:
                        sc_pdf_page = page_matches[-1].group(1)
                        
                    doc_page_matches = list(re.finditer(r'Page\s+(\d+)\s+of\s+\d+', text[:sc_idx], re.IGNORECASE))
                    if doc_page_matches:
                        sc_doc_page = doc_page_matches[-1].group(1)
                    else:
                        sc_doc_page = doc_page_number
                        
                    sc_line_start = text.count('\n', 0, sc_idx) + 1
                    sc_line_end = sc_line_start + sc.count('\n')
                    sc_line_num = f"{sc_line_start}-{sc_line_end}" if sc_line_start != sc_line_end else str(sc_line_start)
                
                # Now clean the page markers out of the sub-chunk before embedding
                clean_sc = re.sub(r'\n?---\s*PAGE\s+\d+\s*---\n?', '\n', sc).strip()
                
                chunk_meta = metadata.copy()
                chunk_meta["section"] = section_name
                chunk_meta["section_number"] = sec_num
                chunk_meta["clause_number"] = clause_num
                chunk_meta["page_number"] = sc_pdf_page
                chunk_meta["pdf_page_number"] = sc_pdf_page
                chunk_meta["document_page_number"] = sc_doc_page if sc_idx != -1 else doc_page_number
                chunk_meta["line_number"] = sc_line_num
                chunk_meta["parent_section"] = parent_section
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["chunk_id"] = f"{metadata.get('document_id', 'doc_unknown')}_chunk_{chunk_index}"
                chunks.append({
                    "text": context_prefix + clean_sc,
                    "metadata": chunk_meta
                })
                chunk_index += 1
        else:
            chunk_meta = metadata.copy()
            chunk_meta["section"] = section_name
            chunk_meta["section_number"] = sec_num
            chunk_meta["clause_number"] = clause_num
            chunk_meta["page_number"] = pdf_page_number # Legacy compat
            chunk_meta["pdf_page_number"] = pdf_page_number
            chunk_meta["document_page_number"] = doc_page_number
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

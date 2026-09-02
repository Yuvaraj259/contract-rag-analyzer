import re
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_sections(text):
    """
    Heuristically splits a document (like a software contract or resume) into major sections based on common headings.
    """
    section_headers = [
        # Introductory headers
        r"RECITALS", r"PREAMBLE", r"PARTIES", r"BACKGROUND",
        
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
        r"ENTIRE AGREEMENT AND GOVERNING LAW AND JURISDICTION", r"ENTIRE AGREEMENT AND GOVERNING LAW",
        r"LANGUAGE", r"NOTICE", r"EFFECT OF HEADINGS", r"BINDING EFFECT",
        
        # Legacy Resume Headers
        "PROFILE", "SUMMARY", "OBJECTIVE",
        "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "WORK HISTORY",
        "EDUCATION", "ACADEMIC BACKGROUND",
        "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
        "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS",
        "CERTIFICATIONS", "AWARDS", "PUBLICATIONS"
        # Exhibits
        r"EXHIBIT\s+[A-Z\d]+", r"SCHEDULE\s+[A-Z\d]+", r"APPENDIX\s+[A-Z\d]+", r"ANNEX\s+[A-Z\d]+"
    ]
    
    # Regex to match headers (e.g. "ARTICLE 5 - COMPENSATION" on its own line)
    # Allows for trailing text on the line, but captures the whole line as the section name
    header_pattern = re.compile(
        r'^\s*(' + '|'.join(section_headers) + r')(?:[\s\:\-\.].{0,80})?$',
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

def extract_page_mapping(text):
    """
    Returns a dictionary mapping pdf_page_number (int) to doc_page (str or None)
    """
    mapping = {}
    parts = re.split(r'---\s*PAGE\s+(\d+)\s*---', text)
    
    for i in range(1, len(parts), 2):
        pdf_page_num = int(parts[i])
        page_content = parts[i+1]
        
        doc_page = None
        doc_page_matches = list(re.finditer(r'Page\s+(\d+)(?:\s*(?:of|/)\s*(\d+))?', page_content, re.IGNORECASE))
        if doc_page_matches:
            doc_page = doc_page_matches[-1].group(1)
        else:
            alt_matches = list(re.finditer(r'^\s*-\s*(\d+)\s*-\s*$', page_content, re.MULTILINE))
            if alt_matches:
                doc_page = alt_matches[-1].group(1)
                
        mapping[pdf_page_num] = doc_page
    return mapping

def build_line_index(text, page_mapping):
    lines_info = []
    current_pdf_page = 1
    current_line_num = 1
    char_pos = 0
    
    lines = text.split('\n')
    for line in lines:
        m = re.match(r'^---\s*PAGE\s+(\d+)\s*---$', line.strip())
        if m:
            current_pdf_page = int(m.group(1))
            current_line_num = 1
            lines_info.append({
                "is_marker": True,
                "text": line,
                "pdf_page": current_pdf_page,
                "doc_page": page_mapping.get(current_pdf_page),
                "line_num": 0,
                "char_start": char_pos,
                "char_end": char_pos + len(line)
            })
        else:
            lines_info.append({
                "is_marker": False,
                "text": line,
                "pdf_page": current_pdf_page,
                "doc_page": page_mapping.get(current_pdf_page),
                "line_num": current_line_num,
                "char_start": char_pos,
                "char_end": char_pos + len(line)
            })
            current_line_num += 1
            
        char_pos += len(line) + 1 # +1 for \n
        
    return lines_info

import json

def extract_page_ranges(section_lines):
    if not section_lines:
        return []
        
    page_ranges = []
    current_pdf = None
    group_lines = []
    
    def flush_group():
        nonlocal page_ranges, group_lines
        if not group_lines:
            return
        pdf_page = group_lines[0]["pdf_page"]
        doc_page = group_lines[0]["doc_page"]
        start_l = group_lines[0]["line_num"]
        end_l = group_lines[-1]["line_num"]
        
        page_ranges.append({
            "pdf_page": pdf_page,
            "document_page": doc_page,
            "line_start": start_l,
            "line_end": end_l
        })
        group_lines.clear()

    for li in section_lines:
        if li["pdf_page"] != current_pdf:
            flush_group()
            current_pdf = li["pdf_page"]
        group_lines.append(li)
        
    flush_group()
    return page_ranges

def chunk_document(text, metadata):
    """
    Splits the document into section-based chunks with structured line and page metadata.
    """
    sections = split_into_sections(text)
    page_mapping = extract_page_mapping(text)
    lines_info = build_line_index(text, page_mapping)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    chunk_index = 0
    parent_section = "None"
    
    for sec_data in sections:
        section_name = sec_data["name"]
        section_content = sec_data["content"]
        idx = sec_data["start_idx"]
        
        if not section_content.strip():
            continue
            
        sec_num, clause_num = extract_section_numbers(section_name)
        if "ARTICLE" in section_name:
            parent_section = section_name
            
        section_end = idx + len(section_content)
        
        # Get lines for this section
        section_lines = []
        for li in lines_info:
            if not li["is_marker"] and li["char_end"] >= idx and li["char_start"] < section_end:
                section_lines.append(li)
                
        if not section_lines:
            continue
            
        clean_section_content = "\n".join([li["text"] for li in section_lines])
        
        if len(clean_section_content) > 2000:
            sub_chunks = text_splitter.split_text(clean_section_content)
            current_line_idx = 0
            
            for sc in sub_chunks:
                sc_lines = sc.split('\n')
                start_match = -1
                for i in range(current_line_idx, len(section_lines)):
                    if section_lines[i]["text"].strip() == sc_lines[0].strip() and sc_lines[0].strip() != "":
                        start_match = i
                        break
                
                if start_match != -1:
                    end_match = min(start_match + len(sc_lines), len(section_lines))
                    sc_line_objs = section_lines[start_match:end_match]
                    current_line_idx = start_match + max(1, len(sc_lines) - 5)
                else:
                    sc_line_objs = section_lines
                
                page_ranges = extract_page_ranges(sc_line_objs)
                final_text = "\n".join([li["text"] for li in sc_line_objs])
                
                chunk_meta = metadata.copy()
                chunk_meta["section"] = section_name
                chunk_meta["section_number"] = sec_num
                chunk_meta["clause_number"] = clause_num
                chunk_meta["parent_section"] = parent_section
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["chunk_id"] = f"{metadata.get('document_id', 'doc_unknown')}_chunk_{chunk_index}"
                chunk_meta["page_ranges"] = json.dumps(page_ranges)
                
                chunks.append({
                    "text": final_text,
                    "metadata": chunk_meta
                })
                chunk_index += 1
        else:
            page_ranges = extract_page_ranges(section_lines)
            final_text = "\n".join([li["text"] for li in section_lines])
            
            chunk_meta = metadata.copy()
            chunk_meta["section"] = section_name
            chunk_meta["section_number"] = sec_num
            chunk_meta["clause_number"] = clause_num
            chunk_meta["parent_section"] = parent_section
            chunk_meta["chunk_index"] = chunk_index
            chunk_meta["chunk_id"] = f"{metadata.get('document_id', 'doc_unknown')}_chunk_{chunk_index}"
            chunk_meta["page_ranges"] = json.dumps(page_ranges)
            
            chunks.append({
                "text": final_text,
                "metadata": chunk_meta
            })
            chunk_index += 1
            
    return chunks

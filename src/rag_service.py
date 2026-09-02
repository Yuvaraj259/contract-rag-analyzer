import os
import json
import re
import streamlit as st
from langchain_ollama import OllamaLLM

@st.cache_resource
def get_llm():
    model_name = "qwen2.5:7b"
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    return OllamaLLM(
        model=model_name,
        base_url=ollama_url,
        temperature=0.1,
        num_ctx=8192
    )

def contextualize_query(query: str, history: list) -> str:
    """
    Given a chat history and the latest user query, reformulates the query
    to be a standalone question without needing the chat history context.
    """
    if not history:
        return query
        
    llm = get_llm()
    if not llm:
        return query
        
    hist_str = "\n".join([f"User: {h['q']}\nAI: {h['a']}" for h in history[-3:]])
    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history. Do NOT answer the question, just reformulate it.

Chat History:
{hist_str}

Follow Up Input: {query}
Standalone question:"""
    
    try:
        resp = llm.invoke(prompt).strip()
        # Clean up any potential markdown or prefixes
        if resp.lower().startswith("standalone question:"):
            resp = resp[20:].strip()
        return resp
    except Exception as e:
        print(f"Contextualization failed: {e}")
        return query

def format_chunk_context(doc):
    metadata = doc.metadata
    page_ranges_str = metadata.get("page_ranges", "[]")
    try:
        page_ranges = json.loads(page_ranges_str) if isinstance(page_ranges_str, str) else page_ranges_str
    except Exception:
        page_ranges = []

    chunk_id = metadata.get("chunk_id", "unknown")
    source_file = metadata.get("source_file", "Unknown")
    section = metadata.get("section", "Unknown")

    header = f"[CHUNK_ID: {chunk_id}]\n[Document: {source_file}]\n[Section: {section}]\n"
    for pr in page_ranges:
        doc_page = pr.get("document_page")
        pdf_page = pr.get("pdf_page")
        start_l = pr.get("line_start")
        end_l = pr.get("line_end")
        page_str = f"Document Page {doc_page}" if doc_page is not None else f"PDF Page {pdf_page}"
        line_str = f"Lines {start_l}-{end_l}" if start_l != end_l else f"Line {start_l}"
        header += f"[{page_str} | {line_str}]\n"

    return header + "\n" + doc.page_content + "\n"


def log_retrieval(query, retrieved_docs):
    """
    Diagnostic logging — prints exactly what was retrieved for a query,
    BEFORE the LLM ever sees it. This is the only way to tell retrieval
    bugs apart from generation (hallucination) bugs.
    """
    print(f"\n{'='*80}\n[RETRIEVAL LOG] Query: {query}\n{'='*80}")
    if not retrieved_docs:
        print("  !! NO DOCUMENTS RETRIEVED !!")
        return
    for i, d in enumerate(retrieved_docs):
        meta = d.metadata
        print(f"  [{i}] chunk_id={meta.get('chunk_id')} "
              f"source={meta.get('source_file')} "
              f"section={meta.get('section')}")
        print(f"      text_preview: {d.page_content[:200]!r}")
    print(f"{'='*80}\n")


def extract_numbers_and_key_terms(text):
    """
    Pulls out anything in the answer that should be independently
    verifiable against source text: dollar amounts, percentages,
    dates, and quoted numeric/duration phrases.
    """
    patterns = [
        r'\$[\d,]+(?:\.\d+)?',            # $2,500,000
        r'\b\d+(?:\.\d+)?\s?%',           # 7%
        r'\b\d+\s?(?:days?|months?|years?|hours?)\b',  # 20 days, 6 months
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text, flags=re.IGNORECASE))
    return found


def verify_grounding(query, answer_text, cited_chunk_texts):
    """
    Anti-hallucination gate using a secondary LLM verification step.
    Checks if the specific contextual concept extracted in the answer is explicitly supported by the cited chunks.
    """
    claims = extract_numbers_and_key_terms(answer_text)
    if not claims:
        return True, []  # nothing numeric to verify; don't block prose answers
        
    combined_source = "\n".join(cited_chunk_texts)
    
    # Fast-path: if the numbers aren't even IN the text, it's definitely hallucinated.
    def normalize(s):
        s = s.lower()
        replacements = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
        for w, d in replacements.items():
            s = s.replace(w, d)
        return re.sub(r'[,\s\$%]+', '', s)

    normalized_source = normalize(combined_source)
    unverified_lexical = []
    for claim in claims:
        if normalize(claim) not in normalized_source:
            unverified_lexical.append(claim)
            
    if unverified_lexical:
        return False, unverified_lexical

    # LLM Validation for context
    llm = get_llm()
    if not llm:
        return True, []

    prompt = f"""You are a strict Legal Verification System.
Task: Verify if the PROPOSED ANSWER accurately answers the QUERY based ONLY on the PROVIDED CONTEXT.

QUERY: {query}
PROPOSED ANSWER: {answer_text}

PROVIDED CONTEXT:
{combined_source}

Does the PROVIDED CONTEXT explicitly state the PROPOSED ANSWER to the QUERY?
Reply ONLY with "YES" or "NO". Do not explain."""

    try:
        response = llm.invoke(prompt).strip().upper()
        if "YES" in response and "NO" not in response:
            return True, []
        else:
            return False, ["Contextual Mismatch"]
    except Exception:
        return True, []


def get_citation_string(cid, retrieved_docs, ext_text=None, src_text=None):
    if not cid:
        return ""
    if cid.startswith("metadata_"):
        source_file = cid.replace("metadata_", "")
        return f" *(File: {source_file} | Section: Document Metadata)*"

    doc_for_chunk = next((d for d in retrieved_docs if d.metadata.get("chunk_id") == cid), None)
    if not doc_for_chunk:
        def clean_id(i):
            return i.replace("chunk_id_", "").replace("doc_", "").strip()
        doc_for_chunk = next(
            (d for d in retrieved_docs if clean_id(d.metadata.get("chunk_id", "")) == clean_id(cid)),
            None
        )

    if not doc_for_chunk:
        # CRITICAL: don't silently drop this — a citation to a chunk that
        # isn't in the retrieved set is itself a red flag worth surfacing.
        print(f"[CITATION WARNING] LLM cited chunk_id='{cid}' which is NOT in the retrieved set.")
        return " *(Citation unverified — cited chunk not found in retrieved context)*"

    meta = doc_for_chunk.metadata
    pr_str = meta.get("page_ranges", "[]")
    try:
        prs = json.loads(pr_str) if isinstance(pr_str, str) else list(pr_str)
    except Exception:
        prs = []

    if ext_text and src_text and prs:
        idx = src_text.find(ext_text)
        if idx != -1:
            lines_before = src_text[:idx].count('\n')
            lines_in = ext_text.strip().count('\n')
            base_start = prs[0].get("line_start", 0)
            prs = [dict(prs[0])]
            prs[0]["line_start"] = base_start + lines_before
            prs[0]["line_end"] = base_start + lines_before + lines_in

    range_strs = []
    for pr in prs:
        dp = pr.get("document_page")
        pp = pr.get("pdf_page")
        sl = pr.get("line_start")
        el = pr.get("line_end")
        p_str = f"Page: {dp}" if dp is not None else f"PDF Page: {pp}"
        l_str = f"Lines: {sl}-{el}" if sl != el else f"Line: {sl}"
        range_strs.append(f"{p_str}, {l_str}")

    ranges_formatted = " | ".join(range_strs)
    sec = meta.get("section", "Unknown")
    src = meta.get("source_file", "Unknown")
    return {"file": src, "section": sec, "ranges": ranges_formatted}


def extract_definition_from_chunk(term, text):
    esc_term = re.escape(term)
    p1 = re.compile(
        rf'(?:^|\n)[^\n]*?["\u201c\u201d]?{esc_term}["\u201c\u201d]?\s+'
        rf'(?:means|shall mean|is defined as|refers to).*?(?=\n\s*\d+\.\d+|\n\s*\n|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    m1 = p1.search(text)
    if m1:
        return re.sub(r'^\s*\d+\.\d+\.\d+\s*', '', m1.group(0).strip())

    p2 = re.compile(
        rf'(?:^|\n)[^\n]*?{esc_term}\s*[:\-]\s+.*?(?=\n\s*\d+\.\d+|\n\s*\n|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    m2 = p2.search(text)
    if m2:
        return m2.group(0).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        s_lower = s.lower()
        if term.lower() in s_lower and ("mean" in s_lower or "defin" in s_lower):
            return s.strip()
    return ""


def classify_answer_shape(query: str) -> str:
    """
    Decides whether a query wants a DEFINED TERM lookup or a FACTUAL answer
    (dollar amount, date, count, duration, party name, etc.).

    This is a hard, deterministic gate that runs BEFORE the LLM prompt is
    built — it stops factual questions ("how much", "how many", "when",
    "what is the amount") from ever being routed into the definitions
    extraction path, which is what was causing "upfront payment" and
    "public appearances" to come back as "I did not find a definition."
    """
    q = query.lower().strip()

    factual_signals = [
        "how much", "how many", "what amount", "what is the amount",
        "when is", "when does", "when must", "what date",
        "what percentage", "what rate", "what royalty",
        "who is", "who are", "which party", "how long",
        "what is the term", "what is the initial term",
    ]
    if any(sig in q for sig in factual_signals):
        return "factual"

    definition_signals = [
        "define ", "definition of", "what does", "meaning of",
        "how is", "term \"", "term '",
    ]
    # Only treat as a definition lookup if it ALSO looks like it's asking
    # about a capitalized/defined contractual term, not a plain fact.
    if any(sig in q for sig in definition_signals) and re.search(r'"[A-Z][a-zA-Z ]+"', query):
        return "definition"

    return "factual"  # default to factual — safer failure mode than silently
                       # dropping into "not found" for defined-term misses


def generate_answer(query, retrieved_docs, debug=True):
    llm = get_llm()
    if not llm:
        return "LLM not available to answer this query."

    if debug:
        log_retrieval(query, retrieved_docs)

    if not retrieved_docs:
        return "I could not retrieve any relevant contract text for this question."

    context_chunks = [format_chunk_context(doc) for doc in retrieved_docs]
    context = "\n\n".join(context_chunks)

    answer_shape = classify_answer_shape(query)

    if answer_shape == "definition":
        prompt = f"""You are an expert Legal Analyst extracting a DEFINED TERM.

Context:
{context}

The user wants the definition of a specific defined term used in this contract.

Question: {query}

RULES:
1. Find the chunk that explicitly defines this term (usually in a "Definitions" section).
2. Return ONLY the term name and the exact chunk_id where it's defined.
3. NEVER cite a chunk_id starting with "metadata_".
4. If no chunk explicitly defines the term, set status to NOT_FOUND.
5. Do NOT write the definition yourself — Python will extract it verbatim.

Output ONLY this JSON:
{{
  "definitions": [
    {{"term": "...", "status": "SUPPORTED or NOT_FOUND", "evidence_id": "chunk_id or empty"}}
  ]
}}
"""
    else:
        prompt = f"""You are an expert Legal Analyst. Answer using ONLY the provided contract context.

Context:
{context}

Question: {query}

CRITICAL RULES:
1. THE CONTEXT TEXT IS THE ONLY SOURCE OF TRUTH. Do not use outside knowledge.
2. ZERO HALLUCINATION: Every number, date, name, or amount in your answer MUST be copied
   directly from the context text above. If you cannot find a specific number/date/fact
   in the context, say so explicitly instead of guessing or estimating.
3. If the answer is not present in the context, respond exactly with:
   "I cannot find this information in the retrieved contract text."
   Do NOT invent a plausible-sounding number.
4. List every chunk_id you actually used as evidence in "supporting_chunks".
5. NEVER cite a chunk_id starting with "metadata_" as evidence for a fact.
6. You MAY perform basic arithmetic only if explicitly asked for a total, and only
   using numbers that appear verbatim in the context.

Output ONLY this JSON:
{{
  "answer": "Your answer text, in plain prose, no inline citation tags.",
  "supporting_chunks": ["chunk_id_1", "chunk_id_2"]
}}
"""

    try:
        with open("data/debug_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        raw_resp = llm.invoke(prompt).strip()

        if debug:
            print(f"\n[RAW LLM OUTPUT for query: {query}]\n{raw_resp}\n")

        start_idx = raw_resp.find('{')
        end_idx = raw_resp.rfind('}') + 1
        if start_idx == -1 or end_idx == -1:
            return raw_resp

        resp_dict = json.loads(raw_resp[start_idx:end_idx])

        # ---------- DEFINITION PATH ----------
        definitions = resp_dict.get("definitions", [])
        if answer_shape == "definition" and definitions:
            output = ""
            for item in definitions:
                term = item.get("term", "Unknown Term")
                status = item.get("status", "NOT_FOUND")
                evidence_id = item.get("evidence_id", "")

                if status != "SUPPORTED" or not evidence_id or evidence_id.startswith("metadata_"):
                    output += f"- **{term}:** I did not find a definition for this term in the retrieved contractual evidence.\n\n"
                    continue

                doc_for_chunk = next((d for d in retrieved_docs if d.metadata.get("chunk_id") == evidence_id), None)
                if not doc_for_chunk:
                    def clean_id(cid):
                        return cid.replace("chunk_id_", "").replace("doc_", "")
                    doc_for_chunk = next(
                        (d for d in retrieved_docs if clean_id(d.metadata.get("chunk_id", "")) == clean_id(evidence_id)),
                        None
                    )

                if not doc_for_chunk:
                    print(f"[HALLUCINATED CITATION] evidence_id='{evidence_id}' not in retrieved set for term '{term}'")
                    output += f"- **{term}:** I did not find a definition for this term in the retrieved contractual evidence.\n\n"
                    continue

                source_text = doc_for_chunk.page_content
                extracted_text = extract_definition_from_chunk(term, source_text)
                evidence_id = doc_for_chunk.metadata.get("chunk_id", evidence_id)

                if not extracted_text:
                    output += f"- **{term}:** [Extraction Error: Could not reliably parse the definition from chunk {evidence_id}.]\n\n"
                else:
                    cit = get_citation_string(evidence_id, retrieved_docs, ext_text=extracted_text, src_text=source_text)
                    cit_str = f" *(File: {cit['file']} | Section: {cit['section']} - {cit['ranges']})*" if isinstance(cit, dict) else cit
                    output += f"- **{term}:** {extracted_text}{cit_str}\n\n"
            return output.strip()

        # ---------- FACTUAL PATH ----------
        answer_text = resp_dict.get("answer", "")
        supporting_chunks = resp_dict.get("supporting_chunks", [])

        if not answer_text.strip():
            return "I cannot find this information in the retrieved contract text."

        if "cannot find this information" in answer_text.lower():
            return answer_text

        # Resolve cited chunks and pull their raw text for grounding verification
        cited_texts = []
        valid_citations = []
        for sc in supporting_chunks:
            sc_clean = sc.strip()
            if sc_clean.startswith("metadata_"):
                continue  # never trust metadata_ chunks as factual evidence
            doc_for_chunk = next((d for d in retrieved_docs if d.metadata.get("chunk_id") == sc_clean), None)
            if not doc_for_chunk:
                def clean_id(i):
                    return i.replace("chunk_id_", "").replace("doc_", "").strip()
                doc_for_chunk = next(
                    (d for d in retrieved_docs if clean_id(d.metadata.get("chunk_id", "")) == clean_id(sc_clean)),
                    None
                )
            if doc_for_chunk:
                cited_texts.append(doc_for_chunk.page_content)
                valid_citations.append(doc_for_chunk.metadata.get("chunk_id"))
            else:
                print(f"[HALLUCINATED CITATION] supporting_chunk='{sc_clean}' not in retrieved set.")

        # --- GROUNDING CHECK: the actual anti-hallucination gate ---
        is_grounded, unverified = verify_grounding(query, answer_text, cited_texts)
        if not is_grounded:
            print(f"[GROUNDING FAILURE] query={query!r} unverified_claims={unverified}")
            if "Contextual Mismatch" in unverified:
                answer_text = "I found related text, but it does not explicitly answer the question without making assumptions. Please rephrase the query or check the relevant sections directly."
            else:
                answer_text = (
                    "I found related contract text, but could not verify the specific figures "
                    f"in my draft answer ({', '.join(unverified)}) against the retrieved source text. "
                    "Rather than risk giving you an incorrect number, I'm flagging this for manual review. "
                    "Please rephrase the question or check the relevant section directly."
                )

        # Append citations (no inline tag dependency on the LLM)
        cits = []
        for cid in valid_citations:
            cit = get_citation_string(cid, retrieved_docs)
            if cit:
                cits.append(cit)
                
        if cits:
            # Consolidate citations by file and section
            consolidated = {}
            for c in cits:
                if isinstance(c, str): continue # Skip unverified string
                key = (c["file"], c["section"])
                if key not in consolidated:
                    consolidated[key] = set()
                # Split ranges by ' | ' and add individually to the set
                for r in c["ranges"].split(" | "):
                    consolidated[key].add(r.strip())
                    
            cit_strings = []
            for (file, sec), ranges_set in consolidated.items():
                def get_page_num(r):
                    m = re.search(r'\d+', r)
                    return int(m.group()) if m else 0
                sorted_ranges = sorted(list(ranges_set), key=get_page_num)
                ranges_str = " | ".join(sorted_ranges)
                cit_strings.append(f"(File: {file} | Section: {sec} - {ranges_str})")
                
            if cit_strings:
                if not is_grounded:
                    answer_text += " \n\n*The AI attempted to answer using the following sections, but failed verification:* " + " ".join(cit_strings)
                else:
                    answer_text += " *" + " ".join(cit_strings) + "*"

        # Escape dollar signs so Streamlit doesn't render them as LaTeX math blocks
        answer_text = answer_text.replace('$', r'\$')

        return answer_text

    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return f"Error generating answer: {str(e)}"

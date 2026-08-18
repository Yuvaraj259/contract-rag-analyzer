from src.query_parser import detect_operator, classify_query_intent, extract_search_terms

def test_detect_operator():
    assert detect_operator("find terms or conditions") == "OR"
    assert detect_operator("SLA and liability") == "AND"
    assert detect_operator("Find the SLA penalty") == "AND"

def test_classify_query_intent_no_llm():
    assert classify_query_intent("query", None) == "general_search"

def test_extract_search_terms_no_llm():
    assert extract_search_terms("SLA", None) == ["SLA"]

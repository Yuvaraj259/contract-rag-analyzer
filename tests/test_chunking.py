from src.chunking import split_into_sections, chunk_document

def test_split_into_sections():
    text = "General info\n\nLIMITATION OF LIABILITY\nLiability is limited.\n\nTERMINATION\nCan be terminated."
    sections = split_into_sections(text)
    
    assert "INTRO" in sections
    assert "LIMITATION OF LIABILITY" in sections
    assert "TERMINATION" in sections
    
def test_chunk_document():
    text = "Intro\n\nGOVERNING LAW\nCalifornia law applies."
    chunks = chunk_document(text, {"source_file": "test.pdf"})
    
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["source_file"] == "test.pdf"
    assert chunks[0]["metadata"]["section"] == "INTRO"
    assert chunks[1]["metadata"]["section"] == "GOVERNING LAW"

from src.chunking import chunk_resume, split_into_sections

def test_split_into_sections():
    text = "John Doe\n\nEXPERIENCE\nWorked as Dev for 3 years.\n\nSKILLS:\nPython, Java"
    sections = split_into_sections(text)
    
    assert "INTRO" in sections
    assert "EXPERIENCE" in sections
    assert "SKILLS" in sections
    
def test_chunk_resume():
    text = "John Doe\n\nEXPERIENCE\nWorked as Dev for 3 years.\n\nSKILLS:\nPython, Java"
    chunks = chunk_resume(text, {"source_file": "test.pdf"})
    
    assert len(chunks) == 3
    assert chunks[0]["metadata"]["source_file"] == "test.pdf"

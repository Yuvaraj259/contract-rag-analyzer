from src.resume_parser import extract_metadata, extract_email, extract_phone

def test_extract_email():
    text = "Contact me at test.email@example.com for more info."
    assert extract_email(text) == "test.email@example.com"
    
def test_extract_phone():
    text = "Phone: (123) 456-7890. Call me."
    assert extract_phone(text) == "(123) 456-7890"
    
def test_extract_metadata():
    text = "John Doe\njohn.doe@example.com\n123-456-7890\nSkills: Python"
    metadata = extract_metadata(text, "johndoe.pdf")
    
    assert metadata["email"] == "john.doe@example.com"
    assert metadata["phone"] == "123-456-7890"
    assert metadata["source_file"] == "johndoe.pdf"

import pytest
import os
from src.document_loader import load_document

def test_load_txt(tmp_path):
    # Create a temporary txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, this is a test resume.")
    
    text = load_document(str(test_file))
    assert "Hello, this is a test resume." in text

def test_unsupported_format():
    with pytest.raises(ValueError):
        load_document("test.xyz")

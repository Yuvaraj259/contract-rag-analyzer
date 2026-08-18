from src.contract_parser import extract_contract_title, extract_effective_date, extract_parties, extract_metadata

def test_extract_contract_title():
    text = "SOFTWARE LICENSE AGREEMENT\nThis is a contract."
    assert extract_contract_title(text) == "Software License Agreement"

def test_extract_effective_date():
    text = "This Agreement is dated as of January 1, 2024 by and between..."
    assert extract_effective_date(text) == "January 1, 2024"

def test_extract_parties():
    text = "This Agreement is by and between Acme Corp and Globex Inc."
    parties = extract_parties(text)
    assert "Acme Corp" in parties
    assert "Globex Inc." in parties

def test_extract_metadata():
    text = "SOFTWARE AGREEMENT\nDated as of March 15, 2023.\nBy and between Tech LLC and Client Corp."
    meta = extract_metadata(text, "contract.pdf")
    assert meta["contract_title"] == "Software Agreement"
    assert meta["effective_date"] == "March 15, 2023"
    assert "Tech LLC" in meta["parties"]
    assert meta["source_file"] == "contract.pdf"

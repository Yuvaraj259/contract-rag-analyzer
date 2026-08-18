from src.db_models import SessionLocal, DB_Contract, DB_DocumentMetadata, DB_Party, init_db
import hashlib

def get_file_hash(file_bytes):
    """Generate SHA-256 hash of file contents to detect duplicates."""
    return hashlib.sha256(file_bytes).hexdigest()

def is_duplicate_file(file_hash):
    """Check if a file with this hash has already been uploaded."""
    db = SessionLocal()
    try:
        exists = db.query(DB_DocumentMetadata).filter(DB_DocumentMetadata.file_hash == file_hash).first()
        return exists is not None
    finally:
        db.close()

def save_contract_metadata(metadata_dict, file_hash, original_filename):
    """Saves contract metadata and file tracking to PostgreSQL."""
    db = SessionLocal()
    try:
        title = metadata_dict.get("contract_title", "Unknown Contract")
        
        contract = DB_Contract(
            title=title,
            effective_date=metadata_dict.get("effective_date", "Unknown")
        )
        db.add(contract)
        db.commit()
        db.refresh(contract)
            
        # Save document metadata
        doc_meta = DB_DocumentMetadata(
            contract_id=contract.id,
            file_hash=file_hash,
            original_filename=original_filename
        )
        db.add(doc_meta)
        
        # Save Parties
        parties = metadata_dict.get("parties", [])
        for party in parties:
            db_party = DB_Party(contract_id=contract.id, party_name=party)
            db.add(db_party)
                
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"Error saving to PostgreSQL: {e}")
    finally:
        db.close()

def clear_postgres_db():
    """Clears all records from PostgreSQL."""
    db = SessionLocal()
    try:
        db.query(DB_DocumentMetadata).delete()
        db.query(DB_Party).delete()
        db.query(DB_Contract).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error clearing PostgreSQL: {e}")
    finally:
        db.close()

def get_contract_info(filenames: list) -> list:
    """Returns a list of dicts with title, date, filename, and parties for given files."""
    if not filenames:
        return []
    db = SessionLocal()
    try:
        results = db.query(DB_Contract, DB_DocumentMetadata.original_filename)\
            .join(DB_DocumentMetadata, DB_Contract.id == DB_DocumentMetadata.contract_id)\
            .filter(DB_DocumentMetadata.original_filename.in_(filenames))\
            .all()
            
        info = []
        for contract, filename in results:
            parties = [p.party_name for p in contract.parties]
            info.append({
                "title": contract.title,
                "effective_date": contract.effective_date,
                "parties": parties,
                "file": filename
            })
            
        return info
    finally:
        db.close()

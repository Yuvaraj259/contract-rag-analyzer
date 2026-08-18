import os
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/contract_rag")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DB_Contract(Base):
    __tablename__ = "contracts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, index=True)
    effective_date = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    parties = relationship("DB_Party", back_populates="contract", cascade="all, delete-orphan")
    documents = relationship("DB_DocumentMetadata", back_populates="contract", cascade="all, delete-orphan")

class DB_Party(Base):
    __tablename__ = "parties"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    contract_id = Column(String, ForeignKey("contracts.id"))
    party_name = Column(String, index=True)
    
    contract = relationship("DB_Contract", back_populates="parties")

class DB_DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    contract_id = Column(String, ForeignKey("contracts.id"))
    file_hash = Column(String, unique=True, index=True)
    original_filename = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    contract = relationship("DB_Contract", back_populates="documents")

def init_db():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

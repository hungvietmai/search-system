"""
Database models for Leaf Search System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class LeafImage(Base):
    """Model for storing leaf image metadata"""
    __tablename__ = "leaf_images"
    
    # Use file_id as primary key - eliminates confusion with dual IDs
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String(500), nullable=False, unique=True)
    segmented_path = Column(String(500))
    species = Column(String(200), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)  # 'lab' or 'field'
    
    # Vector database references
    milvus_id = Column(Integer, unique=True, index=True, nullable=True)
    faiss_id = Column(Integer, unique=True, index=True, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    feedback_records = relationship("UserFeedback", back_populates="leaf_image", cascade="all, delete-orphan")
    
    # Table-level constraints
    __table_args__ = (
        # Composite indexes for common query patterns
        Index('idx_species_source', 'species', 'source'),
        Index('idx_species_created', 'species', 'created_at'),
        Index('idx_source_created', 'source', 'created_at'),
        Index('idx_milvus_lookup', 'milvus_id', 'file_id'),
        Index('idx_faiss_lookup', 'faiss_id', 'file_id'),
        
        # Check constraints for data integrity
        CheckConstraint("source IN ('lab', 'field')", name='check_valid_source'),
        CheckConstraint("species != ''", name='check_species_not_empty'),
    )
    
    def __repr__(self):
        return f"<LeafImage(file_id={self.file_id}, species={self.species})>"


class SearchHistory(Base):
    """Model for storing search history"""
    __tablename__ = "search_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_image_path = Column(String(500), nullable=False)
    search_engine = Column(String(50), nullable=False, index=True)  # 'milvus' or 'faiss'
    top_k = Column(Integer, nullable=False)
    search_time_ms = Column(Float, nullable=False)
    results = Column(Text, nullable=False)  # JSON string of results
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Table-level constraints and indexes
    __table_args__ = (
        # Composite indexes for analytics queries
        Index('idx_engine_created', 'search_engine', 'created_at'),
        Index('idx_created_engine', 'created_at', 'search_engine'),
        
        # Check constraints
        CheckConstraint("search_engine IN ('milvus', 'faiss')", name='check_valid_search_engine'),
        CheckConstraint("top_k > 0", name='check_positive_top_k'),
        CheckConstraint("search_time_ms >= 0", name='check_non_negative_search_time'),
    )
    
    def __repr__(self):
        return f"<SearchHistory(id={self.id}, engine={self.search_engine})>"


class UserFeedback(Base):
    """Model for storing user feedback on search results"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Query information
    query_image_hash = Column(String(64), nullable=False, index=True)  # Hash of query image
    query_timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    session_id = Column(String(100), index=True, nullable=True)
    
    # Result information - with foreign key constraint
    result_file_id = Column(Integer, ForeignKey('leaf_images.file_id', ondelete='CASCADE'), nullable=False, index=True)
    result_position = Column(Integer, nullable=False)  # Position in results (1-based)
    
    # Feedback
    feedback_type = Column(String(20), nullable=False, index=True)  # relevant, irrelevant, clicked, etc.
    confidence = Column(Float, nullable=False, default=1.0)
    dwell_time = Column(Float, nullable=True)  # Seconds
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Relationships
    leaf_image = relationship("LeafImage", back_populates="feedback_records")
    
    # Table-level constraints and indexes
    __table_args__ = (
        # Composite indexes for common query patterns
        Index('idx_query_hash_created', 'query_image_hash', 'created_at'),
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_file_feedback_type', 'result_file_id', 'feedback_type'),
        Index('idx_feedback_type_created', 'feedback_type', 'created_at'),
        
        # Composite index for deduplication checks
        Index('idx_unique_feedback', 'query_image_hash', 'result_file_id', 'session_id', 'created_at'),
        
        # Check constraints for data integrity
        CheckConstraint("feedback_type IN ('relevant', 'irrelevant', 'clicked', 'selected', 'dismissed')", 
                       name='check_valid_feedback_type'),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name='check_confidence_range'),
        CheckConstraint("result_position > 0", name='check_positive_position'),
        CheckConstraint("dwell_time IS NULL OR dwell_time >= 0", name='check_non_negative_dwell_time'),
    )
    
    def __repr__(self):
        return f"<UserFeedback(id={self.id}, type={self.feedback_type}, file_id={self.result_file_id})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'query_image_hash': self.query_image_hash,
            'query_timestamp': self.query_timestamp.isoformat() if self.query_timestamp is not None else None,
            'session_id': self.session_id,
            'result_file_id': self.result_file_id,
            'result_position': self.result_position,
            'feedback_type': self.feedback_type,
            'confidence': self.confidence,
            'dwell_time': self.dwell_time,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None
        }



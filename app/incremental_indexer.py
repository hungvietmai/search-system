"""
Incremental Indexing System
Add/update/delete entries without rebuilding the entire index

Features:
- Add new images incrementally
- Update existing entries
- Delete entries
- Batch updates
- Index synchronization
- Change tracking
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
from sqlalchemy.orm import Session

from app.models import LeafImage

logger = logging.getLogger(__name__)


class IndexOperation(Enum):
    """Index operation types"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    SYNC = "sync"


@dataclass
class IndexChange:
    """Record of an index change"""
    operation: IndexOperation
    file_id: int
    species: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class IncrementalIndexer:
    """
    Incremental indexing manager
    Handles adding, updating, and deleting entries without full rebuild
    """
    
    def __init__(self,
                 milvus_client=None,
                 faiss_client=None,
                 feature_extractor=None):
        """
        Initialize incremental indexer
        
        Args:
            milvus_client: Milvus client instance
            faiss_client: Faiss client instance
            feature_extractor: Feature extractor instance
        """
        self.milvus_client = milvus_client
        self.faiss_client = faiss_client
        self.feature_extractor = feature_extractor
        self.change_log: List[IndexChange] = []
        
        logger.info("Incremental indexer initialized")
    
    def add_single(self,
                   file_id: int,
                   image_path: str,
                   species: str,
                   source: str,
                   segmented_path: Optional[str],
                   db: Session) -> IndexChange:
        """
        Add a single image to the index
        
        Args:
            file_id: File ID
            image_path: Path to image
            species: Species name
            source: Image source
            segmented_path: Path to segmented image
            db: Database session
            
        Returns:
            IndexChange record
        """
        change = IndexChange(
            operation=IndexOperation.ADD,
            file_id=file_id,
            species=species,
            image_path=image_path
        )
        
        try:
            # Extract features
            if not self.feature_extractor:
                raise ValueError("Feature extractor not initialized")
            
            logger.info(f"Extracting features for {image_path}")
            features = self.feature_extractor.extract_features(image_path)
            
            # Add to Milvus
            milvus_id = None
            if self.milvus_client:
                try:
                    # Check if already exists
                    existing = db.query(LeafImage).filter(
                        LeafImage.file_id == file_id
                    ).first()
                    
                    if existing and existing.milvus_id is not None:
                        # Delete old entry first
                        self.milvus_client.delete([file_id])
                    
                    # Insert new entry
                    milvus_id = self.milvus_client.insert_single(file_id, features)
                    logger.info(f"Added to Milvus: {milvus_id}")
                except Exception as e:
                    logger.warning(f"Milvus indexing failed: {e}")
            
            # Add to Faiss
            faiss_id = None
            if self.faiss_client:
                try:
                    faiss_id = self.faiss_client.add_single(file_id, features)
                    logger.info(f"Added to Faiss: {faiss_id}")
                except Exception as e:
                    logger.warning(f"Faiss indexing failed: {e}")
            
            # Update database
            existing = db.query(LeafImage).filter(
                LeafImage.file_id == file_id
            ).first()
            
            if existing:
                # Update existing entry
                existing.image_path = image_path  # type: ignore
                existing.species = species  # type: ignore
                existing.source = source  # type: ignore
                existing.segmented_path = segmented_path  # type: ignore
                if milvus_id is not None:
                    existing.milvus_id = milvus_id
                if faiss_id is not None:
                    existing.faiss_id = faiss_id
                existing.updated_at = datetime.now(timezone.utc)  # type: ignore
            else:
                # Create new entry
                new_image = LeafImage(
                    file_id=file_id,
                    image_path=image_path,
                    species=species,
                    source=source,
                    segmented_path=segmented_path,
                    milvus_id=milvus_id,
                    faiss_id=faiss_id
                )
                db.add(new_image)
            
            db.commit()
            
            change.success = True
            logger.info(f"Successfully added {file_id} incrementally")
            
        except Exception as e:
            logger.error(f"Failed to add {file_id}: {e}")
            change.success = False
            change.error_message = str(e)
            db.rollback()
        
        self.change_log.append(change)
        return change
    
    def update_single(self,
                     file_id: int,
                     new_image_path: Optional[str] = None,
                     new_species: Optional[str] = None,
                     db: Optional[Session] = None) -> IndexChange:
        """
        Update an existing index entry
        
        Args:
            file_id: File ID to update
            new_image_path: New image path (if changed)
            new_species: New species name (if changed)
            db: Database session
            
        Returns:
            IndexChange record
        """
        change = IndexChange(
            operation=IndexOperation.UPDATE,
            file_id=file_id,
            species=new_species,
            image_path=new_image_path
        )
        
        try:
            if db is None:
                raise ValueError("Database session not provided")
                
            # Get existing entry
            existing = db.query(LeafImage).filter(
                LeafImage.file_id == file_id
            ).first()
            
            if not existing:
                raise ValueError(f"File ID {file_id} not found")
            
            # If image path changed, re-extract features
            if new_image_path and new_image_path != existing.image_path:
                if not self.feature_extractor:
                    raise ValueError("Feature extractor not initialized")
                logger.info(f"Re-extracting features for updated image {file_id}")
                features = self.feature_extractor.extract_features(new_image_path)
                
                # Update in Milvus
                if self.milvus_client and existing.milvus_id is not None:
                    # Delete old
                    self.milvus_client.delete([file_id])
                    # Add new
                    milvus_id = self.milvus_client.insert_single(file_id, features)
                    existing.milvus_id = milvus_id
                
                # Update in Faiss (requires rebuild for updates)
                # Note: Faiss doesn't support efficient updates
                if self.faiss_client:
                    logger.warning("Faiss update requires index rebuild")
                
                existing.image_path = new_image_path  # type: ignore
            
            # Update species if changed
            if new_species:
                existing.species = new_species  # type: ignore
            
            existing.updated_at = datetime.now(timezone.utc)  # type: ignore
            db.commit()
            
            change.success = True
            logger.info(f"Successfully updated {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to update {file_id}: {e}")
            change.success = False
            change.error_message = str(e)
            if db is not None:
                db.rollback()
        
        self.change_log.append(change)
        return change
    
    def delete_single(self, file_id: int, db: Session) -> IndexChange:
        """
        Delete an entry from the index
        
        Args:
            file_id: File ID to delete
            db: Database session
            
        Returns:
            IndexChange record
        """
        change = IndexChange(
            operation=IndexOperation.DELETE,
            file_id=file_id
        )
        
        try:
            # Get entry
            existing = db.query(LeafImage).filter(
                LeafImage.file_id == file_id
            ).first()
            
            if not existing:
                raise ValueError(f"File ID {file_id} not found")
            
            # Delete from Milvus
            if self.milvus_client and existing.milvus_id is not None:
                try:
                    self.milvus_client.delete([file_id])
                    logger.info(f"Deleted from Milvus: {file_id}")
                except Exception as e:
                    logger.warning(f"Milvus deletion failed: {e}")
            
            # Note: Faiss doesn't support efficient deletion
            if self.faiss_client and existing.faiss_id is not None:
                logger.warning("Faiss deletion requires index rebuild")
            
            # Delete from database
            db.delete(existing)
            db.commit()
            
            change.success = True
            logger.info(f"Successfully deleted {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete {file_id}: {e}")
            change.success = False
            change.error_message = str(e)
            db.rollback()
        
        self.change_log.append(change)
        return change
    
    def add_batch(self,
                  entries: List[Dict],
                  db: Session) -> List[IndexChange]:
        """
        Add multiple entries incrementally
        
        Args:
            entries: List of entry dictionaries with:
                    - file_id
                    - image_path
                    - species
                    - source
                    - segmented_path (optional)
            db: Database session
            
        Returns:
            List of IndexChange records
        """
        changes = []
        
        logger.info(f"Adding {len(entries)} entries incrementally")
        
        for entry in entries:
            change = self.add_single(
                file_id=entry['file_id'],
                image_path=entry['image_path'],
                species=entry['species'],
                source=entry['source'],
                segmented_path=entry.get('segmented_path'),
                db=db
            )
            changes.append(change)
        
        success_count = sum(1 for c in changes if c.success)
        logger.info(f"Batch add complete: {success_count}/{len(entries)} successful")
        
        return changes
    
    def synchronize_index(self, db: Session) -> Dict[str, int]:
        """
        Synchronize index with database
        Finds and fixes inconsistencies
        
        Args:
            db: Database session
            
        Returns:
            Statistics dictionary
        """
        logger.info("Starting index synchronization")
        
        stats = {
            'total_db_entries': 0,
            'missing_in_milvus': 0,
            'missing_in_faiss': 0,
            'reindexed': 0,
            'errors': 0
        }
        
        # Get all database entries
        all_images = db.query(LeafImage).all()
        stats['total_db_entries'] = len(all_images)
        
        for image in all_images:
            try:
                # Check Milvus
                if self.milvus_client and image.milvus_id is None:
                    stats['missing_in_milvus'] += 1
                    # Re-index
                    if not self.feature_extractor:
                        raise ValueError("Feature extractor not initialized")
                    features = self.feature_extractor.extract_features(image.image_path)
                    milvus_id = self.milvus_client.insert_single(image.file_id, features)
                    image.milvus_id = milvus_id
                    stats['reindexed'] += 1
                
                # Check Faiss
                if self.faiss_client and image.faiss_id is None:
                    stats['missing_in_faiss'] += 1
                    # Note: Faiss sync requires rebuild
                
                db.commit()
                
            except Exception as e:
                logger.error(f"Sync error for {image.file_id}: {e}")
                stats['errors'] += 1
                db.rollback()
        
        logger.info(f"Synchronization complete: {stats}")
        return stats
    
    def get_change_log(self,
                      operation: Optional[IndexOperation] = None,
                      limit: int = 100) -> List[IndexChange]:
        """
        Get change log
        
        Args:
            operation: Filter by operation type
            limit: Maximum number of changes to return
            
        Returns:
            List of changes
        """
        changes = self.change_log
        
        if operation:
            changes = [c for c in changes if c.operation == operation]
        
        # Return most recent first
        changes = sorted(changes, key=lambda c: c.timestamp or datetime.min, reverse=True)
        
        return changes[:limit]
    
    def get_statistics(self) -> Dict:
        """Get indexing statistics"""
        total = len(self.change_log)
        
        return {
            'total_operations': total,
            'successful': sum(1 for c in self.change_log if c.success),
            'failed': sum(1 for c in self.change_log if not c.success),
            'adds': sum(1 for c in self.change_log if c.operation == IndexOperation.ADD),
            'updates': sum(1 for c in self.change_log if c.operation == IndexOperation.UPDATE),
            'deletes': sum(1 for c in self.change_log if c.operation == IndexOperation.DELETE),
            'recent_changes': [
                {
                    'operation': c.operation.value,
                    'file_id': c.file_id,
                    'timestamp': c.timestamp.isoformat() if c.timestamp else None,
                    'success': c.success
                }
                for c in self.get_change_log(limit=10)
            ]
        }


# Global incremental indexer
_incremental_indexer = None


def get_incremental_indexer() -> IncrementalIndexer:
    """
    Get or create global incremental indexer
    
    Returns:
        IncrementalIndexer instance
    """
    global _incremental_indexer
    
    if _incremental_indexer is None:
        from app.feature_extractor import get_feature_extractor
        from app.milvus_client import get_milvus_client
        from app.faiss_client import get_faiss_client
        
        _incremental_indexer = IncrementalIndexer(
            milvus_client=get_milvus_client(),
            faiss_client=get_faiss_client(),
            feature_extractor=get_feature_extractor()
        )
    
    return _incremental_indexer


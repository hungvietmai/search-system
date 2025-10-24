"""
Data ingestion script for Leafsnap dataset
Extracts features using ResNet-50 and indexes them in Milvus and Faiss
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Optional

from config import settings
from app.database import init_db, get_db_context
from app.models import LeafImage
from app.feature_extractor import get_feature_extractor
from app.milvus_client import get_milvus_client
from app.faiss_client import get_faiss_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load dataset metadata from the leafsnap dataset file
    
    Args:
        metadata_path: Path to the metadata file
        
    Returns:
        DataFrame with dataset information
    """
    logger.info(f"Loading dataset metadata from {metadata_path}")
    
    try:
        df = pd.read_csv(metadata_path, sep='\t')
        logger.info(f"Loaded {len(df)} entries from dataset")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Species count: {df['species'].nunique()}")
        logger.info(f"Lab images: {len(df[df['source'] == 'lab'])}")
        logger.info(f"Field images: {len(df[df['source'] == 'field'])}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset metadata: {e}")
        raise


def verify_image_paths(df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """
    Verify that image paths exist
    
    Args:
        df: DataFrame with image paths
        base_path: Base path for the dataset (project root)
        
    Returns:
        Filtered DataFrame with only existing images
    """
    logger.info("Verifying image paths...")
    
    # Paths in metadata are already relative to project root
    existing_mask = df['image_path'].apply(lambda x: Path(x).exists())
    missing_count = int((~existing_mask).sum())
    
    if missing_count > 0:
        logger.warning(f"{missing_count} images not found on disk")
    
    result = df[existing_mask].copy()
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Expected DataFrame after filtering")
    logger.info(f"{len(result)} images verified")
    
    return result


def ingest_to_database(df: pd.DataFrame) -> Dict[int, int]:
    """
    Ingest metadata into SQLite database
    
    Args:
        df: DataFrame with image metadata
        
    Returns:
        Mapping of file_id to database id
    """
    logger.info("Ingesting metadata to database...")
    
    file_id_to_db_id = {}
    
    with get_db_context() as db:
        # Clear existing data if requested
        # db.query(LeafImage).delete()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Database ingestion"):
            try:
                # Check if already exists
                existing = db.query(LeafImage).filter(
                    LeafImage.file_id == row['file_id']
                ).first()
                
                if existing:
                    file_id_to_db_id[row['file_id']] = existing.file_id
                    continue
                
                # Create new entry
                # Extract scalar values from Series to avoid type issues
                file_id_val = int(row['file_id'])
                image_path_val = str(row['image_path'])
                species_val = str(row['species'])
                source_val = str(row['source'])
                
                # Handle segmented_path which can be None/NaN
                seg_path_raw = row['segmented_path']
                seg_path = None
                if seg_path_raw is not None and str(seg_path_raw) != 'nan':
                    seg_path = str(seg_path_raw)
                
                leaf_image = LeafImage(
                    file_id=file_id_val,
                    image_path=image_path_val,
                    segmented_path=seg_path,
                    species=species_val,
                    source=source_val
                )
                
                db.add(leaf_image)
                db.flush()
                
                file_id_to_db_id[row['file_id']] = leaf_image.file_id
                
            except Exception as e:
                logger.error(f"Failed to insert {row['file_id']}: {e}")
                continue
        
        db.commit()
    
    logger.info(f"Ingested {len(file_id_to_db_id)} entries to database")
    return file_id_to_db_id


def extract_and_index_features(df: pd.DataFrame, 
                               use_milvus: bool = True,
                               use_faiss: bool = True,
                               batch_size: Optional[int] = None):
    """
    Extract features and index in vector databases
    
    Args:
        df: DataFrame with image metadata
        use_milvus: Whether to index in Milvus
        use_faiss: Whether to index in Faiss
        batch_size: Batch size for feature extraction
    """
    if batch_size is None:
        batch_size = settings.batch_size
    
    # Paths in metadata are already relative to project root
    # No need to use base_path since paths are already correct
    
    # Initialize feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = get_feature_extractor()
    
    # Initialize Milvus
    milvus_client = None
    if use_milvus:
        try:
            logger.info("Initializing Milvus client...")
            temp_client = get_milvus_client()
            if temp_client is not None:
                # Create collection (drop if exists for fresh start)
                temp_client.create_collection(drop_existing=False)
                temp_client.load_collection()
                milvus_client = temp_client 
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            logger.warning("Continuing without Milvus indexing")
            milvus_client = None
    
    # Initialize Faiss
    faiss_client = None
    if use_faiss:
        try:
            logger.info("Initializing Faiss client...")
            faiss_client = get_faiss_client()
        except Exception as e:
            logger.error(f"Failed to initialize Faiss: {e}")
            logger.warning("Continuing without Faiss indexing")
            faiss_client = None
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    with get_db_context() as db:
        for batch_idx in tqdm(range(total_batches), desc="Feature extraction"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Prepare image paths
            image_paths = []
            file_ids = []
            db_ids = []
            
            for idx, row in batch_df.iterrows():
                # Paths in metadata are already relative to project root
                image_path = Path(row['image_path'])
                if image_path.exists():
                    image_paths.append(str(image_path))
                    file_ids.append(row['file_id'])
                    
                    # Get database ID (file_id is the primary key)
                    leaf = db.query(LeafImage).filter(
                        LeafImage.file_id == row['file_id']
                    ).first()
                    if leaf:
                        db_ids.append(leaf.file_id)
            
            if not image_paths:
                continue
            
            try:
                # Extract features
                features = feature_extractor.extract_features_batch(image_paths, batch_size=len(image_paths))
                
                if len(features) == 0:
                    continue
                
                # Index in Milvus
                if milvus_client:
                    try:
                        milvus_ids = milvus_client.insert(file_ids[:len(features)], features)
                        
                        # Update database with Milvus IDs
                        for i, (db_file_id, milvus_id) in enumerate(zip(db_ids[:len(features)], milvus_ids)):
                            db.query(LeafImage).filter(LeafImage.file_id == db_file_id).update(
                                {"milvus_id": milvus_id}
                            )
                        db.commit()
                        
                    except Exception as e:
                        logger.error(f"Failed to index batch in Milvus: {e}")
                
                # Index in Faiss
                if faiss_client:
                    try:
                        faiss_client.add(file_ids[:len(features)], features)
                        
                        # Update database with Faiss IDs
                        # For Faiss, the index is the position in the index
                        current_count = faiss_client.get_count() - len(features)
                        for i, db_file_id in enumerate(db_ids[:len(features)]):
                            db.query(LeafImage).filter(LeafImage.file_id == db_file_id).update(
                                {"faiss_id": current_count + i}
                            )
                        db.commit()
                        
                    except Exception as e:
                        logger.error(f"Failed to index batch in Faiss: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                continue
    
    # Save Faiss index
    if faiss_client:
        logger.info("Saving Faiss index...")
        faiss_client.save()
    
    logger.info("Feature extraction and indexing complete!")


def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest Leafsnap dataset")
    parser.add_argument("--metadata", type=str, default=settings.dataset_metadata,
                       help="Path to metadata file")
    parser.add_argument("--batch-size", type=int, default=settings.batch_size,
                       help="Batch size for processing")
    parser.add_argument("--skip-milvus", action="store_true",
                       help="Skip Milvus indexing")
    parser.add_argument("--skip-faiss", action="store_true",
                       help="Skip Faiss indexing")
    parser.add_argument("--reset-db", action="store_true",
                       help="Reset database before ingestion")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Reset database if requested
    if args.reset_db:
        logger.warning("Resetting database...")
        with get_db_context() as db:
            db.query(LeafImage).delete()
            db.commit()
    
    # Load dataset metadata
    df = load_dataset_metadata(args.metadata)
    
    # Limit if requested
    if args.limit:
        logger.info(f"Limiting to {args.limit} images for testing")
        df = df.head(args.limit)
    
    # Verify image paths
    df = verify_image_paths(df, Path(args.metadata))
    
    # Ingest to database
    ingest_to_database(df)
    
    # Extract features and index
    extract_and_index_features(
        df,
        use_milvus=not args.skip_milvus,
        use_faiss=not args.skip_faiss,
        batch_size=args.batch_size
    )
    
    logger.info("Ingestion complete!")
    
    # Print summary
    with get_db_context() as db:
        total = db.query(LeafImage).count()
        with_milvus = db.query(LeafImage).filter(LeafImage.milvus_id.isnot(None)).count()
        with_faiss = db.query(LeafImage).filter(LeafImage.faiss_id.isnot(None)).count()
        
        logger.info(f"\nSummary:")
        logger.info(f"  Total images in DB: {total}")
        logger.info(f"  Indexed in Milvus: {with_milvus}")
        logger.info(f"  Indexed in Faiss: {with_faiss}")


if __name__ == "__main__":
    main()


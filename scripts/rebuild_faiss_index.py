"""
Rebuild FAISS index with a specific similarity metric

This script rebuilds the FAISS index from the existing database
with the specified similarity metric (cosine, l2, angular, etc.)

Usage:
    python scripts/rebuild_faiss_index.py --metric cosine
    python scripts/rebuild_faiss_index.py --metric l2
    python scripts/rebuild_faiss_index.py --metric angular
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
from tqdm import tqdm

from config import settings
from app.database import get_db_context
from app.models import LeafImage
from app.feature_extractor import get_feature_extractor
from app.faiss_client import FaissClient
from app.similarity_metrics import SimilarityMetric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rebuild_index(metric: SimilarityMetric, batch_size: int = 32):
    """
    Rebuild FAISS index with specified metric
    
    Args:
        metric: Similarity metric to use
        batch_size: Batch size for processing
    """
    logger.info(f"Rebuilding FAISS index with metric: {metric.value}")
    
    # Delete old index files to ensure clean rebuild
    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.faiss_metadata_path)
    
    if index_path.exists():
        logger.info(f"Deleting old index: {index_path}")
        index_path.unlink()
    
    if metadata_path.exists():
        logger.info(f"Deleting old metadata: {metadata_path}")
        metadata_path.unlink()
    
    # Create new FAISS client with specified metric
    logger.info(f"Creating new FAISS index with {metric.value} metric...")
    faiss_client = FaissClient(
        dimension=settings.feature_dim,
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
        metric=metric
    )
    
    # Get feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = get_feature_extractor()
    
    # Get all images from database
    with get_db_context() as db:
        images = db.query(LeafImage).order_by(LeafImage.file_id).all()
        logger.info(f"Found {len(images)} images in database")
        
        if len(images) == 0:
            logger.error("No images found in database!")
            return
        
        # Process in batches
        total_batches = (len(images) + batch_size - 1) // batch_size
        total_indexed = 0
        
        for batch_idx in tqdm(range(total_batches), desc="Indexing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx]
            
            # Collect file IDs and paths
            file_ids = []
            image_paths = []
            
            for img in batch_images:
                if img.image_path is not None:
                    file_ids.append(img.file_id)
                    # Use absolute path
                    abs_path = Path(settings.dataset_path).parent / img.image_path
                    image_paths.append(str(abs_path))
            
            if not image_paths:
                continue
            
            try:
                # Extract features for batch
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(image_paths)} images)")
                features = feature_extractor.extract_features_batch(
                    image_paths, 
                    batch_size=len(image_paths)
                )
                
                if len(features) == 0:
                    logger.warning(f"No features extracted for batch {batch_idx}")
                    continue
                
                # Add to FAISS index
                faiss_client.add(file_ids[:len(features)], features)
                total_indexed += len(features)
                
                # Update database with FAISS IDs
                current_count = faiss_client.get_count() - len(features)
                for i, img in enumerate(batch_images[:len(features)]):
                    img.faiss_id = current_count + i
                
                db.commit()
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                continue
    
    # Save index
    logger.info("Saving FAISS index...")
    faiss_client.save()
    
    logger.info(f"✅ Index rebuilt successfully!")
    logger.info(f"   Total images indexed: {total_indexed}")
    logger.info(f"   Metric: {metric.value}")
    logger.info(f"   Index path: {settings.faiss_index_path}")
    logger.info(f"   Metadata path: {settings.faiss_metadata_path}")
    
    # Verify index
    health = faiss_client.check_health()
    logger.info(f"   Index health: {health}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index with specific similarity metric"
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["cosine", "inner_product", "l2", "l1", "angular"],
        help="Similarity metric to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Convert string to enum
    metric = SimilarityMetric(args.metric)
    
    # Confirm with user
    print("\n" + "="*60)
    print("FAISS Index Rebuild")
    print("="*60)
    print(f"Current index: {settings.faiss_index_path}")
    print(f"New metric: {metric.value}")
    print(f"This will OVERWRITE the existing index!")
    print("="*60)
    
    if not args.yes:
        response = input("\nProceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    else:
        print("\nProceeding with rebuild (--yes flag)...")
    
    # Rebuild index
    rebuild_index(metric, args.batch_size)
    
    print("\n" + "="*60)
    print("✅ FAISS Index Rebuilt Successfully!")
    print("="*60)
    print(f"You can now search with metric: {metric.value}")
    print("\nExample curl command:")
    print(f"  curl -X 'POST' \\")
    print(f"    'http://127.0.0.1:8000/search?similarity_metric={metric.value}' \\")
    print(f"    -H 'accept: application/json' \\")
    print(f"    -H 'Content-Type: multipart/form-data' \\")
    print(f"    -F 'file=@your_image.jpg;type=image/jpeg'")
    print("="*60)


if __name__ == "__main__":
    main()


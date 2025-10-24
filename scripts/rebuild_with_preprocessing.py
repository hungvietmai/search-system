"""
Rebuild FAISS index WITH preprocessing enabled
Optimized with parallel DataLoader and GPU streaming
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from app.database import get_db_context
from app.models import LeafImage
from app.faiss_client import FaissClient
from app.similarity_metrics import SimilarityMetric
from app.preprocessors import PreprocessingProfile, AdvancedLeafPreprocessor
from config import settings
import torchvision.models as models
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeafDataset(Dataset):
    """Dataset for parallel image loading"""
    
    def __init__(self, image_data, use_preprocessing=False, adaptive_profile=True):
        """
        Args:
            image_data: List of (image_path, source, file_id) tuples
            use_preprocessing: Whether to apply preprocessing
            adaptive_profile: Use LAB/FIELD profiles based on source
        """
        self.image_data = image_data
        self.use_preprocessing = use_preprocessing
        self.adaptive_profile = adaptive_profile
        
        # Don't initialize preprocessor here - it contains non-picklable OpenCV objects
        # Will be created lazily in each worker process
        self._preprocessor = None
        
        # Standard transforms for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image_path, source, file_id = self.image_data[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                # Lazy initialization of preprocessor in each worker
                if self._preprocessor is None:
                    self._preprocessor = AdvancedLeafPreprocessor()
                
                if self.adaptive_profile:
                    profile = PreprocessingProfile.LAB if source == 'lab' else PreprocessingProfile.FIELD
                else:
                    profile = PreprocessingProfile.AUTO
                image = self._preprocessor.preprocess(image, profile=profile)
            
            # Transform to tensor
            image_tensor = self.transform(image)
            
            return image_tensor, file_id
            
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            # Return dummy tensor and invalid file_id
            return torch.zeros((3, 224, 224)), -1


def rebuild_index_with_preprocessing(
    metric: SimilarityMetric = SimilarityMetric.COSINE,
    batch_size: int = 128,  # Larger batches for better GPU utilization
    adaptive_profile: bool = True,
    use_advanced: bool = True,
    num_workers: int = 6  # More workers for faster preprocessing
):
    """
    Rebuild FAISS index with parallel processing and advanced preprocessing
    
    Args:
        metric: Similarity metric to use
        batch_size: Batch size for processing (smaller for preprocessing)
        adaptive_profile: Use adaptive preprocessing
        use_advanced: Use advanced preprocessing (ALWAYS ENABLED)
        num_workers: Number of DataLoader workers (4-6 recommended for preprocessing)
    """
    print("=" * 80)
    print("REBUILDING FAISS INDEX (WITH ADVANCED PREPROCESSING)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Metric: {metric.value}")
    print(f"  Advanced Preprocessing: ENABLED (required)")
    print(f"  Profile Mode: {'ADAPTIVE (Field + Lab)' if adaptive_profile else 'Fixed'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Parallel workers: {num_workers} (for faster preprocessing)")
    print()
    
    # Get device and configure for maximum GPU utilization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Enable TF32 for faster computation on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set cudnn benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        logger.info(f"GPU optimizations enabled: TF32, cuDNN benchmark")
    else:
        logger.info(f"Using device: {device}")
    
    # Load ResNet-50 model
    logger.info("Loading ResNet-50 model...")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Get all images from database
    logger.info("Loading image data from database...")
    with get_db_context() as db:
        images = db.query(LeafImage).all()
        # Extract data while session is active
        image_data = [(img.image_path, img.source, img.file_id) for img in images]
        total = len(image_data)
    
    logger.info(f"Found {total} images to index")
    
    # Delete old index files
    logger.info(f"Creating new FAISS index with metric: {metric.value}")
    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.faiss_metadata_path)
    
    if index_path.exists():
        logger.info(f"Removing old index: {index_path}")
        index_path.unlink()
    if metadata_path.exists():
        logger.info(f"Removing old metadata: {metadata_path}")
        metadata_path.unlink()
    
    # Create FAISS client
    faiss_client = FaissClient(metric=metric)
    
    # Create dataset and dataloader with preprocessing
    # Always use advanced preprocessing as requested
    dataset = LeafDataset(image_data, use_preprocessing=True, adaptive_profile=adaptive_profile)
    
    # Use parallel workers to speed up CPU-bound preprocessing
    # Each worker preprocesses images in parallel
    if num_workers == 0:
        num_workers = min(6, (os.cpu_count() or 4) // 2)  # 4-6 workers optimal
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),  # Pin memory for faster GPU transfer
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch more for GPU
        persistent_workers=num_workers > 0,
        timeout=300  # 5 min timeout for slow preprocessing
    )
    
    logger.info(f"DataLoader configured with {num_workers} workers for parallel preprocessing")
    logger.info(f"Each worker will preprocess images independently for speedup")
    
    # Process images with GPU streaming
    all_features = []
    all_file_ids = []
    failed = 0
    faiss_buffer = []
    
    # Initialize CUDA stream if using GPU
    stream = torch.cuda.Stream() if device.type == 'cuda' else None
    
    with torch.inference_mode():
        for images_batch, file_ids_batch in tqdm(loader, desc="Extracting features"):
            # Filter out failed images (file_id == -1)
            valid_mask = file_ids_batch != -1
            if not valid_mask.any():
                failed += len(file_ids_batch)
                continue
            
            images_batch = images_batch[valid_mask]
            file_ids_batch = file_ids_batch[valid_mask]
            
            if device.type == 'cuda' and stream is not None:
                # GPU streaming for overlap (preprocessing on CPU, inference on GPU in parallel)
                with torch.cuda.stream(stream):  # type: ignore
                    # Non-blocking transfer from pinned memory to GPU
                    images_gpu = images_batch.to(device, non_blocking=True)
                    
                    # Mixed precision (FP16) for 2x faster inference on modern GPUs
                    with torch.cuda.amp.autocast():
                        features = model(images_gpu)
                        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
                        # L2 normalize on GPU (faster than CPU)
                        features = torch.nn.functional.normalize(features, dim=1)
                
                # Wait for GPU stream to complete
                torch.cuda.current_stream().wait_stream(stream)  # type: ignore
                
                # Async transfer to CPU (overlaps with next batch preprocessing)
                features_np = features.cpu().numpy()
            else:
                # CPU processing
                features = model(images_batch)
                features = features.squeeze(-1).squeeze(-1)
                features = torch.nn.functional.normalize(features, dim=1)
                features_np = features.numpy()
            
            # Accumulate in buffer
            faiss_buffer.append(features_np)
            all_file_ids.extend(file_ids_batch.tolist())
            
            # Add to FAISS in bigger chunks for efficiency
            if len(faiss_buffer) >= 8:
                features_chunk = np.vstack(faiss_buffer)
                all_features.append(features_chunk)
                faiss_buffer.clear()
    
    # Flush remaining buffer
    if faiss_buffer:
        features_chunk = np.vstack(faiss_buffer)
        all_features.append(features_chunk)
    
    # Combine all features
    if not all_features:
        logger.error("No features extracted!")
        return
    
    features_array = np.vstack(all_features)
    logger.info(f"Extracted {len(all_file_ids)} feature vectors (failed: {failed})")
    
    # Add to FAISS index
    logger.info("Adding features to index...")
    faiss_client.add(all_file_ids, features_array)
    
    # Save index
    logger.info("Saving index...")
    faiss_client.save()
    
    # Verify
    logger.info("\nVerifying index...")
    print(f"\nIndex successfully created:")
    if faiss_client.index is not None:
        print(f"  Total vectors indexed: {faiss_client.index.ntotal}")
        print(f"  Index type: {type(faiss_client.index).__name__}")
    print(f"  Metric: {metric.value}")
    
    print("\n" + "=" * 80)
    print("INDEX REBUILT SUCCESSFULLY")
    print("=" * 80)
    print(f"\nIndexing Stats:")
    print(f"  Total images: {total}")
    print(f"  Successfully indexed: {len(all_file_ids)}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {(len(all_file_ids)/total)*100:.1f}%")
    print("\nNext steps:")
    print("1. Update app/main.py to enable query preprocessing:")
    print("   use_query_segmentation=True")
    print("   is_query=True")
    print("\n2. Restart the server:")
    print("   uvicorn app.main:app --reload")
    print("\n3. Test search!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index with optimized parallel processing (always uses COSINE)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for GPU inference (default: 128, larger=better GPU utilization)"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="Use adaptive preprocessing (LAB for lab images, FIELD for field images)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel preprocessing workers (default: 6, range: 4-8)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Always use COSINE metric (hardcoded as requested)
    metric = SimilarityMetric.COSINE
    use_advanced = True  # Always enabled as requested
    num_workers = args.workers
    
    # Estimate time based on configuration
    has_gpu = torch.cuda.is_available()
    if num_workers >= 6 and has_gpu:
        time_est = "30-45 minutes (6+ workers + GPU optimized)"
    elif num_workers >= 4 and has_gpu:
        time_est = "40-60 minutes (4+ workers + GPU)"
    elif num_workers >= 4:
        time_est = "60-90 minutes (4+ workers CPU only)"
    else:
        time_est = "90-120 minutes (limited parallelism)"
    
    # Confirm
    print("\n" + "=" * 80)
    print("REBUILD FAISS INDEX (WITH PREPROCESSING)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Metric: COSINE (always)")
    print(f"  Advanced preprocessing: ENABLED (LAB + FIELD profiles)")
    print(f"  Adaptive profiles: YES")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  GPU: {torch.cuda.is_available()}")
    print(f"  CPU cores: {os.cpu_count()}")
    print("\nThis will:")
    print("  1. Process all 30,866 images")
    print("  2. Apply advanced preprocessing (GrabCut, K-means, rotation, edges)")
    print(f"  3. Use {num_workers} parallel workers to speed up preprocessing")
    print("  4. Extract features with GPU + mixed precision")
    print("  5. OVERWRITE existing index")
    print(f"\nEstimated time: {time_est}")
    print("\nOptimization tips:")
    print(f"  - More workers = faster (try --workers 6 or --workers 8)")
    print(f"  - Smaller batches = less memory (current: {args.batch_size})")
    print("=" * 80)
    
    if not args.yes:
        response = input("\nProceed? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    else:
        print("\nAuto-confirming (--yes flag)...")
        print("Starting rebuild...")
    
    rebuild_index_with_preprocessing(
        metric=metric,
        batch_size=args.batch_size,
        adaptive_profile=args.adaptive,
        use_advanced=use_advanced,
        num_workers=num_workers
    )


if __name__ == "__main__":
    main()

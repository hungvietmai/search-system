"""
Batch Processing for Feature Extraction
Optimized for processing large numbers of images efficiently

Features:
- Batch feature extraction
- GPU optimization
- Progress tracking
- Error handling and retry
- Memory-efficient processing
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing"""
    file_ids: List[int]
    features: np.ndarray
    image_paths: List[str]
    success_count: int
    error_count: int
    errors: List[Dict[str, str]]
    processing_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'file_ids': self.file_ids,
            'image_paths': self.image_paths,
            'feature_shape': self.features.shape if self.features is not None else None,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'errors': self.errors,
            'processing_time': self.processing_time,
            'images_per_second': self.success_count / self.processing_time if self.processing_time > 0 else 0
        }


class BatchFeatureExtractor:
    """Batch feature extraction with GPU optimization"""
    
    def __init__(self,
                 feature_extractor,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 use_gpu: bool = True):
        """
        Initialize batch feature extractor
        
        Args:
            feature_extractor: Feature extractor instance
            batch_size: Number of images per batch
            num_workers: Number of worker threads for loading
            use_gpu: Whether to use GPU
        """
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info(f"Batch processing with GPU (batch_size={batch_size})")
        else:
            logger.info(f"Batch processing with CPU (batch_size={batch_size})")
    
    def extract_batch(self,
                     image_paths: List[str],
                     file_ids: Optional[List[int]] = None) -> BatchResult:
        """
        Extract features for a batch of images
        
        Args:
            image_paths: List of image file paths
            file_ids: Optional list of file IDs
            
        Returns:
            BatchResult with features and statistics
        """
        start_time = datetime.now()
        
        if file_ids is None:
            file_ids = list(range(len(image_paths)))
        
        features_list = []
        success_file_ids = []
        success_paths = []
        errors = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_file_ids = file_ids[i:i + self.batch_size]
            
            try:
                # Load images
                images = []
                valid_indices = []
                
                for j, path in enumerate(batch_paths):
                    try:
                        img = self.feature_extractor.load_image(path)
                        images.append(img)
                        valid_indices.append(j)
                    except Exception as e:
                        errors.append({
                            'file_id': batch_file_ids[j],
                            'path': path,
                            'error': str(e)
                        })
                        logger.warning(f"Failed to load {path}: {e}")
                
                if not images:
                    continue
                
                # Convert to tensors
                image_tensors = torch.stack([
                    self.feature_extractor.preprocess_image(img)
                    for img in images
                ])
                
                # Move to GPU if available
                if self.use_gpu:
                    image_tensors = image_tensors.cuda()
                
                # Extract features in batch
                with torch.no_grad():
                    batch_features = self.feature_extractor.model(image_tensors)
                    batch_features = batch_features.cpu().numpy()
                
                # Collect results
                for j, idx in enumerate(valid_indices):
                    features_list.append(batch_features[j])
                    success_file_ids.append(batch_file_ids[idx])
                    success_paths.append(batch_paths[idx])
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for j, path in enumerate(batch_paths):
                    errors.append({
                        'file_id': batch_file_ids[j],
                        'path': path,
                        'error': f"Batch error: {str(e)}"
                    })
        
        # Stack features
        if features_list:
            features = np.vstack(features_list)
        else:
            features = np.array([])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchResult(
            file_ids=success_file_ids,
            features=features,
            image_paths=success_paths,
            success_count=len(success_file_ids),
            error_count=len(errors),
            errors=errors,
            processing_time=processing_time
        )
    
    def extract_from_directory(self,
                              directory: Path,
                              file_pattern: str = "*.jpg",
                              progress: bool = True) -> BatchResult:
        """
        Extract features from all images in a directory
        
        Args:
            directory: Directory containing images
            file_pattern: File pattern for images
            progress: Show progress bar
            
        Returns:
            BatchResult with all features
        """
        # Find all images
        image_paths = list(Path(directory).rglob(file_pattern))
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        all_features = []
        all_file_ids = []
        all_paths = []
        all_errors = []
        total_time = 0.0
        
        # Process with progress bar
        iterator = tqdm(range(0, len(image_paths), self.batch_size), 
                       desc="Extracting features",
                       disable=not progress)
        
        for i in iterator:
            batch_paths = [str(p) for p in image_paths[i:i + self.batch_size]]
            batch_file_ids = list(range(i, i + len(batch_paths)))
            
            result = self.extract_batch(batch_paths, batch_file_ids)
            
            if result.success_count > 0:
                all_features.append(result.features)
                all_file_ids.extend(result.file_ids)
                all_paths.extend(result.image_paths)
            
            all_errors.extend(result.errors)
            total_time += result.processing_time
        
        # Combine results
        if all_features:
            features = np.vstack(all_features)
        else:
            features = np.array([])
        
        return BatchResult(
            file_ids=all_file_ids,
            features=features,
            image_paths=all_paths,
            success_count=len(all_file_ids),
            error_count=len(all_errors),
            errors=all_errors,
            processing_time=total_time
        )


class ParallelBatchProcessor:
    """Parallel batch processing using multiple workers"""
    
    def __init__(self,
                 feature_extractor,
                 num_processes: int = 4,
                 batch_size: int = 32):
        """
        Initialize parallel batch processor
        
        Args:
            feature_extractor: Feature extractor instance
            num_processes: Number of parallel processes
            batch_size: Batch size per process
        """
        self.feature_extractor = feature_extractor
        self.num_processes = num_processes
        self.batch_size = batch_size
        
        logger.info(f"Parallel batch processor with {num_processes} processes")
    
    def process_batch_worker(self, batch_data: Tuple[List[str], List[int]]) -> BatchResult:
        """
        Worker function for processing a batch
        
        Args:
            batch_data: Tuple of (image_paths, file_ids)
            
        Returns:
            BatchResult
        """
        image_paths, file_ids = batch_data
        
        batch_extractor = BatchFeatureExtractor(
            self.feature_extractor,
            batch_size=self.batch_size,
            num_workers=2,
            use_gpu=False  # Each process uses CPU to avoid GPU conflicts
        )
        
        return batch_extractor.extract_batch(image_paths, file_ids)
    
    def extract_parallel(self,
                        image_paths: List[str],
                        file_ids: Optional[List[int]] = None,
                        progress: bool = True) -> BatchResult:
        """
        Extract features in parallel
        
        Args:
            image_paths: List of image paths
            file_ids: List of file IDs
            progress: Show progress bar
            
        Returns:
            Combined BatchResult
        """
        if file_ids is None:
            file_ids = list(range(len(image_paths)))
        
        start_time = datetime.now()
        
        # Split into chunks for parallel processing
        chunk_size = len(image_paths) // self.num_processes
        if chunk_size == 0:
            chunk_size = len(image_paths)
        
        chunks = []
        for i in range(0, len(image_paths), chunk_size):
            end_idx = min(i + chunk_size, len(image_paths))
            chunks.append((
                image_paths[i:end_idx],
                file_ids[i:end_idx]
            ))
        
        logger.info(f"Processing {len(image_paths)} images in {len(chunks)} chunks")
        
        # Process in parallel
        all_features = []
        all_file_ids = []
        all_paths = []
        all_errors = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = {
                executor.submit(self.process_batch_worker, chunk): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results with progress
            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=len(futures), desc="Processing chunks")
            
            for future in iterator:
                try:
                    result = future.result()
                    
                    if result.success_count > 0:
                        all_features.append(result.features)
                        all_file_ids.extend(result.file_ids)
                        all_paths.extend(result.image_paths)
                    
                    all_errors.extend(result.errors)
                    
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        # Combine results
        if all_features:
            features = np.vstack(all_features)
        else:
            features = np.array([])
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return BatchResult(
            file_ids=all_file_ids,
            features=features,
            image_paths=all_paths,
            success_count=len(all_file_ids),
            error_count=len(all_errors),
            errors=all_errors,
            processing_time=total_time
        )


class StreamingBatchProcessor:
    """Memory-efficient streaming batch processor"""
    
    def __init__(self,
                 feature_extractor,
                 batch_size: int = 32,
                 callback: Optional[Callable] = None):
        """
        Initialize streaming processor
        
        Args:
            feature_extractor: Feature extractor
            batch_size: Batch size
            callback: Optional callback(file_id, features) for each result
        """
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.callback = callback
        
        self.batch_extractor = BatchFeatureExtractor(
            feature_extractor,
            batch_size=batch_size
        )
    
    def process_stream(self,
                      image_paths_generator,
                      progress: bool = True):
        """
        Process images from a generator (memory-efficient)
        
        Args:
            image_paths_generator: Generator yielding (file_id, image_path)
            progress: Show progress
            
        Yields:
            (file_id, features) for each processed image
        """
        batch_paths = []
        batch_file_ids = []
        
        for file_id, image_path in image_paths_generator:
            batch_paths.append(image_path)
            batch_file_ids.append(file_id)
            
            # Process when batch is full
            if len(batch_paths) >= self.batch_size:
                result = self.batch_extractor.extract_batch(batch_paths, batch_file_ids)
                
                # Yield results
                for fid, features in zip(result.file_ids, result.features):
                    if self.callback:
                        self.callback(fid, features)
                    yield fid, features
                
                # Clear batch
                batch_paths = []
                batch_file_ids = []
        
        # Process remaining
        if batch_paths:
            result = self.batch_extractor.extract_batch(batch_paths, batch_file_ids)
            
            for fid, features in zip(result.file_ids, result.features):
                if self.callback:
                    self.callback(fid, features)
                yield fid, features


def create_batch_processor(feature_extractor,
                          mode: str = 'batch',
                          **kwargs):
    """
    Factory function to create batch processor
    
    Args:
        feature_extractor: Feature extractor instance
        mode: 'batch', 'parallel', or 'streaming'
        **kwargs: Additional arguments
        
    Returns:
        Batch processor instance
    """
    if mode == 'batch':
        return BatchFeatureExtractor(feature_extractor, **kwargs)
    elif mode == 'parallel':
        return ParallelBatchProcessor(feature_extractor, **kwargs)
    elif mode == 'streaming':
        return StreamingBatchProcessor(feature_extractor, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


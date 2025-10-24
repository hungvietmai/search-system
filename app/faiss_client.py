"""
Faiss-based similarity search client
Now with multi-metric support!
"""
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from config import settings
from app.similarity_metrics import SimilarityMetric, MetricCalculator, convert_metric_to_faiss

logger = logging.getLogger(__name__)


class FaissClient:
    """Client for Faiss-based similarity search with multi-metric support"""
    
    def __init__(self, dimension: Optional[int] = None, index_path: Optional[str] = None, 
                 metadata_path: Optional[str] = None, metric: Optional[SimilarityMetric] = None):
        """
        Initialize Faiss client
        
        Args:
            dimension: Dimension of the embedding vectors
            index_path: Path to save/load index
            metadata_path: Path to save/load metadata
            metric: Similarity metric to use (default: L2)
        """
        self.dimension = dimension or settings.feature_dim
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = Path(metadata_path or settings.faiss_metadata_path)
        
        # Convert string metric to enum if needed
        if metric:
            self.metric = metric
        else:
            metric_str = getattr(settings, 'similarity_metric', 'l2')
            self.metric = SimilarityMetric(metric_str) if isinstance(metric_str, str) else metric_str
        
        self.index = None
        self.file_ids = []  # Maps index position to file_id
        self.loaded = False
        
        # Create parent directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
        else:
            self._create_index()
    
    def _create_index(self):
        """Create a new Faiss index with specified metric"""
        try:
            # Create appropriate index based on metric
            # Using Flat indices (no training required) for simplicity and reliability
            if self.metric == SimilarityMetric.COSINE or self.metric == SimilarityMetric.ANGULAR:
                # For cosine similarity, use inner product with normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created Faiss index with COSINE/ANGULAR metric")
                
            elif self.metric == SimilarityMetric.INNER_PRODUCT:
                # Inner product metric
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created Faiss index with INNER_PRODUCT metric")
                
            elif self.metric == SimilarityMetric.L2:
                # L2 distance metric (default)
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created Faiss index with L2 metric")
                
            else:
                # For other metrics (L1, etc.), use L2 as fallback with custom post-processing
                logger.warning(f"Metric {self.metric} not natively supported by Faiss, using L2 with post-processing")
                self.index = faiss.IndexFlatL2(self.dimension)
            
            self.file_ids = []
            self.loaded = True
            
            logger.info(f"Created new Faiss Flat index with dimension {self.dimension}, metric {self.metric.value}")
            
        except Exception as e:
            logger.error(f"Failed to create Faiss index: {e}")
            raise
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices)
        
        Args:
            embeddings: Training data (N x dimension)
        """
        try:
            if self.index is None:
                raise ValueError("Index not initialized")
            if not self.index.is_trained:
                logger.info(f"Training index with {len(embeddings)} samples")
                self.index.train(embeddings.astype('float32'))  # type: ignore
                logger.info("Index training complete")
        except Exception as e:
            logger.error(f"Failed to train index: {e}")
            raise
    
    def add(self, file_ids: List[int], embeddings: np.ndarray):
        """
        Add embeddings to the index
        
        Args:
            file_ids: List of file IDs
            embeddings: Numpy array of embeddings (N x dimension)
        """
        try:
            if self.index is None:
                raise ValueError("Index not initialized")
                
            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')
            
            # NOTE: Features are already L2-normalized in feature_extractor.py
            # For cosine/angular similarity, we need to ensure they're normalized
            # But we should NOT double-normalize
            if self.metric in [SimilarityMetric.COSINE, SimilarityMetric.ANGULAR]:
                # Check if already normalized (norm should be ~1.0)
                norms = np.linalg.norm(embeddings, axis=1)
                is_normalized = np.allclose(norms, 1.0, atol=1e-3)
                
                if not is_normalized:
                    logger.warning(f"Embeddings not normalized for {self.metric.value}, normalizing now")
                    embeddings = embeddings / (norms[:, np.newaxis] + 1e-8)
                else:
                    logger.debug(f"Embeddings already normalized for {self.metric.value}")
            
            # Train if not yet trained
            if not self.index.is_trained:
                self.train(embeddings)
            
            # Add to index
            self.index.add(embeddings)  # type: ignore
            
            # Store file_ids
            self.file_ids.extend(file_ids)
            
            logger.info(f"Added {len(file_ids)} embeddings to Faiss index")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               metric: Optional[SimilarityMetric] = None) -> Tuple[List[int], List[float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            metric: Override the default metric for this search (experimental)
            
        Returns:
            Tuple of (file_ids, distances/scores)
        """
        try:
            if not self.loaded or self.index is None:
                raise ValueError("Index not loaded")
            
            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return [], []
            
            # Use provided metric or default
            search_metric = metric or self.metric
            
            # CRITICAL: Validate metric compatibility with index type
            index_type = type(self.index).__name__
            if search_metric in [SimilarityMetric.COSINE, SimilarityMetric.ANGULAR, SimilarityMetric.INNER_PRODUCT]:
                if "IP" not in index_type:
                    logger.error(f"❌ METRIC MISMATCH: Index is {index_type} but searching with {search_metric.value}")
                    logger.error("⚠️  This will return WRONG results! Rebuild index with cosine/angular metric.")
                    logger.error(f"⚠️  Index metric: {self.metric.value}, Search metric: {search_metric.value}")
                    raise ValueError(
                        f"Metric mismatch: Index built with {self.metric.value}, "
                        f"but searching with {search_metric.value}. Rebuild the index!"
                    )
            elif search_metric == SimilarityMetric.L2:
                if "L2" not in index_type:
                    logger.error(f"❌ METRIC MISMATCH: Index is {index_type} but searching with {search_metric.value}")
                    raise ValueError(
                        f"Metric mismatch: Index built with {self.metric.value}, "
                        f"but searching with {search_metric.value}. Rebuild the index!"
                    )
            
            # Ensure query is float32 and 2D
            query = query_embedding.astype('float32').reshape(1, -1)
            
            # Check if query needs normalization for cosine/angular similarity
            if search_metric in [SimilarityMetric.COSINE, SimilarityMetric.ANGULAR]:
                query_norm = np.linalg.norm(query)
                is_normalized = np.isclose(query_norm, 1.0, atol=1e-3)
                
                if not is_normalized:
                    logger.warning(f"Query not normalized for {search_metric.value}, normalizing now")
                    query = query / (query_norm + 1e-8)
                else:
                    logger.debug(f"Query already normalized for {search_metric.value}")
            
            # Search
            distances, indices = self.index.search(query, min(top_k, self.index.ntotal))  # type: ignore
            
            # Convert to lists
            distances = distances[0].tolist()
            indices = indices[0].tolist()
            
            # CRITICAL FIX: FAISS IndexFlatIP returns actual similarity scores, NOT negative values!
            # For L2 distance (IndexFlatL2), lower is better (it's a distance)
            # For Inner Product (IndexFlatIP), higher is better (it's a similarity)
            # The previous code incorrectly negated the scores, causing negative distances
            # NO NEGATION NEEDED - FAISS returns the correct values!
            
            logger.debug(f"Search with {search_metric.value}: raw distances = {distances[:3]}")
            
            # Map indices to file_ids
            file_ids = []
            valid_distances = []
            
            for idx, dist in zip(indices, distances):
                if idx != -1 and idx < len(self.file_ids):
                    file_ids.append(self.file_ids[idx])
                    valid_distances.append(dist)
            
            logger.info(f"Found {len(file_ids)} results with {search_metric.value} metric")
            
            return file_ids, valid_distances
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            # Save Faiss index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            metadata = {
                'file_ids': self.file_ids,
                'dimension': self.dimension,
                'metric': self.metric.value  # Save metric type
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved Faiss index to {self.index_path} (metric: {self.metric.value})")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load(self):
        """Load index and metadata from disk"""
        try:
            if not self.index_path.exists():
                logger.warning(f"Index file not found: {self.index_path}")
                self._create_index()
                return
            
            # Load Faiss index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.file_ids = metadata['file_ids']
            self.dimension = metadata['dimension']
            
            # Load metric if available (for backward compatibility)
            if 'metric' in metadata:
                self.metric = SimilarityMetric(metadata['metric'])
            else:
                logger.warning("No metric found in metadata, using default L2")
                self.metric = SimilarityMetric.L2
            
            self.loaded = True
            
            logger.info(f"Loaded Faiss index from {self.index_path} with {self.index.ntotal} vectors (metric: {self.metric.value})")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._create_index()
    
    def delete(self, file_ids: List[int]):
        """
        Delete embeddings by file IDs
        Note: Faiss doesn't support direct deletion, so we need to rebuild the index
        
        Args:
            file_ids: List of file IDs to delete
        """
        try:
            logger.warning("Faiss doesn't support direct deletion. Consider rebuilding the index.")
            # For now, just log a warning
            # To implement deletion, you would need to:
            # 1. Remove file_ids from self.file_ids
            # 2. Get all embeddings except the ones to delete
            # 3. Create a new index and add remaining embeddings
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of vectors in index"""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def check_health(self) -> Dict[str, Any]:
        """Check Faiss index health"""
        try:
            return {
                "loaded": self.loaded,
                "count": self.get_count(),
                "dimension": self.dimension,
                "trained": self.index.is_trained if self.index else False
            }
        except Exception as e:
            return {
                "loaded": False,
                "error": str(e)
            }
    
    def reset(self):
        """Reset the index"""
        try:
            self._create_index()
            logger.info("Faiss index reset")
        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            raise


# Global Faiss client instance
_faiss_client = None


def get_faiss_client(metric: Optional[SimilarityMetric] = None) -> FaissClient:
    """
    Get or create global Faiss client instance
    
    Args:
        metric: Similarity metric to use (if creating new instance)
        
    Returns:
        FaissClient instance
    """
    global _faiss_client
    if _faiss_client is None:
        _faiss_client = FaissClient(metric=metric)
    return _faiss_client



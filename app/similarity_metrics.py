"""
Similarity Metrics Module

Implements multiple similarity/distance metrics for vector comparison:
- Cosine similarity
- Inner product (dot product)
- L2 distance (Euclidean)
- L1 distance (Manhattan)
- Normalized inner product

Each metric can be used for similarity search with different properties:
- Cosine: Angle-based, invariant to vector magnitude
- Inner Product: Considers both direction and magnitude
- L2: Euclidean distance, most common
- L1: Manhattan distance, more robust to outliers
"""

import numpy as np
from enum import Enum
from typing import Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Available similarity metrics"""
    COSINE = "cosine"  # Cosine similarity (1 - cosine distance)
    INNER_PRODUCT = "inner_product"  # Dot product
    L2 = "l2"  # Euclidean distance
    L1 = "l1"  # Manhattan distance
    ANGULAR = "angular"  # Angular distance


class MetricCalculator:
    """Calculator for various similarity metrics"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity between two vectors
        Range: [-1, 1], higher is more similar
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine distance (1 - cosine similarity)
        Range: [0, 2], lower is more similar
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine distance
        """
        similarity = MetricCalculator.cosine_similarity(vec1, vec2)
        return 1.0 - similarity
    
    @staticmethod
    def inner_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Inner product (dot product) between two vectors
        Higher values indicate higher similarity
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Inner product score
        """
        return float(np.dot(vec1, vec2))
    
    @staticmethod
    def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Euclidean (L2) distance between two vectors
        Range: [0, inf], lower is more similar
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            L2 distance
        """
        return float(np.linalg.norm(vec1 - vec2))
    
    @staticmethod
    def l1_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Manhattan (L1) distance between two vectors
        Range: [0, inf], lower is more similar
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            L1 distance
        """
        return float(np.sum(np.abs(vec1 - vec2)))
    
    @staticmethod
    def angular_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Angular distance in radians
        Range: [0, pi], lower is more similar
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Angular distance in radians
        """
        cos_sim = MetricCalculator.cosine_similarity(vec1, vec2)
        # Clamp to [-1, 1] to avoid numerical errors
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return float(np.arccos(cos_sim))
    
    @staticmethod
    def compute_metric(vec1: np.ndarray, vec2: np.ndarray, 
                      metric: SimilarityMetric) -> float:
        """
        Compute similarity/distance using specified metric
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Metric to use
            
        Returns:
            Similarity/distance score
        """
        if metric == SimilarityMetric.COSINE:
            return MetricCalculator.cosine_similarity(vec1, vec2)
        elif metric == SimilarityMetric.INNER_PRODUCT:
            return MetricCalculator.inner_product(vec1, vec2)
        elif metric == SimilarityMetric.L2:
            return MetricCalculator.l2_distance(vec1, vec2)
        elif metric == SimilarityMetric.L1:
            return MetricCalculator.l1_distance(vec1, vec2)
        elif metric == SimilarityMetric.ANGULAR:
            return MetricCalculator.angular_distance(vec1, vec2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    @staticmethod
    def batch_compute(query: np.ndarray, vectors: np.ndarray, 
                     metric: SimilarityMetric) -> np.ndarray:
        """
        Compute similarity/distance between query and multiple vectors
        
        Args:
            query: Query vector (1D)
            vectors: Matrix of vectors (N x D)
            metric: Metric to use
            
        Returns:
            Array of scores (N,)
        """
        # Ensure query is 1D
        if query.ndim > 1:
            query = query.flatten()
        
        if metric == SimilarityMetric.COSINE:
            # Normalize query
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            
            # Normalize all vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            vectors_norm = vectors / norms
            
            # Compute dot products
            similarities = np.dot(vectors_norm, query_norm)
            return similarities
        
        elif metric == SimilarityMetric.INNER_PRODUCT:
            # Compute dot products
            return np.dot(vectors, query)
        
        elif metric == SimilarityMetric.L2:
            # Compute L2 distances
            diff = vectors - query
            return np.linalg.norm(diff, axis=1)
        
        elif metric == SimilarityMetric.L1:
            # Compute L1 distances
            diff = vectors - query
            return np.sum(np.abs(diff), axis=1)
        
        elif metric == SimilarityMetric.ANGULAR:
            # Compute angular distances
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            vectors_norm = vectors / norms
            
            cos_sims = np.dot(vectors_norm, query_norm)
            cos_sims = np.clip(cos_sims, -1.0, 1.0)
            return np.arccos(cos_sims)
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    @staticmethod
    def is_distance_metric(metric: SimilarityMetric) -> bool:
        """
        Check if metric is distance-based (lower is better) or similarity-based (higher is better)
        
        Args:
            metric: Metric to check
            
        Returns:
            True if distance metric, False if similarity metric
        """
        distance_metrics = {SimilarityMetric.L2, SimilarityMetric.L1, SimilarityMetric.ANGULAR}
        return metric in distance_metrics
    
    @staticmethod
    def normalize_vector_for_metric(vector: np.ndarray, 
                                   metric: SimilarityMetric) -> np.ndarray:
        """
        Normalize vector appropriately for the given metric
        
        Args:
            vector: Input vector
            metric: Target metric
            
        Returns:
            Normalized vector
        """
        if metric == SimilarityMetric.COSINE or metric == SimilarityMetric.ANGULAR:
            # L2 normalize for cosine/angular
            return vector / (np.linalg.norm(vector) + 1e-8)
        elif metric == SimilarityMetric.L2:
            # L2 normalize is standard
            return vector / (np.linalg.norm(vector) + 1e-8)
        else:
            # No normalization needed for inner product and L1
            return vector


def convert_metric_to_faiss(metric: SimilarityMetric) -> str:
    """
    Convert SimilarityMetric to Faiss index type string
    
    Args:
        metric: Similarity metric
        
    Returns:
        Faiss index type string
    """
    if metric == SimilarityMetric.COSINE or metric == SimilarityMetric.ANGULAR:
        return "COSINE"
    elif metric == SimilarityMetric.INNER_PRODUCT:
        return "INNER_PRODUCT"
    elif metric == SimilarityMetric.L2:
        return "L2"
    else:
        # Default to L2 for other metrics
        logger.warning(f"Metric {metric} not directly supported by Faiss, using L2")
        return "L2"


def convert_metric_to_milvus(metric: SimilarityMetric) -> str:
    """
    Convert SimilarityMetric to Milvus metric type string
    
    Args:
        metric: Similarity metric
        
    Returns:
        Milvus metric type string
    """
    if metric == SimilarityMetric.COSINE:
        return "COSINE"
    elif metric == SimilarityMetric.INNER_PRODUCT:
        return "IP"
    elif metric == SimilarityMetric.L2:
        return "L2"
    elif metric == SimilarityMetric.L1:
        return "L1"
    else:
        # Default to L2
        logger.warning(f"Metric {metric} not directly supported by Milvus, using L2")
        return "L2"


# Pre-compute metric properties for quick lookup
METRIC_PROPERTIES = {
    SimilarityMetric.COSINE: {
        'is_distance': False,
        'range': (-1.0, 1.0),
        'higher_is_better': True,
        'requires_normalization': True,
        'description': 'Cosine similarity - angle between vectors'
    },
    SimilarityMetric.INNER_PRODUCT: {
        'is_distance': False,
        'range': (float('-inf'), float('inf')),
        'higher_is_better': True,
        'requires_normalization': False,
        'description': 'Inner product - dot product of vectors'
    },
    SimilarityMetric.L2: {
        'is_distance': True,
        'range': (0.0, float('inf')),
        'higher_is_better': False,
        'requires_normalization': False,
        'description': 'Euclidean distance - L2 norm'
    },
    SimilarityMetric.L1: {
        'is_distance': True,
        'range': (0.0, float('inf')),
        'higher_is_better': False,
        'requires_normalization': False,
        'description': 'Manhattan distance - L1 norm'
    },
    SimilarityMetric.ANGULAR: {
        'is_distance': True,
        'range': (0.0, np.pi),
        'higher_is_better': False,
        'requires_normalization': True,
        'description': 'Angular distance in radians'
    }
}


def get_metric_info(metric: SimilarityMetric) -> dict:
    """
    Get information about a metric
    
    Args:
        metric: Similarity metric
        
    Returns:
        Dictionary with metric properties
    """
    return METRIC_PROPERTIES.get(metric, {})


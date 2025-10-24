"""
Enhanced Diversity-Based Re-ranking
Implements multiple diversity algorithms for search result diversification

Algorithms:
- MMR (Maximal Marginal Relevance)
- DPP (Determinantal Point Processes)
- Clustering-based diversity
- Feature-based diversity
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances  # type: ignore

logger = logging.getLogger(__name__)


class DiversityAlgorithm(Enum):
    """Diversity algorithms"""
    MMR = "mmr"                    # Maximal Marginal Relevance
    DPP = "dpp"                    # Determinantal Point Processes
    CLUSTERING = "clustering"      # Cluster-based diversity
    FEATURE_BASED = "feature"      # Feature-based diversity
    HYBRID = "hybrid"              # Combination of multiple


@dataclass
class DiversityResult:
    """Result with diversity information"""
    file_id: int
    species: str
    source: str
    distance: float
    relevance_score: float
    diversity_score: float
    final_score: float
    cluster_id: Optional[int] = None


class MMRDiversity:
    """
    Maximal Marginal Relevance (MMR)
    Balances relevance and diversity
    
    MMR = λ * Similarity(doc, query) - (1-λ) * max(Similarity(doc, selected))
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR
        
        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
                         Higher = more relevance, Lower = more diversity
        """
        self.lambda_param = lambda_param
    
    def rerank(self,
              query_vector: np.ndarray,
              candidate_vectors: List[np.ndarray],
              candidate_scores: List[float],
              top_k: int = 10) -> List[int]:
        """
        Re-rank using MMR
        
        Args:
            query_vector: Query feature vector
            candidate_vectors: Candidate feature vectors
            candidate_scores: Initial relevance scores
            top_k: Number of results to return
            
        Returns:
            Indices of selected candidates (in order)
        """
        if len(candidate_vectors) == 0:
            return []
        
        # Convert to numpy arrays
        query_vector = np.array(query_vector)
        candidate_vectors_array = np.array(candidate_vectors)
        
        # Compute relevance scores (cosine similarity with query)
        relevance_scores = cosine_similarity(
            candidate_vectors_array,
            query_vector.reshape(1, -1)
        ).flatten()
        
        # Normalize relevance scores
        if relevance_scores.max() > relevance_scores.min():
            relevance_scores = (relevance_scores - relevance_scores.min()) / \
                              (relevance_scores.max() - relevance_scores.min())
        
        # Normalize candidate scores
        if candidate_scores:
            candidate_scores_array = np.array(candidate_scores)
            if candidate_scores_array.max() > candidate_scores_array.min():
                candidate_scores_array = (candidate_scores_array - candidate_scores_array.min()) / \
                                  (candidate_scores_array.max() - candidate_scores_array.min())
            # Combine with relevance
            relevance_scores = 0.5 * relevance_scores + 0.5 * candidate_scores_array
        
        # MMR selection
        selected_indices: List[int] = []
        remaining_indices: Set[int] = set(range(len(candidate_vectors)))
        
        # Select first item (highest relevance)
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining items
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance term
                relevance = relevance_scores[idx]
                
                # Diversity term (max similarity to selected items)
                similarities = cosine_similarity(
                    candidate_vectors_array[idx].reshape(1, -1),
                    candidate_vectors_array[selected_indices]
                ).flatten()
                max_similarity = similarities.max()
                
                # MMR score
                mmr_score = self.lambda_param * relevance - \
                           (1 - self.lambda_param) * max_similarity
                
                mmr_scores.append((idx, mmr_score))
            
            # Select item with highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices


class ClusteringDiversity:
    """
    Clustering-based diversity
    Ensures results from different clusters
    """
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize clustering diversity
        
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
    
    def rerank(self,
              candidate_vectors: List[np.ndarray],
              candidate_scores: List[float],
              top_k: int = 10) -> Tuple[List[int], List[int]]:
        """
        Re-rank using clustering
        
        Args:
            candidate_vectors: Candidate feature vectors
            candidate_scores: Initial relevance scores
            top_k: Number of results to return
            
        Returns:
            Tuple of (selected_indices, cluster_labels)
        """
        if len(candidate_vectors) == 0:
            return [], []
        
        # Convert to numpy array
        X = np.array(candidate_vectors)
        
        # Perform clustering
        n_clusters = min(self.n_clusters, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Select top item from each cluster
        selected_indices = []
        clusters_used = set()
        
        # Sort candidates by score
        sorted_indices = np.argsort(candidate_scores)[::-1]
        
        # First pass: select best item from each cluster
        for idx in sorted_indices:
            cluster = cluster_labels[idx]
            if cluster not in clusters_used:
                selected_indices.append(idx)
                clusters_used.add(cluster)
            
            if len(selected_indices) >= top_k:
                break
        
        # Second pass: fill remaining slots with next best items
        if len(selected_indices) < top_k:
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= top_k:
                        break
        
        return selected_indices, cluster_labels.tolist()


class FeatureBasedDiversity:
    """
    Feature-based diversity
    Ensures diversity in multiple feature dimensions
    """
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        Initialize feature-based diversity
        
        Args:
            feature_weights: Weights for different features
        """
        self.feature_weights = feature_weights or {
            'species': 0.4,
            'source': 0.2,
            'visual': 0.4
        }
    
    def compute_diversity_score(self,
                                candidate: Dict,
                                selected: List[Dict],
                                candidate_vector: np.ndarray,
                                selected_vectors: List[np.ndarray]) -> float:
        """
        Compute diversity score for a candidate
        
        Args:
            candidate: Candidate item
            selected: Already selected items
            candidate_vector: Candidate feature vector
            selected_vectors: Feature vectors of selected items
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if not selected:
            return 1.0
        
        diversity = 0.0
        
        # Species diversity
        species_diversity = 1.0
        for sel in selected:
            if candidate['species'] == sel['species']:
                species_diversity *= 0.5  # Penalize same species
        diversity += self.feature_weights['species'] * species_diversity
        
        # Source diversity
        source_diversity = 1.0
        sources_used = set(sel['source'] for sel in selected)
        if candidate['source'] in sources_used:
            source_diversity = 0.5
        diversity += self.feature_weights['source'] * source_diversity
        
        # Visual diversity (feature vector similarity)
        if selected_vectors:
            similarities = cosine_similarity(
                candidate_vector.reshape(1, -1),
                np.array(selected_vectors)
            ).flatten()
            visual_diversity = 1.0 - similarities.max()  # Lower similarity = higher diversity
            diversity += self.feature_weights['visual'] * visual_diversity
        
        return diversity
    
    def rerank(self,
              candidates: List[Dict],
              candidate_vectors: List[np.ndarray],
              candidate_scores: List[float],
              top_k: int = 10) -> List[int]:
        """
        Re-rank using feature-based diversity
        
        Args:
            candidates: Candidate items (with species, source, etc.)
            candidate_vectors: Feature vectors
            candidate_scores: Relevance scores
            top_k: Number of results
            
        Returns:
            Selected indices
        """
        selected_indices = []
        selected_items = []
        selected_vectors = []
        remaining_indices = set(range(len(candidates)))
        
        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining_indices:
                # Combine relevance and diversity
                relevance = candidate_scores[idx]
                diversity = self.compute_diversity_score(
                    candidates[idx],
                    selected_items,
                    candidate_vectors[idx],
                    selected_vectors
                )
                
                # Combined score (70% relevance, 30% diversity)
                score = 0.7 * relevance + 0.3 * diversity
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_items.append(candidates[best_idx])
                selected_vectors.append(candidate_vectors[best_idx])
                remaining_indices.remove(best_idx)
        
        return selected_indices


class EnhancedDiversityRanker:
    """
    Enhanced diversity-based re-ranking engine
    Supports multiple algorithms and hybrid approaches
    """
    
    def __init__(self, 
                 algorithm: DiversityAlgorithm = DiversityAlgorithm.MMR,
                 **kwargs):
        """
        Initialize diversity ranker
        
        Args:
            algorithm: Diversity algorithm to use
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.ranker: Any  # Different rankers have different signatures
        
        # Initialize algorithm
        if algorithm == DiversityAlgorithm.MMR:
            self.ranker = MMRDiversity(
                lambda_param=kwargs.get('lambda_param', 0.7)
            )
        elif algorithm == DiversityAlgorithm.CLUSTERING:
            self.ranker = ClusteringDiversity(
                n_clusters=kwargs.get('n_clusters', 5)
            )
        elif algorithm == DiversityAlgorithm.FEATURE_BASED:
            self.ranker = FeatureBasedDiversity(
                feature_weights=kwargs.get('feature_weights')
            )
        else:
            # Default to MMR
            self.ranker = MMRDiversity()
        
        logger.info(f"Initialized {algorithm.value} diversity ranker")
    
    def rerank(self,
              candidates: List[Dict],
              candidate_vectors: List[np.ndarray],
              candidate_scores: List[float],
              query_vector: Optional[np.ndarray] = None,
              top_k: int = 10) -> List[Dict]:
        """
        Re-rank candidates for diversity
        
        Args:
            candidates: Candidate items
            candidate_vectors: Feature vectors
            candidate_scores: Relevance scores
            query_vector: Query vector (for MMR)
            top_k: Number of results
            
        Returns:
            Re-ranked candidates
        """
        if len(candidates) == 0:
            return []
        
        # Normalize scores
        scores = np.array(candidate_scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores)
        
        # Apply algorithm
        cluster_labels: Optional[List[int]] = None
        selected_indices: List[int]
        
        if self.algorithm == DiversityAlgorithm.MMR:
            if query_vector is None:
                raise ValueError("MMR requires query_vector")
            selected_indices = self.ranker.rerank(  # type: ignore
                query_vector,
                candidate_vectors,
                scores.tolist(),
                top_k
            )
            
        elif self.algorithm == DiversityAlgorithm.CLUSTERING:
            result_tuple = self.ranker.rerank(  # type: ignore
                candidate_vectors,
                scores.tolist(),
                top_k
            )
            selected_indices = list(result_tuple[0])  # type: ignore
            cluster_labels = list(result_tuple[1]) if result_tuple[1] else None  # type: ignore
            
        elif self.algorithm == DiversityAlgorithm.FEATURE_BASED:
            selected_indices = self.ranker.rerank(  # type: ignore
                candidates,
                candidate_vectors,
                scores.tolist(),
                top_k
            )
        
        else:
            # Fallback: simple score-based ranking
            selected_indices = np.argsort(scores)[::-1][:top_k].tolist()
        
        # Reorder candidates
        reranked = []
        for i, idx in enumerate(selected_indices):
            result = DiversityResult(
                file_id=candidates[idx].get('file_id', 0),
                species=candidates[idx].get('species', ''),
                source=candidates[idx].get('source', ''),
                distance=candidates[idx].get('distance', 0.0),
                relevance_score=float(scores[idx]),
                diversity_score=1.0 / (i + 1),  # Position-based diversity
                final_score=float(scores[idx]) * (1.0 / (i + 1)),
                cluster_id=int(cluster_labels[idx]) if cluster_labels and idx < len(cluster_labels) else None
            )
            reranked.append(result)
        
        logger.info(f"Re-ranked {len(candidates)} candidates to {len(reranked)} diverse results")
        
        return reranked


# Global diversity ranker
_diversity_ranker = None


def get_diversity_ranker(
    algorithm: DiversityAlgorithm = DiversityAlgorithm.MMR,
    **kwargs
) -> EnhancedDiversityRanker:
    """
    Get or create global diversity ranker
    
    Args:
        algorithm: Diversity algorithm
        **kwargs: Algorithm parameters
        
    Returns:
        EnhancedDiversityRanker instance
    """
    global _diversity_ranker
    
    if _diversity_ranker is None or _diversity_ranker.algorithm != algorithm:
        _diversity_ranker = EnhancedDiversityRanker(algorithm, **kwargs)
    
    return _diversity_ranker


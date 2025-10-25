"""
Two-Stage Search with Advanced Re-ranking
Stage 1: Retrieve more candidates using vector similarity
Stage 2: Re-rank using multiple criteria

Expected improvement: +15-20% result relevance
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import LeafImage

logger = logging.getLogger(__name__)


class ReRankingCriterion(Enum):
    """Re-ranking criteria"""
    VECTOR_SIMILARITY = "vector_similarity"
    SPECIES_PRIOR = "species_prior"
    IMAGE_SOURCE_PREFERENCE = "image_source"
    DIVERSITY = "diversity"


@dataclass
class SearchCandidate:
    """Single search candidate with metadata"""
    file_id: int
    distance: float
    species: str
    source: str
    image_path: str
    segmented_path: Optional[str]
    milvus_id: Optional[int] = None
    faiss_id: Optional[int] = None
    
    # Re-ranking scores
    vector_score: float = 0.0
    frequency_score: float = 0.0
    source_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0


class TwoStageSearchEngine:
    """
    Two-stage search engine with intelligent re-ranking
    
    Stage 1: Vector similarity (retrieve candidates)
    Stage 2: Multi-criteria re-ranking
    """
    
    def __init__(self, 
                 retrieval_multiplier: int = 5,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize two-stage search engine
        
        Args:
            retrieval_multiplier: Retrieve top_k * multiplier candidates in stage 1
            weights: Weights for each re-ranking criterion
        """
        self.retrieval_multiplier = retrieval_multiplier
        
        # Default weights for re-ranking criteria
        self.weights = weights or {
            'vector_similarity': 0.70,
            'species_prior':     0.05,  # small nudge, tempered IDF
            'source_preference': 0.15,  # prefer inferred domain
            'diversity':         0.00,  # handled by MMR
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"TwoStageSearchEngine initialized with weights: {self.weights}")
    
    def stage1_retrieval(self,
                        search_function: Callable,
                        query_features: np.ndarray,
                        top_k: int) -> Tuple[List[int], List[float]]:
        """
        Stage 1: Retrieve more candidates than needed
        
        Args:
            search_function: Function that performs vector search
            query_features: Query feature vector
            top_k: Number of final results needed
            
        Returns:
            Tuple of (file_ids, distances)
        """
        # Retrieve more candidates
        retrieve_k = min(top_k * self.retrieval_multiplier, 1000)
        
        logger.info(f"Stage 1: Retrieving {retrieve_k} candidates for top-{top_k} results")
        
        # Call vector search
        file_ids, distances = search_function(query_features, retrieve_k)
        
        return file_ids, distances
    
    def to_similarities(self, distances: List[float]) -> List[float]:
        """
        Convert distances to similarity scores using temperatured softmax
        More robust than min-max normalization when distance spreads are tiny
        
        Args:
            distances: L2 distances from vector search
            
        Returns:
            Similarity scores
        """
        d = np.array(distances, dtype=np.float32)
        # Robust temperature from IQR (fall back if degenerate)
        q1, q3 = np.percentile(d, [25, 75])
        iqr = max(q3 - q1, 1e-6)
        T = 0.25 * iqr  # smaller T => peakier when neighbors are clearly separated
        sims = np.exp(-(d - d.min()) / max(T, 1e-6))
        sims /= sims.sum() + 1e-9
        return sims.tolist()
    
    def compute_species_idf_scores(self, candidates: List[SearchCandidate], db: Session, alpha: float = 0.3) -> List[float]:
        """
        Compute scores based on inverse species frequency (IDF-like) in database
        Less common species get higher scores (rarer species are more distinctive)
        Batched to avoid N DB round-trips.
        
        Args:
            candidates: List of search candidates
            db: Database session
            alpha: Tempering parameter (lower = less influence)
            
        Returns:
            Inverse frequency scores
        """
        # batch query counts for candidate species
        sp_set = {c.species for c in candidates}
        rows = (db.query(LeafImage.species, func.count(LeafImage.species))
                  .filter(LeafImage.species.in_(sp_set))
                  .group_by(LeafImage.species).all())
        counts = {sp: cnt for sp, cnt in rows}
        maxc = max(counts.values()) if counts else 1
        # idf = log(1 + max_count / count) -> higher for rarer species
        idf = {sp: np.log1p(maxc / max(counts.get(sp, 1), 1)) for sp in sp_set}
        vals = np.array([idf[c.species] for c in candidates], dtype=np.float32)
        # normalize to 0..1 and temper so prior never dominates
        vmax, vmin = float(vals.max()), float(vals.min())
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
        else:
            vals = np.full_like(vals, 0.5)
        scores = (vals ** alpha).tolist()
        return scores
    
    def infer_query_domain(self, query_features: np.ndarray, nn_sources: List[str]) -> str:
        """
        Infer the query domain based on nearest neighbor sources or feature characteristics
        Field photos are more common in production, so we bias toward field if uncertain
        
        Args:
            query_features: Query feature vector
            nn_sources: List of sources for nearest neighbors
            
        Returns:
            Inferred domain ('lab' or 'field')
        """
        # simple majority vote of top-N neighbor sources, fallback to feature spread
        if nn_sources:
            return 'field' if nn_sources.count('field') >= nn_sources.count('lab') else 'lab'
        return 'field' if np.std(query_features) < 0.2 else 'lab'

    def compute_source_scores(self, candidates: List[SearchCandidate], prefer: str, seg_conf: Optional[Dict[int, float]] = None) -> List[float]:
        """
        Compute scores based on image source and quality signals
        Can adjust preference based on query domain and quality
        
        Args:
            candidates: List of search candidates
            prefer: Preferred source ('lab' or 'field')
            seg_conf: Optional segmentation confidence scores
            
        Returns:
            Source scores
        """
        out = []
        for c in candidates:
            base = 1.0 if c.source == prefer else 0.7
            if seg_conf and c.file_id in seg_conf:
                # nudge by segmentation/quality (bounded)
                base *= 0.9 + 0.2 * np.clip(seg_conf[c.file_id], 0, 1)
            out.append(float(min(base, 1.0)))
        return out
    
    def mmr_rerank(self, cands: List[SearchCandidate], rel: np.ndarray, emb_getter: Callable[[int], np.ndarray], lam: float = 0.7, start_at: int = 3, k: Optional[int]=None):
        """
        Maximal Marginal Relevance re-ranking to promote diversity
        Balances relevance and dissimilarity to avoid near-duplicates
         
        Args:
            cands: List of search candidates
            rel: Precomputed relevance per candidate (e.g., vector similarities)
            emb_getter: Function to get embedding for a file_id
            lam: Balance between relevance and diversity (higher = more relevance)
            start_at: Start penalizing diversity after this rank (don't penalize top-3)
            k: Number of results to return (default: all)
             
        Returns:
            Re-ranked list of candidates
        """
        import numpy as np
        k = k or len(cands)
        picked, remaining = [], list(range(len(cands)))
        # Precompute embeddings for cosine dissimilarity
        Em = np.stack([emb_getter(c.file_id) for c in cands]).astype(np.float32)
        Em /= np.linalg.norm(Em, axis=1, keepdims=True) + 1e-9
        S = Em @ Em.T  # cosine similarity
        for t in range(k):
            if t < start_at:
                idx = max(remaining, key=lambda i: rel[i])
            else:
                idx = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * (max(S[i, picked]) if picked else 0.0))
            picked.append(idx); remaining.remove(idx)
        return [cands[i] for i in picked]
    
    def stage2_reranking(self,
                        candidates: List[SearchCandidate],
                        query_features: np.ndarray,
                        db: Session,
                        prefer_source: Optional[str] = None,
                        promote_diversity: bool = True,
                        emb_getter: Optional[Callable[[int], np.ndarray]] = None) -> List[SearchCandidate]:
        """
        Stage 2: Re-rank candidates using multiple criteria
         
        Args:
            candidates: List of search candidates from stage 1
            query_features: Query feature vector for domain inference
            db: Database session
            prefer_source: Preferred image source ("lab", "field", or None/"auto" to infer)
            promote_diversity: Whether to promote species diversity
            emb_getter: Function to get embedding for a file_id (required when promote_diversity=True)
             
        Returns:
            Re-ranked candidates
        """
        if not candidates:
            return candidates
         
        logger.info(f"Stage 2: Re-ranking {len(candidates)} candidates")
         
        # Compute individual scores
        vector_scores = [c.vector_score for c in candidates]
        
        idf_scores = self.compute_species_idf_scores(candidates, db)
        
        # Use provided prefer_source or infer if needed
        pref = prefer_source if prefer_source in {"lab", "field"} else \
               self.infer_query_domain(query_features, [c.source for c in candidates[:min(5, len(candidates))]])
        
        # Compute source scores based on preferred source
        source_scores = self.compute_source_scores(candidates, pref)
        
        # Compute diversity scores using MMR approach
        diversity_scores = [1.0] * len(candidates)  # Placeholder since MMR handles diversity differently
        
        # Combine scores using weights to create relevance vector for MMR
        rel = (
            self.weights['vector_similarity'] * np.array(vector_scores) +
            self.weights['species_prior']      * np.array(idf_scores) +
            self.weights['source_preference']  * np.array(source_scores)
        )
        
        # Create mapping by file_id to handle index mismatch after MMR
        idx_by_id = {c.file_id: i for i, c in enumerate(candidates)}
        rel_by_id = {c.file_id: float(rel[idx_by_id[c.file_id]]) for c in candidates}
        idf_by_id = {c.file_id: float(idf_scores[idx_by_id[c.file_id]]) for c in candidates}
        src_by_id = {c.file_id: float(source_scores[idx_by_id[c.file_id]]) for c in candidates}
        
        # Apply MMR re-ranking if diversity is promoted
        if promote_diversity:
            if emb_getter is None:
                raise ValueError("emb_getter is required when promote_diversity=True")
            # Use the combined relevance scores for MMR
            reranked = self.mmr_rerank(candidates, rel, emb_getter, lam=0.7, start_at=3, k=len(candidates))
            # Update scores based on the original candidate indices
            for c in reranked:
                i = idx_by_id[c.file_id]
                c.final_score = rel_by_id[c.file_id]
                c.frequency_score = idf_by_id[c.file_id]
                c.source_score = src_by_id[c.file_id]
                c.diversity_score = 1.0  # MMR handles diversity explicitly
        else:
            # If not promoting diversity, compute final scores and sort by them
            for i, candidate in enumerate(candidates):
                candidate.frequency_score = idf_scores[i]
                candidate.source_score = source_scores[i]
                candidate.diversity_score = diversity_scores[i]
                
                # Compute weighted final score
                candidate.final_score = (
                    self.weights['vector_similarity'] * vector_scores[i] +
                    self.weights['species_prior'] * idf_scores[i] +
                    self.weights['source_preference'] * source_scores[i] +
                    self.weights['diversity'] * diversity_scores[i]
                )
            reranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        
        logger.info(f"Re-ranking complete. Top candidate: {reranked[0].species} (score: {reranked[0].final_score:.4f})")
        
        return reranked
    
    def search(self,
              search_function: Callable,
              query_features: np.ndarray,
              db: Session,
              top_k: int = 10,
              prefer_source: str = "lab",
              promote_diversity: bool = True,
              emb_getter: Optional[Callable[[int], np.ndarray]] = None) -> List[SearchCandidate]:
        """
        Perform complete two-stage search
         
        Args:
            search_function: Vector search function
            query_features: Query feature vector
            db: Database session
            top_k: Number of results to return
            prefer_source: Preferred image source ("lab", "field", or None/"auto" to infer)
            promote_diversity: Whether to promote diversity
            emb_getter: Function to get embedding for a file_id (required when promote_diversity=True)
             
        Returns:
            Top-K re-ranked candidates
        """
        # Stage 1: Retrieve candidates
        file_ids, distances = self.stage1_retrieval(search_function, query_features, top_k)
         
        if not file_ids:
            return []
         
        # Convert to SearchCandidate objects
        candidates = []
        vector_scores = self.to_similarities(distances)
         
        for file_id, distance, vector_score in zip(file_ids, distances, vector_scores):
            # Get image metadata from database
            image = db.query(LeafImage).filter(LeafImage.file_id == file_id).first()
             
            if image:
                candidate = SearchCandidate(
                    file_id=int(image.file_id),  # type: ignore
                    distance=distance,
                    species=str(image.species),
                    source=str(image.source),
                    image_path=str(image.image_path),
                    segmented_path=str(image.segmented_path) if image.segmented_path is not None else None,
                    milvus_id=int(image.milvus_id) if image.milvus_id is not None else None,  # type: ignore
                    faiss_id=int(image.faiss_id) if image.faiss_id is not None else None,  # type: ignore
                    vector_score=vector_score
                )
                candidates.append(candidate)
         
        # Stage 2: Re-rank candidates
        # Use provided prefer_source or infer if needed
        pref = prefer_source
        if pref not in {"lab", "field"}:
            pref = self.infer_query_domain(query_features, [c.source for c in candidates[:min(5, len(candidates))]])
        
        # Use provided emb_getter or raise if needed
        emb_get = emb_getter
        if promote_diversity and emb_get is None:
            raise ValueError("emb_getter is required when promote_diversity=True")
        
        reranked = self.stage2_reranking(
            candidates,
            query_features,  # Pass query_features for domain inference
            db,
            prefer_source=pref,  # Use inferred domain as prefer_source
            promote_diversity=promote_diversity,
            emb_getter=emb_get
        )
         
        # Return top-K
        return reranked[:top_k]


class AdaptiveReRanker(TwoStageSearchEngine):
    """
    Adaptive re-ranker that adjusts weights based on query characteristics
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def detect_query_type(self, query_features: np.ndarray) -> str:
        """
        Detect query type to adjust re-ranking strategy
        
        Args:
            query_features: Query feature vector
            
        Returns:
            Query type: 'specific', 'general', or 'ambiguous'
        """
        # Analyze feature distribution
        feature_std = np.std(query_features)
        feature_max = np.max(np.abs(query_features))
        
        # High std with high max values = specific query
        if feature_std > 0.15 and feature_max > 2.0:
            return 'specific'
        # Low std = general query
        elif feature_std < 0.08:
            return 'general'
        else:
            return 'ambiguous'
    
    def adjust_weights(self, query_type: str) -> Dict[str, float]:
        """
        Adjust re-ranking weights based on query type
        
        Args:
            query_type: Type of query
            
        Returns:
            Adjusted weights
        """
        if query_type == 'specific':
            # For specific queries, trust vector similarity more
            weights = {
                'vector_similarity': 0.80,
                'species_prior': 0.05,
                'source_preference': 0.10,
                'diversity': 0.05
            }
        elif query_type == 'general':
            # For general queries, promote diversity
            weights = {
                'vector_similarity': 0.65,
                'species_prior': 0.05,
                'source_preference': 0.15,
                'diversity': 0.15
            }
        else:  # ambiguous
            # Balanced weights
            weights = {
                'vector_similarity': 0.70,
                'species_prior': 0.05,
                'source_preference': 0.15,
                'diversity': 0.10
            }
        
        logger.info(f"Adjusted weights for {query_type} query: {weights}")
        return weights
    
    def search(self,
              search_function: Callable,
              query_features: np.ndarray,
              db: Session,
              top_k: int = 10,
              prefer_source: str = "lab",
              promote_diversity: bool = True,
              emb_getter: Optional[Callable[[int], np.ndarray]] = None) -> List[SearchCandidate]:
        """
        Adaptive search with automatic weight adjustment

        Args:
            search_function: Vector search function
            query_features: Query feature vector
            db: Database session
            top_k: Number of results to return
            prefer_source: Preferred image source ("lab", "field", or None/"auto" to infer)
            promote_diversity: Whether to promote diversity
            emb_getter: Function to get embedding for a file_id (required when promote_diversity=True)
             
        Returns:
            Top-K re-ranked candidates
        """
        # Detect query type
        query_type = self.detect_query_type(query_features)
        
        # Adjust weights
        original_weights = self.weights.copy()
        self.weights = self.adjust_weights(query_type)
        
        # Perform search
        results = super().search(search_function, query_features, db, top_k, prefer_source, promote_diversity, emb_getter)
        
        # Restore original weights
        self.weights = original_weights
        
        return results


# Global instances
_two_stage_engine = None
_adaptive_engine = None


def get_two_stage_engine(adaptive: bool = False) -> TwoStageSearchEngine:
    """
    Get or create global two-stage search engine
    
    Args:
        adaptive: Whether to use adaptive re-ranker
        
    Returns:
        TwoStageSearchEngine instance
    """
    global _two_stage_engine, _adaptive_engine
    
    if adaptive:
        if _adaptive_engine is None:
            _adaptive_engine = AdaptiveReRanker()
        return _adaptive_engine
    else:
        if _two_stage_engine is None:
            _two_stage_engine = TwoStageSearchEngine()
        return _two_stage_engine

